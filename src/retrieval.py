"""
Loads all three per-strategy FAISS indices and exposes a single retrieve()
call that queries each, then merges results by score. This is the
"multiple chunking strategies, not one naive split" requirement made
concrete: at query time we don't commit to one strategy, we pool candidates
from all three and let score decide, tagging which strategy each hit came
from (useful for the latency/quality writeup).
"""
import pickle
from pathlib import Path
from functools import lru_cache

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).parent.parent / "data"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_model = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


@lru_cache(maxsize=1)
def load_indices():
    indices = {}
    for strategy in ("fixed", "semantic", "metaaware"):
        index_path = DATA_DIR / f"index_{strategy}.faiss"
        chunks_path = DATA_DIR / f"chunks_{strategy}.pkl"
        if not index_path.exists():
            continue
        index = faiss.read_index(str(index_path))
        with open(chunks_path, "rb") as f:
            chunks = pickle.load(f)
        indices[strategy] = (index, chunks)
    return indices


def retrieve(query: str, top_k_per_strategy: int = 5, final_k: int = 5) -> list[dict]:
    """
    Query every strategy's index, pool results, sort by score, return top final_k.
    Each result: {text, score, strategy, chunk_id, is_selected_anywhere}
    """
    indices = load_indices()
    if not indices:
        raise RuntimeError("No indices found. Run `python -m src.index_build` first.")

    model = get_model()
    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")

    pooled = []
    for strategy, (index, chunks) in indices.items():
        scores, idxs = index.search(q_emb, top_k_per_strategy)
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0 or idx >= len(chunks):
                continue
            chunk = chunks[idx]
            pooled.append({
                "text": chunk["text"],
                "score": float(score),
                "strategy": strategy,
                "chunk_id": chunk["id"],
                "is_selected_anywhere": chunk.get("metadata", {}).get("is_selected_anywhere", False),
            })

    # dedupe near-identical text across strategies (metaaware often overlaps semantic),
    # keep the highest-scoring copy
    seen_text = {}
    for r in pooled:
        key = r["text"][:120]
        if key not in seen_text or r["score"] > seen_text[key]["score"]:
            seen_text[key] = r

    ranked = sorted(seen_text.values(), key=lambda r: r["score"], reverse=True)
    return ranked[:final_k]
