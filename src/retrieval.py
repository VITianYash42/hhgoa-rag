"""
Loads all three per-strategy TF-IDF indices and exposes a single retrieve()
call that queries each, then merges results by cosine similarity score.
Multiple chunking strategies pooled at query time, not committed to one -
same design as the neural version, just swapped the embedding backend for
scikit-learn TF-IDF (see index_build.py for why).
"""
import pickle
from pathlib import Path
from functools import lru_cache

from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path(__file__).parent.parent / "data"


@lru_cache(maxsize=1)
def load_indices():
    indices = {}
    for strategy in ("fixed", "semantic", "metaaware"):
        tfidf_path = DATA_DIR / f"tfidf_{strategy}.pkl"
        chunks_path = DATA_DIR / f"chunks_{strategy}.pkl"
        if not tfidf_path.exists():
            continue
        with open(tfidf_path, "rb") as f:
            tfidf_data = pickle.load(f)
        with open(chunks_path, "rb") as f:
            chunks = pickle.load(f)
        indices[strategy] = (tfidf_data["vectorizer"], tfidf_data["matrix"], chunks)
    return indices


def retrieve(query: str, top_k_per_strategy: int = 5, final_k: int = 5) -> list[dict]:
    """
    Query every strategy's TF-IDF index, pool results, sort by cosine
    similarity, return top final_k.
    Each result: {text, score, strategy, chunk_id, is_selected_anywhere}
    """
    indices = load_indices()
    if not indices:
        raise RuntimeError("No indices found. Run `python index_build.py` first.")

    pooled = []
    for strategy, (vectorizer, matrix, chunks) in indices.items():
        q_vec = vectorizer.transform([query])
        sims = cosine_similarity(q_vec, matrix)[0]
        top_idx = sims.argsort()[::-1][:top_k_per_strategy]
        for idx in top_idx:
            score = float(sims[idx])
            if score <= 0:
                continue
            chunk = chunks[idx]
            pooled.append({
                "text": chunk["text"],
                "score": score,
                "strategy": strategy,
                "chunk_id": chunk["id"],
                "is_selected_anywhere": chunk.get("metadata", {}).get("is_selected_anywhere", False),
            })

    seen_text = {}
    for r in pooled:
        key = r["text"][:120]
        if key not in seen_text or r["score"] > seen_text[key]["score"]:
            seen_text[key] = r

    ranked = sorted(seen_text.values(), key=lambda r: r["score"], reverse=True)
    return ranked[:final_k]
