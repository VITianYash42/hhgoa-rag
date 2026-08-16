"""
Embeds every chunk (per strategy) with a local sentence-transformers model
(free, CPU-friendly, no API cost) and builds one FAISS index per strategy.
Run after data_prep.py.

Usage: python -m src.index_build
"""
import json
import pickle
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from chunking import build_all_chunks

DATA_DIR = Path(__file__).parent.parent / "data"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # small, fast, free, CPU-friendly


def main():
    with open(DATA_DIR / "corpus.json") as f:
        corpus = json.load(f)

    print(f"Building chunks (3 strategies) over {len(corpus)} passages...")
    all_chunks = build_all_chunks(corpus)
    for name, chunks in all_chunks.items():
        print(f"  {name}: {len(chunks)} chunks")

    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    for strategy, chunks in all_chunks.items():
        texts = [c["text"] for c in chunks]
        print(f"Embedding {len(texts)} chunks for strategy '{strategy}'...")
        embeddings = model.encode(
            texts, batch_size=64, show_progress_bar=True,
            convert_to_numpy=True, normalize_embeddings=True,
        ).astype("float32")

        index = faiss.IndexFlatIP(embeddings.shape[1])  # cosine sim via normalized IP
        index.add(embeddings)

        faiss.write_index(index, str(DATA_DIR / f"index_{strategy}.faiss"))
        with open(DATA_DIR / f"chunks_{strategy}.pkl", "wb") as f:
            pickle.dump(chunks, f)
        print(f"  Saved index_{strategy}.faiss and chunks_{strategy}.pkl")

    print("Done. Indices ready for retrieval.py")


if __name__ == "__main__":
    main()
