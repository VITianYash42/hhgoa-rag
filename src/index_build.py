"""
Builds a TF-IDF vector index per chunking strategy - NOT a neural embedding
model. Deliberate trade-off: sentence-transformers pulls in PyTorch, a
2-3GB download, which is infeasible on poor internet. TF-IDF is still real
vector-space retrieval (sparse vectors, cosine similarity) and needs only
scikit-learn (~150MB total including scipy/numpy) - no GPU/neural stack.

Cost of this trade-off: TF-IDF matches on word overlap, not semantic
meaning, so it won't catch heavily paraphrased questions as well as a
neural embedding model would. State this explicitly in the submission -
it's a defensible engineering call under a real bandwidth constraint, not
a hidden shortcut.

Run after data_prep.py. Usage: python index_build.py
"""
import json
import pickle
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer

from chunking import build_all_chunks

DATA_DIR = Path(__file__).parent.parent / "data"


def main():
    with open(DATA_DIR / "corpus.json") as f:
        corpus = json.load(f)

    print(f"Building chunks (3 strategies) over {len(corpus)} passages...")
    all_chunks = build_all_chunks(corpus)
    for name, chunks in all_chunks.items():
        print(f"  {name}: {len(chunks)} chunks")

    for strategy, chunks in all_chunks.items():
        texts = [c["text"] for c in chunks]
        print(f"Fitting TF-IDF vectorizer for strategy '{strategy}' ({len(texts)} chunks)...")

        vectorizer = TfidfVectorizer(
            max_features=50000,
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
        )
        matrix = vectorizer.fit_transform(texts)

        with open(DATA_DIR / f"tfidf_{strategy}.pkl", "wb") as f:
            pickle.dump({"vectorizer": vectorizer, "matrix": matrix}, f)
        with open(DATA_DIR / f"chunks_{strategy}.pkl", "wb") as f:
            pickle.dump(chunks, f)
        print(f"  Saved tfidf_{strategy}.pkl and chunks_{strategy}.pkl")

    print("Done. Indices ready for retrieval.py")


if __name__ == "__main__":
    main()
