"""
Loads ai4bharat/IndicMSMARCO (Hindi split) - a small, properly-packaged
companion dataset from the same research group/paper as the originally
linked ai4bharat/MSMARCO-XI.

WHY THIS DATASET INSTEAD OF THE LITERALLY-LINKED ONE:
MSMARCO-XI's per-language files are ~3.5GB+ and appear to be written as a
single (or very few) Parquet row group(s), meaning even a LIMIT-based
partial read via duckdb/pyarrow still requires fetching/decompressing most
of the file before returning any rows - confirmed by a 30+ minute hang with
zero throughput on a constrained connection. HuggingFace's own dataset
viewer for MSMARCO-XI independently fails with the same symptom ("Job
manager crashed while running this job (missing heartbeats)"), which is
external confirmation this isn't a local network issue.

ai4bharat/IndicMSMARCO is published by the same authors, cites the same
paper (IndicRAGSuite, arXiv:2506.01615), and is explicitly described as
built for "RAG Evaluation: Test retrieval-augmented generation systems" -
i.e. it's arguably a *more* apt source for this exact task than the raw
11.4M-row training dump. It's ~13MB total, auto-converts to Parquet
cleanly, and loads via the standard `datasets` library without incident.

Schema differs from MSMARCO-XI: each row is a single (query, passage,
relevance_score) pair rather than a query with a list of passages +
is_selected flags. relevance_score substitutes for is_selected as the
chunking pipeline's "gold passage" signal (thresholded below).
"""
import json
import hashlib
from pathlib import Path
from datasets import load_dataset

SAMPLE_SIZE = None  # None = use the full split (dataset is already small, ~1k rows/language)
LANG = "hi"
RELEVANCE_GOLD_THRESHOLD = 0.5  # relevance_score >= this counts as "gold" (metadata-aware chunking signal)
OUT_DIR = Path(__file__).parent.parent / "data"


def passage_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def main():
    OUT_DIR.mkdir(exist_ok=True)
    print(f"Loading ai4bharat/IndicMSMARCO ({LANG} split) - small dataset, standard load...")
    ds = load_dataset("ai4bharat/IndicMSMARCO", LANG, split="train")
    print(f"Loaded {len(ds)} rows.")

    rows = ds if SAMPLE_SIZE is None else ds.select(range(min(SAMPLE_SIZE, len(ds))))

    corpus = {}  # passage_id -> {"id", "text", "is_selected_anywhere"}
    eval_set = []  # one entry per unique query_id, may reference multiple gold passages

    queries_seen = {}  # query_id -> eval_set index, since IndicMSMARCO is one row per (query, passage) pair

    for row in rows:
        qid = row.get("query_id")
        query_text = (row.get("query") or "").strip()
        passage_text = (row.get("passage") or "").strip()
        relevance = row.get("relevance_score", 0)
        is_gold = float(relevance) >= RELEVANCE_GOLD_THRESHOLD if relevance is not None else False

        if not passage_text:
            continue

        pid = passage_id(passage_text)
        already_gold = corpus.get(pid, {}).get("is_selected_anywhere", False)
        corpus[pid] = {
            "id": pid,
            "text": passage_text,
            "is_selected_anywhere": is_gold or already_gold,
        }

        if not query_text:
            continue

        if qid in queries_seen:
            entry = eval_set[queries_seen[qid]]
            if is_gold and pid not in entry["gold_passage_ids"]:
                entry["gold_passage_ids"].append(pid)
        else:
            queries_seen[qid] = len(eval_set)
            eval_set.append({
                "query_id": qid,
                "query": query_text,
                "answer": None,  # IndicMSMARCO doesn't ship a reference answer per query - generation is evaluated on grounding, not answer-string match
                "gold_passage_ids": [pid] if is_gold else [],
                "query_type": "UNKNOWN",
            })

    corpus_list = list(corpus.values())
    print(f"Deduplicated corpus: {len(corpus_list)} passages.")
    print(f"Eval set: {len(eval_set)} unique queries.")

    with open(OUT_DIR / "corpus.json", "w") as f:
        json.dump(corpus_list, f)
    with open(OUT_DIR / "eval_set.json", "w") as f:
        json.dump(eval_set, f)

    print(f"Saved to {OUT_DIR}/corpus.json and {OUT_DIR}/eval_set.json")


if __name__ == "__main__":
    main()
