"""
Measures latency across a real batch of eval queries (not one lucky run,
per the task's explicit requirement) and reports P50/P70/P100 for each
pipeline stage plus end-to-end.

Runs text-only (query_text, skipping STT) by default so the numbers isolate
retrieval + generation - the parts fully under this system's control. STT
latency is third-party (Sarvam) and reported separately from a small sample
of real audio calls, since bundling a variable third-party network call
into the same distribution would muddy which stage is the bottleneck.

Usage: python -m src.benchmark [--n 50] [--with-stt audio_dir/]
"""
import json
import argparse
import statistics
from pathlib import Path

from harness import run_pipeline

DATA_DIR = Path(__file__).parent.parent / "data"


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(int(len(s) * p / 100), len(s) - 1)
    return s[idx]


def report(name: str, values: list[float]):
    if not values:
        print(f"  {name}: no data")
        return
    print(f"  {name:12s}  P50={percentile(values,50):7.1f}ms  "
          f"P70={percentile(values,70):7.1f}ms  "
          f"P100={percentile(values,100):7.1f}ms  "
          f"mean={statistics.mean(values):7.1f}ms  n={len(values)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="number of eval queries to run")
    args = ap.parse_args()

    with open(DATA_DIR / "eval_set.json") as f:
        eval_set = json.load(f)

    queries = eval_set[:args.n]
    print(f"Running {len(queries)} queries through the pipeline (text-only, STT excluded)...\n")

    retrieval_lat, generation_lat, total_lat = [], [], []
    statuses = {}

    for i, row in enumerate(queries):
        result = run_pipeline(query_text=row["query"])
        retrieval_lat.append(result.timing.retrieval_ms)
        if result.timing.generation_ms:
            generation_lat.append(result.timing.generation_ms)
        total_lat.append(result.timing.total_ms)
        statuses[result.status] = statuses.get(result.status, 0) + 1
        if (i + 1) % 10 == 0:
            print(f"  ...{i+1}/{len(queries)}")

    print("\n=== Latency (ms) ===")
    report("retrieval", retrieval_lat)
    report("generation", generation_lat)
    report("end-to-end", total_lat)

    print("\n=== Status breakdown ===")
    for status, count in statuses.items():
        print(f"  {status}: {count} ({100*count/len(queries):.1f}%)")

    under_200 = sum(1 for t in total_lat if t < 200)
    print(f"\n{under_200}/{len(total_lat)} queries ({100*under_200/len(total_lat):.1f}%) "
          f"completed end-to-end under 200ms (excluding STT).")

    out = {
        "n": len(queries),
        "retrieval_ms": {"p50": percentile(retrieval_lat, 50), "p70": percentile(retrieval_lat, 70), "p100": percentile(retrieval_lat, 100)},
        "generation_ms": {"p50": percentile(generation_lat, 50), "p70": percentile(generation_lat, 70), "p100": percentile(generation_lat, 100)},
        "end_to_end_ms": {"p50": percentile(total_lat, 50), "p70": percentile(total_lat, 70), "p100": percentile(total_lat, 100)},
        "status_breakdown": statuses,
    }
    with open(DATA_DIR / "benchmark_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {DATA_DIR}/benchmark_results.json")


if __name__ == "__main__":
    main()
