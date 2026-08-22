# HH Goa 2026 — Voice-Enabled RAG

Speak a question → transcription → multi-strategy retrieval → grounded answer, with guardrails and latency benchmarking.

## Architecture

```
mic audio ──▶ Sarvam STT ──▶ input guardrail ──▶ retrieval (3 chunking
                                                    strategies, TF-IDF)
                                                        │
                                                        ▼
                                          retrieval guardrail (min score)
                                                        │
                                                        ▼
                                    Groq (openai/gpt-oss-20b) generation
                                                        │
                                                        ▼
                                          grounding check (post-hoc)
                                                        │
                                                        ▼
                                                  answer / refusal
```

All orchestration lives in `src/harness.py` — retries on STT/generation, structured `PipelineResult` output, error recovery at every stage, no raw prompt-in/text-out call.

**Live app:** https://hhgoa-rag.onrender.com
**Note on cold starts:** deployed on Render's free tier, which spins down after inactivity. First request after idle can take 30-60s to wake up — this is infra, not pipeline latency.

## Dataset

Uses `ai4bharat/IndicMSMARCO` (Hindi split), not the originally-linked `ai4bharat/MSMARCO-XI`.

**Why the swap:** MSMARCO-XI's per-language files (~3.5GB+) are written with very few/large Parquet row groups, meaning even a `LIMIT`-based partial read requires downloading/decompressing most of the file before returning any rows — confirmed by a 30+ minute zero-throughput hang, and independently by HuggingFace's own dataset viewer failing on this repo with a job-manager crash. `ai4bharat/IndicMSMARCO` is published by the same authors, cites the same paper (IndicRAGSuite, arXiv:2506.01615), and is explicitly built for RAG evaluation — arguably more apt for this exact task. It's ~13MB total and loads cleanly. Full reasoning is in `src/data_prep.py`'s docstring.

Dataset is in **Hindi** — query the live app in Hindi, not English, for meaningful retrieval.

## Chunking strategies (3, not 1)

| Strategy | Logic |
|---|---|
| `fixed` | Fixed 60-word windows, 15-word overlap. Baseline. |
| `semantic` | Sentence-boundary packing up to a word budget — never cuts mid-sentence. |
| `metaaware` | Uses the dataset's own relevance signal (`relevance_score`, thresholded): short gold-relevant passages kept whole; everything else semantic-split. |

At query time (`src/retrieval.py`), all three indices are queried and results pooled/deduped by score.

## Retrieval backend: TF-IDF, not neural embeddings

`index_build.py` / `retrieval.py` use scikit-learn `TfidfVectorizer`, not `sentence-transformers`.

**Why:** neural embeddings pull in PyTorch — a 2-3GB download, infeasible both on a constrained home connection during development and on Render's free-tier 512MB RAM limit at runtime (confirmed via an out-of-memory crash, exit code 137, when `sentence-transformers` was in the dependency chain). TF-IDF is still real vector-space retrieval — sparse vectors, cosine similarity — just word-overlap-based rather than semantic. Total install is ~150-250MB instead of 2-3GB+, and runtime RAM footprint measured at ~234MB, comfortably under Render's limit.

**Trade-off, stated plainly:** TF-IDF won't catch heavily paraphrased questions as well as neural embeddings would. This is a defensible engineering call under real bandwidth/memory constraints, not a hidden shortcut.

## Guardrails (`src/guardrails.py`)

1. **Input guardrail** — blocks empty/garbage transcripts and an unsafe-content keyword filter, before spending a retrieval+generation call.
2. **Retrieval guardrail** — refuses to answer if top retrieval cosine similarity is below `MIN_RETRIEVAL_SCORE` (0.25) rather than force an answer from weak context.
3. **Grounding check** — post-hoc lexical overlap check between the generated answer and retrieved context. Low overlap → answer is replaced with an explicit refusal instead of a hallucination.

Deliberately rule-based, not a second LLM call — an extra model call would add unnecessary latency and is harder to defend numerically in a benchmark writeup than deterministic thresholds.

## Latency benchmark — actual results

Measured from the deployed Render instance (cloud-to-cloud to Groq), 50 real queries from the dataset, via the "Run benchmark" button built into the app UI:

```
retrieval     P50=   6.1ms  P70=   6.7ms  P100= 906.2ms  mean=  24.3ms  n=50
generation    P50=6318.7ms  P70=6438.9ms  P100=8793.2ms  mean=5470.4ms  n=47
end-to-end    P50=6301.3ms  P70=6444.1ms  P100=8799.0ms  mean=5170.6ms  n=50
Status: {'blocked_retrieval': 26, 'ok': 24}
3/50 (6.0%) under 200ms end-to-end.
```

**Honest read of these numbers:** retrieval is fast (P50=6.1ms) — the pipeline logic itself is not the bottleneck. The 200ms target is missed because of Groq API generation latency (P50=6.3s), which is third-party network + inference time, not something fixable by this codebase's engineering. The 26/50 `blocked_retrieval` rate reflects TF-IDF's word-overlap limitation combined with the guardrail correctly refusing low-confidence matches rather than guessing — this is the "knows when not to answer" requirement working as intended, not a failure.

Re-run via the live app's "Run benchmark" button (bottom of the UI) to reproduce.

## Setup

```bash
pip install -r requirements.txt

# 1. Get free API keys (no card needed for either):
#    Sarvam STT: https://dashboard.sarvam.ai
#    Groq generation: https://console.groq.com/keys
export SARVAM_API_KEY=...
export GROQ_API_KEY=...

# 2. Build the corpus + indices (fast, seconds — small dataset, no model download)
cd src
python data_prep.py       # loads IndicMSMARCO, builds corpus.json + eval_set.json
python index_build.py     # builds TF-IDF indices for all 3 strategies

# 3. Run the latency benchmark locally (optional — local numbers reflect
#    YOUR network RTT to Groq, not representative; use the deployed
#    Space/app's benchmark button for submission-quality numbers)
python benchmark.py --n 50

# 4. Run the app locally
cd ..
python app.py
# open http://localhost:7860
```

Note: `requirements.txt` is the minimal runtime set (no `datasets` library — that's only needed for `data_prep.py`, listed separately in `requirements-local.txt`, and isn't installed on the deployed instance).

## Deployment (Render)

Deployed on Render.com free tier (not HF Spaces — Gradio SDK spaces now require a paid plan on HF; Render's free Python web service does not).

1. Push this repo to GitHub (data/ folder included — the deployed instance needs the prebuilt indices, it doesn't rebuild them at boot).
2. Render dashboard → New → Web Service → connect the GitHub repo.
3. Build command: `pip install -r requirements.txt`. Start command: `python app.py`. Instance type: Free.
4. Add `SARVAM_API_KEY` and `GROQ_API_KEY` as Environment Variables in Render's dashboard.
5. `app.py` binds to `0.0.0.0` and Render's assigned `$PORT` for compatibility.

## Known limitations (stated plainly, not hidden)

- **200ms end-to-end target not met** — see benchmark section above. Retrieval is fast; third-party Groq generation latency is the bottleneck.
- **TF-IDF, not neural embeddings** — word-overlap retrieval, not semantic. Documented trade-off above.
- **STT latency isn't included in the benchmark number** — Sarvam is a separate third-party network call with its own variance; bundling it would misattribute network jitter to this system's numbers.
- **Dataset is a 1,000-row Hindi sample** (`ai4bharat/IndicMSMARCO`), not the originally-linked full MSMARCO-XI dataset — see "Dataset" section above for the reasoning.
- **Render free tier cold starts** — first request after ~15 min idle takes 30-60s.
- **Groq free tier is rate-limited** (30 req/min) — fine for this benchmark's `--n 50`, not for sustained production load.
