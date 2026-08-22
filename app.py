"""
HF Spaces entrypoint. Gradio's Audio component records mic input natively -
this satisfies the "speak the question, real voice-to-text input, not
typed" requirement without a custom frontend.

Env vars needed (set as HF Spaces secrets, not in code):
  SARVAM_API_KEY  - dashboard.sarvam.ai (free tier)
  GROQ_API_KEY    - console.groq.com (free tier)

Requires data/index_*.faiss and data/chunks_*.pkl to already be built and
committed to the Space (run src/data_prep.py + src/index_build.py locally
first, then push the data/ folder alongside the code).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import gradio as gr
from harness import run_pipeline

CSS = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700;800&display=swap');

:root {
  --hhgoa-green: #1a5c3e;
  --hhgoa-green-dark: #0f3d29;
  --hhgoa-pink: #e8177d;
  --hhgoa-cream: #faf3e0;
}

.gradio-container {
  font-family: 'JetBrains Mono', 'Courier New', monospace !important;
  background: var(--hhgoa-cream) !important;
}

h1, h2, h3 {
  color: var(--hhgoa-green-dark) !important;
  font-weight: 800 !important;
}

button.primary {
  background: var(--hhgoa-pink) !important;
  border: none !important;
  color: white !important;
  font-weight: 700 !important;
  border-radius: 10px !important;
}

button.secondary {
  background: var(--hhgoa-green) !important;
  border: none !important;
  color: white !important;
  border-radius: 10px !important;
  font-weight: 700 !important;
}
"""


def handle_query(audio, typed_text):
    if audio:
        result = run_pipeline(audio_path=audio)
    elif typed_text and typed_text.strip():
        result = run_pipeline(query_text=typed_text.strip())
    else:
        return "Record a question or type one.", "", ""

    transcript = result.query_text or "(no transcript)"
    answer = result.answer

    context_lines = []
    for c in result.contexts[:3]:
        context_lines.append(f"[{c['strategy']} | score={c['score']:.3f}] {c['text'][:200]}...")
    context_display = "\n\n".join(context_lines) if context_lines else "(no context retrieved)"

    timing = result.timing
    meta = (
        f"status: {result.status}"
        + (f" ({result.block_reason})" if result.block_reason else "")
        + (f"\nERROR: {result.error}" if result.error else "")
        + f"\nretrieval: {timing.retrieval_ms:.0f}ms | generation: {timing.generation_ms:.0f}ms"
        + f" | total: {timing.total_ms:.0f}ms"
    )

    return f"{transcript}\n\n---\n\n{answer}\n\n---\n{meta}", context_display, ""


def run_benchmark_from_space(n_queries):
    """
    Runs the latency benchmark from wherever this app is actually deployed
    (HF Spaces cloud, not the developer's laptop) so P50/P70/P100 reflect
    real cloud-to-cloud latency to Groq, not local connection quality.
    Local dev-machine benchmarks measure network RTT to the developer's ISP,
    not the pipeline's actual latency characteristics - this button exists
    specifically to get a representative number for the submission.
    """
    import json
    import statistics
    from pathlib import Path as _Path

    data_dir = _Path(__file__).parent / "data"
    with open(data_dir / "eval_set.json") as f:
        eval_set = json.load(f)

    n = min(int(n_queries), len(eval_set))
    queries = eval_set[:n]

    retrieval_lat, generation_lat, total_lat = [], [], []
    statuses = {}
    sample_errors = []

    for row in queries:
        result = run_pipeline(query_text=row["query"])
        retrieval_lat.append(result.timing.retrieval_ms)
        if result.timing.generation_ms:
            generation_lat.append(result.timing.generation_ms)
        total_lat.append(result.timing.total_ms)
        statuses[result.status] = statuses.get(result.status, 0) + 1
        if result.error and len(sample_errors) < 3:
            sample_errors.append(result.error)

    def pct(values, p):
        if not values:
            return 0.0
        s = sorted(values)
        idx = min(int(len(s) * p / 100), len(s) - 1)
        return s[idx]

    lines = [f"Ran {n} queries (cloud-to-cloud, deployed environment)\n"]
    for name, vals in [("retrieval", retrieval_lat), ("generation", generation_lat), ("end-to-end", total_lat)]:
        lines.append(f"{name:12s}  P50={pct(vals,50):7.1f}ms  P70={pct(vals,70):7.1f}ms  "
                      f"P100={pct(vals,100):7.1f}ms  mean={statistics.mean(vals) if vals else 0:7.1f}ms  n={len(vals)}")
    lines.append(f"\nStatus: {statuses}")
    if sample_errors:
        lines.append("\nSample errors (first 3):")
        for e in sample_errors:
            lines.append(f"  - {e}")
    under_200 = sum(1 for t in total_lat if t < 200)
    lines.append(f"{under_200}/{len(total_lat)} ({100*under_200/len(total_lat):.1f}%) under 200ms end-to-end.")

    return "\n".join(lines)


with gr.Blocks(css=CSS, title="HH Goa 2026 — Voice RAG") as demo:
    gr.Markdown("**TASK #2 · #RAGInGoa**\n# 🎙️ Voice-Enabled RAG — HH Goa 2026")
    gr.Markdown("Speak a question (or type one to skip STT). Answers are grounded strictly "
                "in retrieved context — if the pipeline isn't confident, it says so instead of guessing.")

    with gr.Row():
        audio_in = gr.Audio(sources=["microphone"], type="filepath", label="Ask by voice")
        text_in = gr.Textbox(label="...or type your question", placeholder="What was the impact of X?")

    submit_btn = gr.Button("Ask", variant="primary")

    output_box = gr.Textbox(label="Transcript + Answer", lines=10)
    context_box = gr.Textbox(label="Retrieved context (top 3)", lines=8)

    submit_btn.click(fn=handle_query, inputs=[audio_in, text_in], outputs=[output_box, context_box, text_in])

    gr.Markdown("---\n### Latency benchmark (for submission - run this from the deployed Space, not locally)")
    with gr.Row():
        n_input = gr.Number(value=50, label="Number of queries", precision=0)
        bench_btn = gr.Button("Run benchmark")
    bench_output = gr.Textbox(label="P50 / P70 / P100 results", lines=10)
    bench_btn.click(fn=run_benchmark_from_space, inputs=[n_input], outputs=[bench_output])

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
