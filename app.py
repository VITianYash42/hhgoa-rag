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
:root {
    --hh-sand: #f4ecd8;
    --hh-paper: #fff8e9;
    --hh-ink: #132617;
    --hh-green: #2e6f49;
    --hh-green-deep: #18422d;
    --hh-pink: #e73579;
    --hh-cyan: #20bfc4;
    --hh-amber: #ffb83f;
    --hh-border: #cbb990;
    --hh-muted: #5b694f;
}

.gradio-container {
    font-family: 'Cascadia Mono', 'IBM Plex Mono', 'Consolas', monospace !important;
    color: var(--hh-ink) !important;
    max-width: none !important;
    min-height: 100vh !important;
    margin: 0 !important;
    border: 0 !important;
    border-radius: 0 !important;
    padding: 20px 12px !important;
    background:
        radial-gradient(circle at 9% 10%, rgba(32, 191, 196, 0.17) 0, rgba(32, 191, 196, 0) 30%),
        radial-gradient(circle at 90% 90%, rgba(231, 53, 121, 0.14) 0, rgba(231, 53, 121, 0) 30%),
        linear-gradient(160deg, #fff8e9 0%, #f4ecd8 57%, #efdfbe 100%) !important;
    position: relative;
}

.gradio-container .main.fillable.app {
    max-width: 100% !important;
    width: 100% !important;
    margin: 0 !important;
    padding: 10px 8px 24px !important;
}

.gradio-container,
.gradio-container p,
.gradio-container span,
.gradio-container label,
.gradio-container h1,
.gradio-container h2,
.gradio-container h3,
.gradio-container h4,
.gradio-container h5,
.gradio-container h6 {
    color: var(--hh-ink) !important;
}

.hh-hero {
    border: 2px solid var(--hh-ink);
    border-radius: 18px;
    padding: 18px 20px;
    margin-bottom: 16px;
    background: linear-gradient(125deg, rgba(22, 71, 47, 0.94), rgba(47, 124, 79, 0.94));
    color: #fff8eb;
    animation: rise-in 500ms ease-out;
}

.hh-hero,
.hh-hero * {
    color: #fff8eb !important;
}

.hh-kicker {
    font-size: 12px;
    letter-spacing: 1.3px;
    text-transform: uppercase;
    color: #ffd57b !important;
    margin-bottom: 8px;
}

.hh-title {
    margin: 0;
    font-family: 'Bahnschrift', 'Segoe UI', sans-serif;
    font-weight: 700;
    letter-spacing: 0.2px;
    font-size: clamp(1.35rem, 4vw, 2rem);
    line-height: 1.16;
}

.hh-subtitle {
    margin-top: 8px;
    margin-bottom: 0;
    opacity: 0.93;
    font-size: 0.92rem;
}

.hh-strip {
    margin-top: 12px;
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
}

.hh-pill {
    border: 1px solid rgba(253, 248, 237, 0.45);
    border-radius: 999px;
    padding: 6px 12px;
    font-size: 12px;
    background: rgba(253, 248, 237, 0.09);
}

.hh-note {
    border-left: 4px solid var(--hh-cyan);
    border-radius: 10px;
    background: rgba(32, 191, 196, 0.12);
    padding: 10px 12px;
    margin: 12px 2px 16px;
    font-size: 0.86rem;
    color: #12301f !important;
}

.hh-note * {
    color: #12301f !important;
}

.hh-input-card,
.hh-output-card {
    border: 1.5px solid var(--hh-border) !important;
    border-radius: 14px !important;
    background: rgba(255, 248, 233, 0.92) !important;
    box-shadow: 0 8px 28px rgba(24, 66, 45, 0.08) !important;
}

.hh-input-card {
    animation: rise-in 560ms ease-out;
}

.hh-output-card {
    animation: rise-in 660ms ease-out;
}

.hh-ask-btn button,
.hh-bench-btn button {
    border-radius: 12px !important;
    border: 1.5px solid var(--hh-ink) !important;
    color: #fff !important;
    font-weight: 700 !important;
    letter-spacing: 0.35px;
    transition: transform 130ms ease, box-shadow 130ms ease, filter 130ms ease;
}

.hh-ask-btn button {
    background: linear-gradient(90deg, var(--hh-pink), #ff5c88) !important;
    box-shadow: 0 10px 22px rgba(231, 53, 121, 0.25) !important;
}

.hh-bench-btn button {
    background: linear-gradient(90deg, var(--hh-green-deep), var(--hh-green)) !important;
    box-shadow: 0 10px 22px rgba(22, 71, 47, 0.24) !important;
}

.hh-ask-btn button:hover,
.hh-bench-btn button:hover {
    transform: translateY(-1px);
    filter: brightness(1.03);
}

.hh-ask-btn button:active,
.hh-bench-btn button:active {
    transform: translateY(0);
}

.hh-section-title {
    margin-top: 8px;
    margin-bottom: 8px;
    font-family: 'Bahnschrift', 'Segoe UI', sans-serif;
    font-size: 1.1rem;
    color: var(--hh-ink);
}

.hh-footer {
    margin-top: 10px;
    padding: 10px 12px;
    border: 1px dashed var(--hh-border);
    border-radius: 10px;
    font-size: 0.81rem;
    color: #21452f;
    background: rgba(255, 255, 255, 0.42);
}

.hh-input-card label,
.hh-output-card label,
.hh-input-card .wrap,
.hh-output-card .wrap,
.hh-input-card .label-wrap,
.hh-output-card .label-wrap,
.hh-input-card .block-title,
.hh-output-card .block-title {
    color: #1a3725 !important;
}

.hh-input-card input,
.hh-input-card textarea,
.hh-output-card input,
.hh-output-card textarea,
.hh-input-card select,
.hh-output-card select {
    background: #fffaf0 !important;
    color: #132617 !important;
    border: 1px solid #bca983 !important;
}

.hh-input-card input::placeholder,
.hh-input-card textarea::placeholder,
.hh-output-card input::placeholder,
.hh-output-card textarea::placeholder {
    color: #586b52 !important;
    opacity: 1 !important;
}

.hh-input-card button {
    color: #fff8ea !important;
}

footer {
    background: transparent !important;
    color: #2a5138 !important;
}

footer * {
    color: #2a5138 !important;
}

@keyframes rise-in {
    from {
        opacity: 0;
        transform: translateY(8px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@media (max-width: 768px) {
    .gradio-container {
        padding: 14px 12px !important;
    }

    .hh-hero {
        padding: 14px;
    }

    .hh-title {
        font-size: 1.35rem;
    }
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


with gr.Blocks(title="HH Goa 2026 — Voice RAG") as demo:
    gr.HTML(
        """
        <section class="hh-hero">
          <div class="hh-kicker">Task #2 · #RAGInGoa</div>
          <h1 class="hh-title">Voice-Enabled RAG<br/>for Hacker House Goa 2026</h1>
          <p class="hh-subtitle">Speak. Retrieve. Ground. Ship. Built for low-latency answers with clear guardrails.</p>
          <div class="hh-strip">
            <span class="hh-pill">Real mic input</span>
            <span class="hh-pill">Engineered retrieval</span>
            <span class="hh-pill">Benchmarkable latency</span>
          </div>
        </section>
        """
    )

    gr.HTML(
        """
        <div class="hh-note">
          Less noise. More signal. Ask by voice or text, and the app returns grounded output from retrieved context.
        </div>
        """
    )

    with gr.Row():
        audio_in = gr.Audio(
            sources=["microphone"],
            type="filepath",
            label="Ask by voice",
            elem_classes=["hh-input-card"],
        )
        text_in = gr.Textbox(
            label="...or type your question",
            placeholder="What was the impact of X?",
            elem_classes=["hh-input-card"],
        )

    submit_btn = gr.Button("Ask", variant="primary", elem_classes=["hh-ask-btn"])

    output_box = gr.Textbox(label="Transcript + Answer", lines=10, elem_classes=["hh-output-card"])
    context_box = gr.Textbox(label="Retrieved context (top 3)", lines=8, elem_classes=["hh-output-card"])

    submit_btn.click(fn=handle_query, inputs=[audio_in, text_in], outputs=[output_box, context_box, text_in])

    gr.HTML("<h3 class='hh-section-title'>Latency Benchmark</h3>")
    with gr.Row():
        n_input = gr.Number(value=50, label="Number of queries", precision=0, elem_classes=["hh-input-card"])
        bench_btn = gr.Button("Run benchmark", elem_classes=["hh-bench-btn"])
    bench_output = gr.Textbox(label="P50 / P70 / P100 results", lines=10, elem_classes=["hh-output-card"])
    bench_btn.click(fn=run_benchmark_from_space, inputs=[n_input], outputs=[bench_output])

    gr.HTML(
        """
        <div class="hh-footer">
          HH Goa 2026 vibe: intentional, fast, and grounded. Benchmark from deployed Space for real cloud-to-cloud latency.
        </div>
        """
    )

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port, css=CSS)
