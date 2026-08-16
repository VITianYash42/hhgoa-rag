"""
The orchestration layer the task asks for: "structured orchestration around
the model (tool calls, retries, structured input/output handling, error
recovery) rather than a single raw prompt-in, text-out call."

run_pipeline() is the single entrypoint the Gradio app and the benchmark
script both call. It wires: STT -> input guardrail -> retrieval ->
retrieval guardrail -> generation -> grounding check -> structured result.

Retry policy: each external call (STT, generation) gets up to MAX_RETRIES
attempts with short backoff, since both are network calls to third-party
APIs that can transiently fail. Retrieval is local (FAISS), so it doesn't
need retry - if it throws, that's a real bug, not a transient fault.
"""
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

from stt import transcribe, STTError
from retrieval import retrieve
from generate import generate_answer
from guardrails import input_guardrail, retrieval_guardrail, grounding_check, REFUSAL_TEXT

MAX_RETRIES = 2
RETRY_BACKOFF_S = 0.3


@dataclass
class StageTiming:
    stt_ms: float = 0.0
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0


@dataclass
class PipelineResult:
    query_text: str = ""
    answer: str = ""
    status: str = "ok"  # ok | blocked_input | blocked_retrieval | error
    block_reason: Optional[str] = None
    contexts: list = field(default_factory=list)
    grounding: Optional[dict] = None
    timing: StageTiming = field(default_factory=StageTiming)
    error: Optional[str] = None

    def to_dict(self):
        d = asdict(self)
        return d


def _with_retry(fn, *args, max_retries=MAX_RETRIES, **kwargs):
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 - intentionally broad, this is the retry boundary
            last_exc = e
            if attempt < max_retries:
                time.sleep(RETRY_BACKOFF_S * (attempt + 1))
            continue
    raise last_exc


def run_pipeline(audio_path: Optional[str] = None, query_text: Optional[str] = None,
                  top_k: int = 5) -> PipelineResult:
    """
    Entry point. Provide either audio_path (goes through STT) or query_text
    directly (skips STT - used by the benchmark script to isolate retrieval
    + generation latency from third-party STT variance).
    """
    result = PipelineResult()
    t_start = time.perf_counter()
    timing = StageTiming()

    # --- Stage 1: STT ---
    if audio_path and not query_text:
        try:
            stt_out = _with_retry(transcribe, audio_path)
            query_text = stt_out["text"]
            timing.stt_ms = stt_out["latency_ms"]
        except STTError as e:
            result.status = "error"
            result.error = f"STT failed after retries: {e}"
            timing.total_ms = (time.perf_counter() - t_start) * 1000
            result.timing = timing
            return result

    result.query_text = query_text or ""

    # --- Stage 2: input guardrail ---
    gr = input_guardrail(query_text or "")
    if gr["blocked"]:
        result.status = "blocked_input"
        result.block_reason = gr["reason"]
        result.answer = REFUSAL_TEXT
        timing.total_ms = (time.perf_counter() - t_start) * 1000
        result.timing = timing
        return result

    # --- Stage 3: retrieval ---
    t0 = time.perf_counter()
    try:
        contexts = retrieve(query_text, final_k=top_k)
    except Exception as e:  # noqa: BLE001
        result.status = "error"
        result.error = f"Retrieval failed: {e}"
        timing.total_ms = (time.perf_counter() - t_start) * 1000
        result.timing = timing
        return result
    timing.retrieval_ms = (time.perf_counter() - t0) * 1000
    result.contexts = contexts

    # --- Stage 4: retrieval guardrail ---
    rg = retrieval_guardrail(contexts)
    if rg["blocked"]:
        result.status = "blocked_retrieval"
        result.block_reason = rg["reason"]
        result.answer = REFUSAL_TEXT
        timing.total_ms = (time.perf_counter() - t_start) * 1000
        result.timing = timing
        return result

    # --- Stage 5: generation ---
    try:
        gen_out = _with_retry(generate_answer, query_text, contexts)
    except Exception as e:  # noqa: BLE001
        result.status = "error"
        result.error = f"Generation failed after retries: {e}"
        timing.total_ms = (time.perf_counter() - t_start) * 1000
        result.timing = timing
        return result
    timing.generation_ms = gen_out["latency_ms"]
    result.answer = gen_out["answer"]

    # --- Stage 6: grounding check (post-hoc, doesn't block, but flags) ---
    result.grounding = grounding_check(result.answer, contexts)
    if not result.grounding["grounded"]:
        result.answer = REFUSAL_TEXT
        result.status = "blocked_retrieval"
        result.block_reason = "post_hoc_grounding_check_failed"

    timing.total_ms = (time.perf_counter() - t_start) * 1000
    result.timing = timing
    return result
