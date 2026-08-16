"""
Guardrails the harness runs before and after the model call. These are
intentionally rule-based / cheap (regex, thresholds, string checks) rather
than a second LLM call for two reasons: (1) an extra LLM guardrail call
would blow the 200ms budget on its own, (2) cheap deterministic checks are
easier to defend in a benchmarking writeup than "trust me, GPT said it's fine".

Three checks:
1. input_guardrail   - blocks empty/garbage transcripts and a basic unsafe-
                        content keyword filter before we spend a retrieval
                        + generation call on it.
2. retrieval_guardrail - if top retrieval score is below a similarity
                        threshold, we refuse rather than force an ungrounded
                        answer from weak context. This is the "knows when
                        not to answer" requirement.
3. grounding_check    - lightweight post-hoc check: does the generated
                        answer share enough lexical overlap with the
                        retrieved context, or did the model explicitly say
                        it can't answer? Flags likely hallucination for the
                        harness to react to (retry once, or return the
                        refusal message instead of a hallucinated answer).
"""
import re

UNSAFE_PATTERNS = [
    r"\bhow to (make|build) a (bomb|weapon)\b",
    r"\bself[\s-]?harm\b",
    r"\bkill (myself|yourself)\b",
]

MIN_RETRIEVAL_SCORE = 0.12          # TF-IDF cosine sim below this -> refuse, context too weak
                                     # (lower than a neural-embedding threshold would be -
                                     # TF-IDF cosine scores run smaller since they're sparse
                                     # word-overlap vectors, not dense semantic ones. Tune
                                     # this against your own eval_set after benchmarking.)
MIN_GROUNDING_OVERLAP = 0.08        # fraction of answer content-words present in context
REFUSAL_TEXT = "I don't have enough information in the retrieved context to answer that."


def input_guardrail(text: str) -> dict:
    """Returns {'blocked': bool, 'reason': str|None}"""
    if not text or not text.strip():
        return {"blocked": True, "reason": "empty_transcript"}
    if len(text.strip()) < 3:
        return {"blocked": True, "reason": "transcript_too_short"}
    lowered = text.lower()
    for pattern in UNSAFE_PATTERNS:
        if re.search(pattern, lowered):
            return {"blocked": True, "reason": "unsafe_content"}
    return {"blocked": False, "reason": None}


def retrieval_guardrail(contexts: list[dict]) -> dict:
    """Returns {'blocked': bool, 'reason': str|None}. contexts sorted desc by score."""
    if not contexts:
        return {"blocked": True, "reason": "no_context_retrieved"}
    top_score = contexts[0]["score"]
    if top_score < MIN_RETRIEVAL_SCORE:
        return {"blocked": True, "reason": f"low_retrieval_confidence ({top_score:.3f} < {MIN_RETRIEVAL_SCORE})"}
    return {"blocked": False, "reason": None}


def _content_words(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z]{4,}", text.lower())
    stop = {"this", "that", "with", "from", "have", "does", "were", "will", "your", "context"}
    return {w for w in words if w not in stop}


def grounding_check(answer: str, contexts: list[dict]) -> dict:
    """Returns {'grounded': bool, 'overlap_ratio': float, 'is_explicit_refusal': bool}"""
    if answer.strip() == REFUSAL_TEXT:
        return {"grounded": True, "overlap_ratio": None, "is_explicit_refusal": True}

    answer_words = _content_words(answer)
    if not answer_words:
        return {"grounded": False, "overlap_ratio": 0.0, "is_explicit_refusal": False}

    context_words = set()
    for c in contexts:
        context_words |= _content_words(c["text"])

    overlap = len(answer_words & context_words) / len(answer_words)
    return {
        "grounded": overlap >= MIN_GROUNDING_OVERLAP,
        "overlap_ratio": overlap,
        "is_explicit_refusal": False,
    }
