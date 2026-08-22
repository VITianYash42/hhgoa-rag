"""
Answer generation via Groq (Llama 3.1 8B instant) - free tier, no card,
picked specifically because it's the only free option fast enough to have
a shot at the sub-200ms full-pipeline target. Get a free key at
https://console.groq.com/keys and set GROQ_API_KEY.
"""
import os
import time
from groq import Groq

MODEL = "openai/gpt-oss-20b"  # llama-3.1-8b-instant was deprecated by Groq (June 2026); this is their recommended replacement, same speed/cost tier

SYSTEM_PROMPT = """You are a grounded QA assistant. You MUST answer using ONLY the provided context passages.
Rules:
- If the answer is not clearly supported by the context, say exactly: "I don't have enough information in the retrieved context to answer that."
- Do not use outside knowledge, even if you know the answer.
- Keep answers under 3 sentences.
- Do not speculate or hedge with "might be" — either the context supports a claim or you say you can't answer."""

_client = None


def get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set. Get a free key at console.groq.com")
        _client = Groq(api_key=api_key)
    return _client


def generate_answer(query: str, contexts: list[dict], timeout: float = 6.0) -> dict:
    """
    contexts: list of {"text": ..., "score": ..., "strategy": ...} from retrieval.retrieve()
    Returns {"answer": str, "latency_ms": float, "used_context_ids": [...]}
    """
    client = get_client()
    context_block = "\n\n".join(
        f"[Passage {i+1} | strategy={c['strategy']} | score={c['score']:.3f}]\n{c['text']}"
        for i, c in enumerate(contexts)
    )
    user_prompt = f"Context:\n{context_block}\n\nQuestion: {query}\n\nAnswer using only the context above."

    start = time.perf_counter()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=200,
        timeout=timeout,
    )
    latency_ms = (time.perf_counter() - start) * 1000
    answer = resp.choices[0].message.content.strip()
    return {"answer": answer, "latency_ms": latency_ms}
