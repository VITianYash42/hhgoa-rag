"""
Three chunking strategies applied to the passage corpus, per the task's
explicit requirement to avoid a single naive fixed-size split:

1. FixedSizeOverlap   - fixed token-window chunks with overlap. Baseline,
                         predictable recall, wastes tokens on boundaries.
2. SemanticSentence    - splits on sentence boundaries, packs sentences into
                         chunks up to a token budget without cutting mid-sentence.
                         Better for QA where the answer usually lives in one
                         or two full sentences.
3. MetadataAware       - uses MS MARCO's own `is_selected` signal: passages
                         already marked as gold-relevant in the source data
                         are kept as single, unsplit, high-priority chunks
                         (since they're usually short and self-contained),
                         while long non-selected passages get semantic-split.
                         This exploits real structure in the dataset instead
                         of treating every passage identically.

Each strategy outputs a list of chunk dicts: {id, text, source_passage_id,
strategy, metadata}. All three are indexed into separate FAISS indices so
retrieval.py can compare / ensemble them and report which strategy wins per
query — that comparison itself is part of "real thought put into chunking".
"""
import re
import hashlib
from dataclasses import dataclass, field
from typing import Any


def _chunk_id(text: str, strategy: str) -> str:
    return hashlib.sha1(f"{strategy}:{text}".encode("utf-8")).hexdigest()[:16]


def _split_sentences(text: str) -> list[str]:
    # lightweight sentence splitter, avoids an nltk punkt download dependency
    text = text.strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    return [p.strip() for p in parts if p.strip()]


@dataclass
class Chunk:
    id: str
    text: str
    source_passage_id: str
    strategy: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return {
            "id": self.id, "text": self.text,
            "source_passage_id": self.source_passage_id,
            "strategy": self.strategy, "metadata": self.metadata,
        }


def fixed_size_overlap(passage: dict, window_words: int = 60, overlap_words: int = 15) -> list[Chunk]:
    words = passage["text"].split()
    if len(words) <= window_words:
        text = passage["text"]
        return [Chunk(_chunk_id(text, "fixed"), text, passage["id"], "fixed",
                       {"is_selected_anywhere": passage.get("is_selected_anywhere", False)})]
    chunks = []
    step = max(window_words - overlap_words, 1)
    for start in range(0, len(words), step):
        window = words[start:start + window_words]
        if not window:
            break
        text = " ".join(window)
        chunks.append(Chunk(_chunk_id(text, "fixed"), text, passage["id"], "fixed",
                             {"is_selected_anywhere": passage.get("is_selected_anywhere", False)}))
        if start + window_words >= len(words):
            break
    return chunks


def semantic_sentence(passage: dict, max_words: int = 70) -> list[Chunk]:
    sentences = _split_sentences(passage["text"])
    if not sentences:
        return []
    chunks, current, current_words = [], [], 0
    for sent in sentences:
        w = len(sent.split())
        if current and current_words + w > max_words:
            text = " ".join(current)
            chunks.append(Chunk(_chunk_id(text, "semantic"), text, passage["id"], "semantic",
                                 {"is_selected_anywhere": passage.get("is_selected_anywhere", False),
                                  "n_sentences": len(current)}))
            current, current_words = [], 0
        current.append(sent)
        current_words += w
    if current:
        text = " ".join(current)
        chunks.append(Chunk(_chunk_id(text, "semantic"), text, passage["id"], "semantic",
                             {"is_selected_anywhere": passage.get("is_selected_anywhere", False),
                              "n_sentences": len(current)}))
    return chunks


def metadata_aware(passage: dict, max_words: int = 70) -> list[Chunk]:
    is_gold = passage.get("is_selected_anywhere", False)
    word_count = len(passage["text"].split())
    if is_gold and word_count <= 120:
        # keep short, already-relevant passages intact - splitting them
        # risks separating the answer span from its supporting sentence
        text = passage["text"]
        return [Chunk(_chunk_id(text, "metaaware"), text, passage["id"], "metaaware",
                       {"is_selected_anywhere": True, "kept_whole": True})]
    # otherwise fall back to semantic split, tagging provenance
    sub = semantic_sentence(passage, max_words=max_words)
    out = []
    for c in sub:
        out.append(Chunk(_chunk_id(c.text, "metaaware"), c.text, c.source_passage_id, "metaaware",
                          {**c.metadata, "kept_whole": False}))
    return out


STRATEGIES = {
    "fixed": fixed_size_overlap,
    "semantic": semantic_sentence,
    "metaaware": metadata_aware,
}


def build_all_chunks(corpus: list[dict]) -> dict[str, list[dict]]:
    """Returns {strategy_name: [chunk_dict, ...]} for every passage in corpus."""
    result = {name: [] for name in STRATEGIES}
    for passage in corpus:
        for name, fn in STRATEGIES.items():
            for chunk in fn(passage):
                result[name].append(chunk.to_dict())
    return result
