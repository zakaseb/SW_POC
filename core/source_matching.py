"""
source_matching.py
------------------
Utilities to trace each extracted requirement back to its source text.

For every generated requirement we want to record:
  * source_chunk    : the full input chunk the requirement was extracted from
  * source_sentence : the single sentence inside that chunk that is most
                      semantically similar to the requirement's Description
  * conf_score      : the similarity score of that best-matching sentence

Similarity is computed with the locally cached sentence-transformers model
(cosine similarity over sentence embeddings). If that model cannot be loaded
(e.g. offline assets missing), we fall back to a deterministic lexical
similarity so the feature still produces sensible, reproducible output.
"""

from __future__ import annotations

import re
import threading
from difflib import SequenceMatcher
from typing import Callable, List, Optional, Sequence, Tuple

from .config import HF_LOCAL_FILES_ONLY, MODEL_CACHE_DIR, TOKENIZER_LOCAL_PATH, TOKENIZER_MODEL_NAME
from .logger_config import get_logger

logger = get_logger(__name__)

# Split on sentence-ending punctuation followed by whitespace. Chunks are first
# broken on line boundaries so bullet points / table rows become their own
# candidate sentences even without terminal punctuation.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?;])\s+")

_model = None
_model_tried = False
_model_lock = threading.Lock()


def split_into_sentences(text: str) -> List[str]:
    """Break ``text`` into candidate sentences (line-aware)."""
    if not text or not text.strip():
        return []
    sentences: List[str] = []
    for line in text.replace("\r", "\n").split("\n"):
        line = line.strip()
        if not line:
            continue
        for piece in _SENTENCE_SPLIT_RE.split(line):
            piece = piece.strip()
            if piece:
                sentences.append(piece)
    return sentences


def get_similarity_model():
    """Lazily load and cache the local sentence-transformers model.

    Returns ``None`` if the model cannot be loaded so callers can fall back to
    the lexical scorer. The result is memoised (including the ``None`` case) to
    avoid repeated load attempts inside a job.
    """
    global _model, _model_tried
    with _model_lock:
        if _model_tried:
            return _model
        _model_tried = True
        try:
            from sentence_transformers import SentenceTransformer

            source = (
                str(TOKENIZER_LOCAL_PATH)
                if TOKENIZER_LOCAL_PATH and TOKENIZER_LOCAL_PATH.exists()
                else TOKENIZER_MODEL_NAME
            )
            _model = SentenceTransformer(
                source,
                cache_folder=str(MODEL_CACHE_DIR),
                local_files_only=HF_LOCAL_FILES_ONLY,
            )
            logger.info("Loaded sentence-similarity model from: %s", source)
        except Exception as exc:  # noqa: BLE001 - fall back gracefully
            logger.warning(
                "Could not load sentence-similarity model (%s); "
                "falling back to lexical similarity.",
                exc,
            )
            _model = None
        return _model


def _lexical_scores(description: str, sentences: Sequence[str]) -> List[float]:
    """Deterministic offline fallback: normalized string similarity ratio."""
    desc = (description or "").lower()
    return [SequenceMatcher(None, desc, (s or "").lower()).ratio() for s in sentences]


def _embedding_scores(description: str, sentences: Sequence[str]) -> Optional[List[float]]:
    """Cosine similarity of embeddings; returns None if the model is unavailable."""
    model = get_similarity_model()
    if model is None:
        return None
    try:
        from sentence_transformers import util

        embeddings = model.encode(
            [description, *sentences],
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        query = embeddings[0]
        candidates = embeddings[1:]
        cos = util.cos_sim(query, candidates)[0]
        return [float(v) for v in cos]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Embedding similarity failed (%s); using lexical fallback.", exc)
        return None


def best_source_sentence(
    description: str,
    chunk_text: str,
    scorer: Optional[Callable[[str, Sequence[str]], Sequence[float]]] = None,
) -> Tuple[str, float]:
    """Return the sentence in ``chunk_text`` most similar to ``description``.

    Args:
        description: The extracted requirement's description text.
        chunk_text: The full source chunk the requirement came from.
        scorer: Optional callable ``(description, sentences) -> scores`` used to
            override similarity computation (primarily for testing). When not
            provided, embedding similarity is attempted first, then a lexical
            fallback.

    Returns:
        ``(best_sentence, score)`` where ``score`` is rounded to 4 decimals.
        Falls back to ``(chunk_text, 0.0)`` shaped values when no sentences are
        found so the output never contains a blank source sentence for a
        non-empty chunk.
    """
    sentences = split_into_sentences(chunk_text)
    if not sentences:
        return ("", 0.0)

    if scorer is not None:
        scores = list(scorer(description or "", sentences))
    else:
        scores = _embedding_scores(description or "", sentences)
        if scores is None:
            scores = _lexical_scores(description or "", sentences)

    if not scores:
        return (sentences[0], 0.0)

    best_idx = max(range(len(sentences)), key=lambda i: scores[i])
    return (sentences[best_idx], round(float(scores[best_idx]), 4))
