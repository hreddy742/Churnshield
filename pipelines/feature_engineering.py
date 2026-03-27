"""Feature engineering utilities for customer note sentiment extraction."""

from __future__ import annotations

import os
from threading import Lock

try:
    from transformers import pipeline
except ModuleNotFoundError:  # pragma: no cover
    pipeline = None


_SENTIMENT_PIPELINE = None
_SENTIMENT_PIPELINE_LOCK = Lock()
_POSITIVE_WORDS = {"helpful", "smooth", "reliable", "easy", "improved", "appreciate", "professional", "stable"}
_NEGATIVE_WORDS = {
    "frustrated",
    "billing",
    "outage",
    "complaint",
    "dissatisfaction",
    "competitor",
    "disputes",
    "interruptions",
}


def _lexicon_sentiment_scores(texts: list[str]) -> list[float]:
    scores: list[float] = []
    for text in texts:
        tokens = {token.strip(".,!?").lower() for token in text.split()}
        pos_hits = len(tokens.intersection(_POSITIVE_WORDS))
        neg_hits = len(tokens.intersection(_NEGATIVE_WORDS))
        raw_score = 0.5 + 0.12 * pos_hits - 0.12 * neg_hits
        scores.append(float(min(1.0, max(0.0, raw_score))))
    return scores


def _get_sentiment_pipeline():
    """Lazily initialize and return the shared local sentiment pipeline."""

    global _SENTIMENT_PIPELINE

    if pipeline is None or os.getenv("CHURNGUARD_ENABLE_TRANSFORMERS", "").lower() not in {"1", "true", "yes"}:
        return None

    if _SENTIMENT_PIPELINE is None:
        with _SENTIMENT_PIPELINE_LOCK:
            if _SENTIMENT_PIPELINE is None:
                try:
                    _SENTIMENT_PIPELINE = pipeline(
                        "sentiment-analysis",
                        model="distilbert-base-uncased-finetuned-sst-2-english",
                        device=-1,
                    )
                except Exception:
                    _SENTIMENT_PIPELINE = None
    return _SENTIMENT_PIPELINE


def extract_sentiment_score(texts: list[str]) -> list[float]:
    """Return positive-sentiment probabilities in the 0-1 range for input texts."""

    if not texts:
        return []

    cleaned_texts = [text if isinstance(text, str) and text.strip() else "" for text in texts]
    sentiment_pipe = _get_sentiment_pipeline()
    if sentiment_pipe is None:
        return _lexicon_sentiment_scores(cleaned_texts)

    predictions = sentiment_pipe(
        cleaned_texts,
        batch_size=32,
        truncation=True,
        max_length=512,
    )

    try:
        scores: list[float] = []
        for prediction in predictions:
            label = str(prediction["label"]).upper()
            score = float(prediction["score"])
            scores.append(score if label == "POSITIVE" else 1.0 - score)
        return scores
    except Exception:
        return _lexicon_sentiment_scores(cleaned_texts)
