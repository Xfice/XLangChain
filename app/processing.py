"""Data processing helpers for public X dataset analysis."""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

URL_PATTERN = re.compile(r"https?://\S+")
MENTION_PATTERN = re.compile(r"@\w+")
HASHTAG_PATTERN = re.compile(r"#(\w+)")
WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_text(value: str) -> str:
    """Normalize tweet-like text for downstream analysis."""
    text = URL_PATTERN.sub("", value)
    text = MENTION_PATTERN.sub("", text)
    text = WHITESPACE_PATTERN.sub(" ", text)
    return text.strip()


def extract_hashtags(texts: Iterable[str], limit: int = 10) -> list[tuple[str, int]]:
    """Extract most common hashtags from a set of texts."""
    hashtags: list[str] = []
    for text in texts:
        hashtags.extend(HASHTAG_PATTERN.findall(text.lower()))
    return Counter(hashtags).most_common(limit)


def map_sentiment(raw_value: object) -> str:
    """Map common dataset sentiment formats to labels."""
    if raw_value is None:
        return "unknown"
    value = str(raw_value).strip().lower()
    if value in {"4", "1", "positive", "pos"}:
        return "positive"
    if value in {"0", "-1", "negative", "neg"}:
        return "negative"
    if value in {"2", "neutral"}:
        return "neutral"
    return "unknown"
