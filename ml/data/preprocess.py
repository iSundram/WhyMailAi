"""
ml/data/preprocess.py
---------------------
Clean, normalise, and anonymise raw email records before training.

Operations
----------
* lower-case normalisation
* URL replacement  →  <URL>
* email address replacement  →  <EMAIL>
* phone number replacement  →  <PHONE>
* HTML tag stripping
* whitespace normalisation
* deduplication (exact-match on cleaned text)
* length filtering (drop near-empty records)
"""

from __future__ import annotations

import html
import re
import unicodedata
from pathlib import Path

from loguru import logger


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

_URL_RE = re.compile(
    r"https?://[^\s<>\"']+|www\.[^\s<>\"']+"
    r"|ftp://[^\s<>\"']+",
    re.IGNORECASE,
)
_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
_PHONE_RE = re.compile(
    r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b"
)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MULTI_SPACE_RE = re.compile(r"[ \t]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


# ---------------------------------------------------------------------------
# Core cleaning function
# ---------------------------------------------------------------------------

def clean_text(
    text: str,
    *,
    lowercase: bool = True,
    replace_urls: bool = True,
    replace_emails: bool = True,
    replace_phones: bool = True,
    strip_html: bool = True,
    max_chars: int = 4096,
) -> str:
    """Return a cleaned version of *text*."""
    if not isinstance(text, str):
        text = str(text)

    # Decode HTML entities (e.g. &amp; → &)
    text = html.unescape(text)

    # Strip HTML tags
    if strip_html:
        text = _HTML_TAG_RE.sub(" ", text)

    # Unicode normalisation (NFKC)
    text = unicodedata.normalize("NFKC", text)

    # Remove control characters except newline / tab
    text = "".join(
        ch for ch in text if ch == "\n" or ch == "\t" or not unicodedata.category(ch).startswith("C")
    )

    # Replace sensitive tokens
    if replace_emails:
        text = _EMAIL_RE.sub("<EMAIL>", text)
    if replace_urls:
        text = _URL_RE.sub("<URL>", text)
    if replace_phones:
        text = _PHONE_RE.sub("<PHONE>", text)

    # Normalise whitespace
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    text = text.strip()

    if lowercase:
        text = text.lower()

    # Truncate
    if len(text) > max_chars:
        text = text[:max_chars]

    return text


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def preprocess_records(
    records: list[dict],
    *,
    min_chars: int = 10,
    deduplicate: bool = True,
    text_key: str = "text",
    **clean_kwargs,
) -> list[dict]:
    """
    Clean a list of record dicts in-place (returns new list).

    Parameters
    ----------
    records     : list of dicts with at least a ``text_key`` field
    min_chars   : minimum cleaned-text length; shorter records are dropped
    deduplicate : if True, drop exact-duplicate cleaned texts
    text_key    : name of the text field (default ``"text"``)
    **clean_kwargs : forwarded to :func:`clean_text`

    Returns
    -------
    list of cleaned dicts
    """
    out: list[dict] = []
    seen: set[str] = set()
    dropped_short = 0
    dropped_dup = 0

    for rec in records:
        raw = rec.get(text_key, "")
        cleaned = clean_text(raw, **clean_kwargs)
        if len(cleaned) < min_chars:
            dropped_short += 1
            continue
        if deduplicate:
            if cleaned in seen:
                dropped_dup += 1
                continue
            seen.add(cleaned)
        new_rec = dict(rec)
        new_rec[text_key] = cleaned
        out.append(new_rec)

    logger.info(
        f"Preprocess: {len(records):,} → {len(out):,} records "
        f"(dropped {dropped_short:,} short, {dropped_dup:,} duplicates)"
    )
    return out


def preprocess_summarization_records(
    records: list[dict],
    *,
    min_dialogue_chars: int = 50,
    min_summary_chars: int = 10,
    max_dialogue_chars: int = 2048,
    max_summary_chars: int = 512,
    deduplicate: bool = True,
) -> list[dict]:
    """
    Special preprocessing for summarisation records which have both
    ``text`` (the dialogue/thread) and ``summary`` fields.
    """
    out: list[dict] = []
    seen: set[str] = set()
    dropped = 0

    for rec in records:
        dialogue = clean_text(
            rec.get("text", ""),
            lowercase=False,
            max_chars=max_dialogue_chars,
        )
        summary = clean_text(
            rec.get("summary", ""),
            lowercase=False,
            max_chars=max_summary_chars,
        )
        if len(dialogue) < min_dialogue_chars or len(summary) < min_summary_chars:
            dropped += 1
            continue
        if deduplicate and dialogue in seen:
            dropped += 1
            continue
        seen.add(dialogue)
        new_rec = dict(rec)
        new_rec["text"] = dialogue
        new_rec["summary"] = summary
        out.append(new_rec)

    logger.info(
        f"Summarization preprocess: {len(records):,} → {len(out):,} records "
        f"(dropped {dropped:,})"
    )
    return out
