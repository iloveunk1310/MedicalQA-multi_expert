"""Text and Markdown normalization helpers for Vietnamese medical KB."""

import re
import unicodedata
from typing import Any

_TABLE_LINE_RE = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_SEP_RE = re.compile(r"^\s*\|[\s:\-\|]+\|\s*$")
_MULTI_BLANK_RE = re.compile(r"\n{3,}")
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")

_SIDE_EFFECT_PRIORITY = ["thuong_gap", "it_gap", "hiem_gap", "chung", "huong_dan_xu_tri"]
_SIDE_EFFECT_LABELS = {
    "thuong_gap": ("Thường gặp", ("thường gặp",)),
    "it_gap": ("Ít gặp", ("ít gặp",)),
    "hiem_gap": ("Hiếm gặp", ("hiếm gặp",)),
    "chung": ("Tổng quan ADR", ("tổng quan adr", "chung:")),
    "huong_dan_xu_tri": ("Hướng dẫn xử trí ADR", ("hướng dẫn xử trí",)),
}


def normalize_unicode(text: str) -> str:
    """Normalize text to Unicode NFC form."""
    return unicodedata.normalize("NFC", text)


def is_table_line(line: str) -> bool:
    """Return True when a line looks like a Markdown pipe-table row."""
    return bool(_TABLE_LINE_RE.match(line))


def is_table_separator(line: str) -> bool:
    """Return True when a line is a Markdown table separator row (e.g. |---|---|)."""
    return bool(_TABLE_SEP_RE.match(line) and "-" in line)


def compact_markdown_table_line(line: str) -> str:
    """Compact excessive padding and normalize separators in a Markdown table line."""
    stripped = line.strip()
    if is_table_separator(stripped):
        inner = stripped.strip("|")
        n_cols = max(1, len(inner.split("|")))
        return "|" + "|".join(["---"] * n_cols) + "|"

    inner = stripped[1:-1] if stripped.startswith("|") and stripped.endswith("|") else stripped.strip("|")
    cells = [_MULTI_SPACE_RE.sub(" ", cell.strip()) for cell in inner.split("|")]
    return "| " + " | ".join(cells) + " |"


def flatten_side_effect(value: Any) -> str:
    """Flatten drug monograph side_effect dict into labeled paragraphs."""
    if value is None:
        return ""
    if isinstance(value, str):
        return normalize_unicode(value)
    if not isinstance(value, dict):
        return normalize_unicode(str(value))

    ordered_keys: list[str] = [k for k in _SIDE_EFFECT_PRIORITY if k in value]
    for k in value:
        if k not in ordered_keys:
            ordered_keys.append(k)

    parts: list[str] = []
    for key in ordered_keys:
        raw_val = value.get(key)
        if raw_val is None:
            continue
        text = normalize_unicode(str(raw_val)).strip()
        if not text:
            continue

        label_info = _SIDE_EFFECT_LABELS.get(key)
        if label_info is not None:
            label, prefixes = label_info
            lowered = text.lower()
            if not any(lowered.startswith(p) for p in prefixes):
                text = f"{label}:\n{text}"
        else:
            label = normalize_unicode(str(key)).strip()
            if label and not text.lower().startswith(label.lower()):
                text = f"{label}:\n{text}"
        parts.append(text)

    return "\n\n".join(parts)


def normalize_section_text(value: Any, section_key: str = "") -> str:
    """Normalize a raw section value (str or side_effect dict) into clean Markdown/text."""
    if value is None:
        return ""

    if section_key == "side_effect" or isinstance(value, dict):
        raw_text = flatten_side_effect(value)
    else:
        raw_text = normalize_unicode(str(value))

    raw_text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    if not raw_text.strip():
        return ""

    cleaned_lines: list[str] = []
    for line in raw_text.split("\n"):
        if is_table_line(line):
            cleaned_lines.append(compact_markdown_table_line(line))
        else:
            cleaned_lines.append(line.rstrip())

    merged = "\n".join(cleaned_lines)
    merged = _MULTI_BLANK_RE.sub("\n\n", merged)
    return merged.strip()
