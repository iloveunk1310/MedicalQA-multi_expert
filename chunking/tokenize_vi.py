"""Vietnamese word segmentation and ViHealthBERT subword counting utilities."""

from functools import lru_cache
from typing import Any

import underthesea

from .normalize import normalize_unicode


@lru_cache(maxsize=65536)
def _segment_line(line: str) -> str:
    return underthesea.word_tokenize(line, format="text")


def segment_vi(text: str) -> str:
    """Word-segment Vietnamese text with underthesea while preserving line breaks."""
    if not text:
        return ""
    nfc = normalize_unicode(text)
    if not nfc.strip():
        return ""

    segmented_lines: list[str] = []
    for line in nfc.split("\n"):
        stripped = line.strip()
        if stripped:
            segmented_lines.append(_segment_line(stripped))
        else:
            segmented_lines.append("")
    return "\n".join(segmented_lines)


def count_vi_words(text: str) -> int:
    """Count Vietnamese word tokens (underthesea segmented) in text."""
    seg = segment_vi(text)
    return len(seg.split()) if seg else 0


def count_vi_words_from_segmented(seg_text: str) -> int:
    """Count whitespace-delimited tokens in already word-segmented text."""
    return len(seg_text.split()) if seg_text else 0


class ViHealthBertCounter:
    """Wrapper around ViHealthBERT tokenizer for accurate subword length measurement."""

    def __init__(
        self,
        model_name: str = "demdecuong/vihealthbert-base-word",
        tokenizer: Any = None,
    ) -> None:
        self.model_name = model_name
        self._seg_len_cache: dict[tuple[str, bool], int] = {}
        if tokenizer is not None:
            self.tokenizer = tokenizer
            return

        try:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load tokenizer '{model_name}': {exc}. "
                "Check internet connectivity or local HuggingFace cache, or pass "
                "'--child-length words' explicitly for offline word-based fallback."
            ) from exc

    def count_subwords_from_seg(self, seg_text: str, add_special_tokens: bool = True) -> int:
        """Count subword tokens on already Vietnamese-word-segmented text."""
        key = (seg_text, add_special_tokens)
        if len(seg_text) <= 4096 and key in self._seg_len_cache:
            return self._seg_len_cache[key]

        token_ids = self.tokenizer.encode(seg_text, add_special_tokens=add_special_tokens)
        length = len(token_ids)
        if len(seg_text) <= 4096:
            if len(self._seg_len_cache) >= 65536:
                self._seg_len_cache.clear()
            self._seg_len_cache[key] = length
        return length

    def count_subwords(self, text: str, add_special_tokens: bool = True) -> int:
        """Word-segment text with underthesea and count ViHealthBERT subword tokens."""
        return self.count_subwords_from_seg(
            segment_vi(text),
            add_special_tokens=add_special_tokens,
        )

