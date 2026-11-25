"""Configuration dataclass for Contextual Parent-Child KB chunking."""

from dataclasses import dataclass


@dataclass
class ChunkConfig:
    """Configuration parameters for parent-child knowledge base chunking."""

    parent_max_words: int = 512
    child_max_subwords: int = 256
    child_overlap_subwords: int = 48
    child_max_words: int = 150
    child_overlap_words: int = 30
    child_length_mode: str = "subword"
    tokenizer_name: str = "demdecuong/vihealthbert-base-word"
    guidelines_path: str = "data/clinical_guidelines_unified.jsonl"
    drugs_path: str = "data/drug_monographs_unified.jsonl"
    out_dir: str = "data/kb"
    max_heading_chars: int = 60
    workers: int = 1

    def __post_init__(self) -> None:
        if self.child_length_mode not in ("subword", "words"):
            raise ValueError(
                f"Invalid child_length_mode={self.child_length_mode!r}; expected 'subword' or 'words'."
            )
        if self.parent_max_words <= 0:
            raise ValueError("parent_max_words must be > 0")
        if self.child_max_subwords <= 0 or self.child_max_words <= 0:
            raise ValueError("child max length limits must be > 0")
        if self.child_overlap_subwords < 0 or self.child_overlap_words < 0:
            raise ValueError("child overlap values must be >= 0")
        if self.workers < 1:
            raise ValueError("workers must be >= 1")

