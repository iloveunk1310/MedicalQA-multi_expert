"""Configuration and retrieval result dataclasses for RAG."""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class RAGConfig:
    """Configuration for dense indexing and retrieval over kb_children/kb_parents."""

    kb_dir: str = "data/kb"
    index_dir: str = "data/kb/index"
    encoder_model: str = "demdecuong/vihealthbert-base-word"
    pooling: str = "mean"
    max_length: int = 256
    batch_size: int = 64
    top_k: int = 3
    search_k: int = 30
    context_unit: str = "parent"
    query_mode: str = "question"
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.pooling not in ("mean", "cls"):
            raise ValueError(f"Invalid pooling={self.pooling!r}; expected 'mean' or 'cls'.")
        if self.context_unit not in ("parent", "child"):
            raise ValueError(
                f"Invalid context_unit={self.context_unit!r}; expected 'parent' or 'child'."
            )
        if self.query_mode not in ("question", "question_options"):
            raise ValueError(
                f"Invalid query_mode={self.query_mode!r}; expected 'question' or 'question_options'."
            )
        if self.max_length <= 0 or self.batch_size <= 0 or self.top_k <= 0 or self.search_k <= 0:
            raise ValueError("max_length, batch_size, top_k, and search_k must be > 0.")


@dataclass
class RetrievedChunk:
    """Single retrieved context unit (parent or child) with scores and provenance."""

    context_id: str
    parent_id: str
    child_ids: list[str]
    score: float
    text: str
    rerank_score: Optional[float] = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """Structured retrieval output for a single query, compatible with Hybrid/Reranker hooks."""

    query: str
    contexts: list[str]
    chunks: list[RetrievedChunk]
    retrieved_ids: list[str]
    retrieved_child_ids: list[str]
    top_retrieval_score: Optional[float] = None
    top_rerank_score: Optional[float] = None

    def format_context(self, sep: str = "\n\n---\n\n") -> str:
        """Join retrieved context strings into a single prompt-ready block."""
        return sep.join(self.contexts)
