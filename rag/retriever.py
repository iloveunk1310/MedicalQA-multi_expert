"""Retrievers and parent-child context resolution for Naive RAG and future Hybrid RAG."""

from abc import ABC, abstractmethod
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import faiss

from .config import RAGConfig, RetrievalResult, RetrievedChunk
from .encoder import ViHealthBertEncoder


def _load_jsonl_map(path: Path, id_key: str) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"KB file not found: {path}")
    records: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            rec = json.loads(stripped)
            records[rec[id_key]] = rec
    return records


def resolve_candidates_to_contexts(
    candidates: Sequence[tuple[str, float]],
    children_by_id: Mapping[str, Mapping[str, Any]],
    parents_by_id: Mapping[str, Mapping[str, Any]],
    top_k: int = 3,
    context_unit: str = "parent",
    reranker: Any = None,
    query: str = "",
) -> RetrievalResult:
    """Resolve child (child_id, score) candidates into deduplicated parent or child contexts."""
    if context_unit not in ("parent", "child"):
        raise ValueError(f"Invalid context_unit={context_unit!r}; expected 'parent' or 'child'.")

    chunks: list[RetrievedChunk] = []

    if context_unit == "parent":
        parent_best: dict[str, float] = {}
        parent_children: dict[str, list[tuple[str, float]]] = {}
        for cid, score in candidates:
            child_rec = children_by_id.get(cid)
            if child_rec is None:
                continue
            pid = child_rec["parent_id"]
            if pid not in parents_by_id:
                continue
            sc = float(score)
            if pid not in parent_best or sc > parent_best[pid]:
                parent_best[pid] = sc
            parent_children.setdefault(pid, []).append((cid, sc))

        sorted_pids = sorted(parent_best.keys(), key=lambda p: parent_best[p], reverse=True)
        limit_pool = sorted_pids if reranker is not None else sorted_pids[:top_k]
        for pid in limit_pool:
            p_rec = parents_by_id[pid]
            seen_cids: set[str] = set()
            ordered_cids: list[str] = []
            for cid, _ in sorted(parent_children[pid], key=lambda x: x[1], reverse=True):
                if cid not in seen_cids:
                    seen_cids.add(cid)
                    ordered_cids.append(cid)
            chunks.append(
                RetrievedChunk(
                    context_id=pid,
                    parent_id=pid,
                    child_ids=ordered_cids,
                    score=parent_best[pid],
                    text=p_rec["text_for_llm"],
                    meta={
                        "doc_id": p_rec.get("doc_id"),
                        "doc_type": p_rec.get("doc_type"),
                        "specialty": p_rec.get("specialty"),
                        "entity_name": p_rec.get("entity_name"),
                        "section_key": p_rec.get("section_key"),
                        "section_name": p_rec.get("section_name"),
                    },
                )
            )
    else:
        child_best: dict[str, float] = {}
        for cid, score in candidates:
            if cid not in children_by_id:
                continue
            sc = float(score)
            if cid not in child_best or sc > child_best[cid]:
                child_best[cid] = sc

        sorted_cids = sorted(child_best.keys(), key=lambda c: child_best[c], reverse=True)
        limit_pool = sorted_cids if reranker is not None else sorted_cids[:top_k]
        for cid in limit_pool:
            c_rec = children_by_id[cid]
            chunks.append(
                RetrievedChunk(
                    context_id=cid,
                    parent_id=c_rec["parent_id"],
                    child_ids=[cid],
                    score=child_best[cid],
                    text=c_rec["text_for_dense"],
                    meta={
                        "doc_id": c_rec.get("doc_id"),
                        "doc_type": c_rec.get("doc_type"),
                        "specialty": c_rec.get("specialty"),
                        "entity_name": c_rec.get("entity_name"),
                        "section_key": c_rec.get("section_key"),
                    },
                )
            )

    if reranker is not None and chunks:
        chunks = list(reranker.rerank(query, chunks, top_k=top_k))
    else:
        chunks = chunks[:top_k]

    contexts = [c.text for c in chunks]
    retrieved_ids = [c.context_id for c in chunks]
    retrieved_child_ids: list[str] = []
    seen_child: set[str] = set()
    for ch in chunks:
        for cid in ch.child_ids:
            if cid not in seen_child:
                seen_child.add(cid)
                retrieved_child_ids.append(cid)

    top_retrieval_score = chunks[0].score if chunks else None
    top_rerank_score = chunks[0].rerank_score if chunks else None

    return RetrievalResult(
        query=query,
        contexts=contexts,
        chunks=chunks,
        retrieved_ids=retrieved_ids,
        retrieved_child_ids=retrieved_child_ids,
        top_retrieval_score=top_retrieval_score,
        top_rerank_score=top_rerank_score,
    )


class BaseRetriever(ABC):
    """Abstract retriever interface shared by DenseRetriever and future HybridRetriever."""

    @abstractmethod
    def retrieve_batch(
        self,
        queries: Sequence[str],
        top_k: Optional[int] = None,
    ) -> list[RetrievalResult]:
        """Retrieve contexts for a batch of queries in input order."""

    def retrieve(self, query: str, top_k: Optional[int] = None) -> RetrievalResult:
        """Retrieve contexts for a single query."""
        return self.retrieve_batch([query], top_k=top_k)[0]

    def unload(self) -> None:
        """Optional hook to release encoder/reranker resources."""


class DenseRetriever(BaseRetriever):
    """FAISS IndexFlatIP dense retriever over ViHealthBERT child embeddings."""

    def __init__(
        self,
        config: RAGConfig,
        encoder: Optional[ViHealthBertEncoder] = None,
        reranker: Any = None,
    ) -> None:
        self.config = config
        self.reranker = reranker

        index_dir = Path(config.index_dir)
        kb_dir = Path(config.kb_dir)

        manifest_path = index_dir / "index_manifest.json"
        ids_path = index_dir / "child_ids.json"
        faiss_path = index_dir / "children_dense.faiss"

        if not manifest_path.exists() or not ids_path.exists() or not faiss_path.exists():
            raise FileNotFoundError(
                f"Incomplete dense index in '{index_dir}'. "
                "Run 'python -m rag.build_index' first."
            )

        with manifest_path.open("r", encoding="utf-8") as f:
            self.manifest: dict[str, Any] = json.load(f)
        with ids_path.open("r", encoding="utf-8") as f:
            self.child_ids: list[str] = json.load(f)
        self.index = faiss.read_index(str(faiss_path))

        self.children_by_id = _load_jsonl_map(kb_dir / "kb_children.jsonl", "child_id")
        self.parents_by_id = _load_jsonl_map(kb_dir / "kb_parents.jsonl", "parent_id")

        n_manifest = self.manifest.get("n_children")
        if (
            len(self.child_ids) != self.index.ntotal
            or n_manifest != len(self.child_ids)
            or len(self.children_by_id) != n_manifest
        ):
            raise ValueError(
                f"Index manifest mismatch: index.ntotal={self.index.ntotal}, "
                f"child_ids={len(self.child_ids)}, manifest.n_children={n_manifest}, "
                f"kb_children.jsonl={len(self.children_by_id)}. "
                "Please rebuild the index with 'python -m rag.build_index'."
            )

        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = ViHealthBertEncoder(
                model_name=self.manifest.get("encoder_model", config.encoder_model),
                pooling=self.manifest.get("pooling", config.pooling),
                max_length=self.manifest.get("max_length", config.max_length),
                device=config.device,
            )

    def retrieve_batch(
        self,
        queries: Sequence[str],
        top_k: Optional[int] = None,
    ) -> list[RetrievalResult]:
        if not queries:
            return []

        k = top_k if top_k is not None else self.config.top_k
        q_embs = self.encoder.encode_queries(queries, batch_size=self.config.batch_size)
        if self.index.ntotal == 0:
            return [
                RetrievalResult(
                    query=q,
                    contexts=[],
                    chunks=[],
                    retrieved_ids=[],
                    retrieved_child_ids=[],
                )
                for q in queries
            ]

        min_pool = k * 5 if self.config.context_unit == "parent" else k
        search_k = min(self.index.ntotal, max(self.config.search_k, min_pool))
        scores, indices = self.index.search(q_embs, search_k)

        results: list[RetrievalResult] = []
        for q, row_scores, row_indices in zip(queries, scores, indices):
            candidates: list[tuple[str, float]] = []
            for idx, sc in zip(row_indices, row_scores):
                if int(idx) < 0:
                    continue
                candidates.append((self.child_ids[int(idx)], float(sc)))
            results.append(
                resolve_candidates_to_contexts(
                    candidates=candidates,
                    children_by_id=self.children_by_id,
                    parents_by_id=self.parents_by_id,
                    top_k=k,
                    context_unit=self.config.context_unit,
                    reranker=self.reranker,
                    query=q,
                )
            )
        return results

    def unload(self) -> None:
        self.encoder.unload()
