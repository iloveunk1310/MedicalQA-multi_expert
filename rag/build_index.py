"""Build FAISS IndexFlatIP dense index and manifest from data/kb/kb_children.jsonl."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Optional

import faiss

from .config import RAGConfig
from .encoder import ViHealthBertEncoder


def build_dense_index(
    kb_dir: str = RAGConfig.kb_dir,
    index_dir: str = RAGConfig.index_dir,
    encoder_model: str = RAGConfig.encoder_model,
    pooling: str = RAGConfig.pooling,
    max_length: int = RAGConfig.max_length,
    batch_size: int = RAGConfig.batch_size,
    device: str = RAGConfig.device,
    limit: Optional[int] = None,
    encoder: Optional[ViHealthBertEncoder] = None,
) -> dict[str, Any]:
    """Encode kb_children.jsonl into a FAISS IndexFlatIP index and write manifest files."""
    t0 = time.perf_counter()
    children_path = Path(kb_dir) / "kb_children.jsonl"
    if not children_path.exists():
        raise FileNotFoundError(f"KB children file not found: {children_path}")

    child_ids: list[str] = []
    seg_texts: list[str] = []
    with children_path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            rec = json.loads(stripped)
            child_ids.append(rec["child_id"])
            seg_texts.append(rec["text_for_dense_seg"])
            if limit is not None and len(child_ids) >= limit:
                break

    if not child_ids:
        raise ValueError(f"No child records found in {children_path}")

    own_encoder = encoder is None
    if encoder is None:
        encoder = ViHealthBertEncoder(
            model_name=encoder_model,
            pooling=pooling,
            max_length=max_length,
            device=device,
        )

    try:
        embeddings = encoder.encode_segmented_texts(
            seg_texts,
            batch_size=batch_size,
            show_progress=True,
        )
    finally:
        if own_encoder:
            encoder.unload()

    dim = int(embeddings.shape[1])
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    out_dir = Path(index_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "children_dense.faiss"
    ids_path = out_dir / "child_ids.json"
    manifest_path = out_dir / "index_manifest.json"

    faiss.write_index(index, str(index_path))
    with ids_path.open("w", encoding="utf-8") as f:
        json.dump(child_ids, f, ensure_ascii=False)

    stat = children_path.stat()
    elapsed = round(time.perf_counter() - t0, 2)
    manifest: dict[str, Any] = {
        "encoder_model": encoder_model,
        "pooling": pooling,
        "max_length": max_length,
        "embedding_dim": dim,
        "index_type": "IndexFlatIP",
        "normalized_l2": True,
        "n_children": len(child_ids),
        "limit": limit,
        "kb_children_file": str(children_path),
        "kb_children_size_bytes": stat.st_size,
        "kb_children_mtime": stat.st_mtime,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed,
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return manifest


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FAISS dense index for Vietnamese medical KB.")
    parser.add_argument("--kb-dir", default=RAGConfig.kb_dir)
    parser.add_argument("--index-dir", default=RAGConfig.index_dir)
    parser.add_argument("--encoder-model", default=RAGConfig.encoder_model)
    parser.add_argument("--pooling", choices=["mean", "cls"], default=RAGConfig.pooling)
    parser.add_argument("--max-length", type=int, default=RAGConfig.max_length)
    parser.add_argument("--batch-size", type=int, default=RAGConfig.batch_size)
    parser.add_argument("--device", default=RAGConfig.device)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    manifest = build_dense_index(
        kb_dir=args.kb_dir,
        index_dir=args.index_dir,
        encoder_model=args.encoder_model,
        pooling=args.pooling,
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=args.device,
        limit=args.limit,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
