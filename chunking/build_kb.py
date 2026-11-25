"""CLI and pipeline entrypoint to build Contextual Parent-Child KB JSONL files."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
import math
from pathlib import Path
import time
from typing import Any, Optional

from .config import ChunkConfig
from .normalize import is_table_line, is_table_separator, normalize_section_text
from .prefix import (
    build_short_prefix,
    extract_entity_name,
    extract_source,
    extract_specialty_list,
)
from .splitter import split_parent_to_children, split_section_to_parents
from .tokenize_vi import ViHealthBertCounter

_WORKER_CONFIG: Optional[ChunkConfig] = None
_WORKER_COUNTER: Optional[ViHealthBertCounter] = None


def has_broken_table_header(text: str) -> bool:
    """Return True if any contiguous Markdown table line run lacks a valid 2-line header."""
    lines = text.splitlines()
    i = 0
    n = len(lines)
    while i < n:
        if is_table_line(lines[i]):
            run: list[str] = []
            while i < n and is_table_line(lines[i]):
                run.append(lines[i])
                i += 1
            if len(run) < 2 or not is_table_separator(run[1]):
                return True
        else:
            i += 1
    return False


def _build_doc_meta(doc: dict[str, Any]) -> dict[str, Any]:
    if doc.get("doc_type") == "drug_monograph":
        return {
            "inn_name": doc.get("inn_name"),
            "atc_code": doc.get("atc_code"),
            "drug_class": doc.get("drug_class"),
            "group_monograph": doc.get("group_monograph"),
            "source_page": doc.get("source_page"),
        }
    return {
        "source_file": doc.get("source_file"),
        "title_inferred": doc.get("title_inferred"),
    }


def process_document(
    doc: dict[str, Any],
    config: ChunkConfig,
    counter: Optional[ViHealthBertCounter] = None,
) -> dict[str, Any]:
    """Process a single guideline or drug monograph document into parent and child records."""
    doc_id = str(doc.get("doc_id") or "")
    doc_type = str(doc.get("doc_type") or "")
    source = extract_source(doc)
    decision_number = doc.get("decision_number")
    effective_date = doc.get("effective_date")
    specialty = extract_specialty_list(doc)
    entity_name = extract_entity_name(doc)
    meta = _build_doc_meta(doc)

    sections = doc.get("sections") or {}
    sections_processed = 0
    skipped_empty = 0
    parent_records: list[dict[str, Any]] = []
    child_records: list[dict[str, Any]] = []

    for section_key, raw_val in sections.items():
        norm_text = normalize_section_text(raw_val, section_key=section_key)
        if not norm_text:
            skipped_empty += 1
            continue

        sections_processed += 1
        parent_drafts = split_section_to_parents(doc, section_key, norm_text, config)
        n_parts = len(parent_drafts)

        for p_idx, p_draft in enumerate(parent_drafts, start=1):
            parent_id = f"{doc_id}__{section_key}__p{p_idx:02d}"
            text_for_llm = f"{p_draft['contextual_prefix']}\n{p_draft['text']}"
            parent_records.append(
                {
                    "parent_id": parent_id,
                    "doc_id": doc_id,
                    "doc_type": doc_type,
                    "source": source,
                    "decision_number": decision_number,
                    "effective_date": effective_date,
                    "specialty": specialty,
                    "entity_name": entity_name,
                    "section_key": section_key,
                    "section_name": p_draft["section_name"],
                    "contextual_prefix": p_draft["contextual_prefix"],
                    "text": p_draft["text"],
                    "text_for_llm": text_for_llm,
                    "word_count": p_draft["word_count"],
                    "part_index": p_idx,
                    "n_parts": n_parts,
                    "meta": meta,
                }
            )

            short_prefix = build_short_prefix(entity_name, p_draft["section_name"])
            child_drafts = split_parent_to_children(
                p_draft["text"],
                short_prefix,
                config,
                counter=counter,
            )
            for c_idx, c_draft in enumerate(child_drafts, start=1):
                child_id = f"{parent_id}__c{c_idx:02d}"
                child_records.append(
                    {
                        "child_id": child_id,
                        "parent_id": parent_id,
                        "doc_id": doc_id,
                        "doc_type": doc_type,
                        "specialty": specialty,
                        "entity_name": entity_name,
                        "section_key": section_key,
                        "short_prefix": short_prefix,
                        "text": c_draft["text"],
                        "text_for_dense": c_draft["text_for_dense"],
                        "text_for_dense_seg": c_draft["text_for_dense_seg"],
                        "subword_count": c_draft["subword_count"],
                        "word_count": c_draft["word_count"],
                    }
                )

    return {
        "doc_id": doc_id,
        "doc_type": doc_type,
        "sections_processed": sections_processed,
        "skipped_empty_sections": skipped_empty,
        "parents": parent_records,
        "children": child_records,
    }


def _init_worker(config_dict: dict[str, Any]) -> None:
    global _WORKER_CONFIG, _WORKER_COUNTER
    _WORKER_CONFIG = ChunkConfig(**config_dict)
    if _WORKER_CONFIG.child_length_mode == "subword":
        _WORKER_COUNTER = ViHealthBertCounter(_WORKER_CONFIG.tokenizer_name)
    else:
        _WORKER_COUNTER = None


def _process_batch(batch: list[tuple[str, dict[str, Any]]]) -> list[tuple[str, dict[str, Any]]]:
    assert _WORKER_CONFIG is not None
    return [
        (src_name, process_document(doc, _WORKER_CONFIG, _WORKER_COUNTER))
        for src_name, doc in batch
    ]


def _load_jsonl(path: str) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    docs: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                docs.append(json.loads(stripped))
    return docs


def _compute_stats(values: list[int]) -> Optional[dict[str, Any]]:
    if not values:
        return None
    ordered = sorted(values)
    n = len(ordered)

    def _pct(q: float) -> int:
        idx = min(n - 1, max(0, int(math.ceil(q * n) - 1)))
        return ordered[idx]

    return {
        "count": n,
        "min": ordered[0],
        "p50": _pct(0.50),
        "p90": _pct(0.90),
        "p99": _pct(0.99),
        "max": ordered[-1],
        "mean": round(sum(ordered) / n, 2),
    }


def build_kb(
    config: ChunkConfig,
    limit: Optional[int] = None,
    counter: Optional[ViHealthBertCounter] = None,
) -> dict[str, Any]:
    """Run full KB chunking pipeline and write kb_parents.jsonl, kb_children.jsonl, chunk_report.json."""
    t0 = time.perf_counter()

    guidelines = _load_jsonl(config.guidelines_path)
    drugs = _load_jsonl(config.drugs_path)

    if limit is not None:
        if limit < 0:
            raise ValueError("limit must be >= 0")
        if guidelines and drugs:
            g_lim = min(len(guidelines), (limit + 1) // 2)
            d_lim = min(len(drugs), limit - g_lim)
            if g_lim + d_lim < limit:
                extra = limit - (g_lim + d_lim)
                if g_lim < len(guidelines):
                    add_g = min(extra, len(guidelines) - g_lim)
                    g_lim += add_g
                    extra -= add_g
                if extra > 0 and d_lim < len(drugs):
                    d_lim += min(extra, len(drugs) - d_lim)
            guidelines = guidelines[:g_lim]
            drugs = drugs[:d_lim]
        elif guidelines:
            guidelines = guidelines[:limit]
        elif drugs:
            drugs = drugs[:limit]

    tagged_docs: list[tuple[str, dict[str, Any]]] = [
        ("clinical_guidelines", doc) for doc in guidelines
    ] + [("drug_monographs", doc) for doc in drugs]

    # Fail fast in subword mode if tokenizer cannot be loaded before spawning workers
    if config.child_length_mode == "subword" and counter is None:
        counter = ViHealthBertCounter(config.tokenizer_name)

    results: list[tuple[str, dict[str, Any]]] = []
    if config.workers > 1 and len(tagged_docs) > 1 and (
        config.child_length_mode == "words" or counter is not None
    ):
        batch_size = max(1, min(32, math.ceil(len(tagged_docs) / (config.workers * 4))))
        batches = [
            tagged_docs[i : i + batch_size]
            for i in range(0, len(tagged_docs), batch_size)
        ]
        # Only use ProcessPoolExecutor when we don't have an unpicklable injected mock counter
        use_pool = True
        if config.child_length_mode == "subword" and counter is not None:
            # Verify default HF tokenizer path is used (not a test mock object)
            use_pool = counter.tokenizer.__class__.__module__.startswith("transformers")
        if use_pool:
            with ProcessPoolExecutor(
                max_workers=config.workers,
                initializer=_init_worker,
                initargs=(asdict(config),),
            ) as executor:
                for batch_res in executor.map(_process_batch, batches):
                    results.extend(batch_res)
        else:
            for src_name, doc in tagged_docs:
                results.append((src_name, process_document(doc, config, counter)))
    else:
        for src_name, doc in tagged_docs:
            results.append((src_name, process_document(doc, config, counter)))

    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    parents_path = out_dir / "kb_parents.jsonl"
    children_path = out_dir / "kb_children.jsonl"
    report_path = out_dir / "chunk_report.json"

    docs_by_source: Counter[str] = Counter()
    docs_by_type: Counter[str] = Counter()
    sections_processed = 0
    skipped_empty_sections = 0

    all_parents: list[dict[str, Any]] = []
    all_children: list[dict[str, Any]] = []
    parent_ids_set: set[str] = set()
    children_per_parent: Counter[str] = Counter()

    with (
        parents_path.open("w", encoding="utf-8") as pf,
        children_path.open("w", encoding="utf-8") as cf,
    ):
        for src_name, res in results:
            docs_by_source[src_name] += 1
            docs_by_type[res["doc_type"]] += 1
            sections_processed += res["sections_processed"]
            skipped_empty_sections += res["skipped_empty_sections"]

            for p in res["parents"]:
                pf.write(json.dumps(p, ensure_ascii=False) + "\n")
                all_parents.append(p)
                parent_ids_set.add(p["parent_id"])
            for c in res["children"]:
                cf.write(json.dumps(c, ensure_ascii=False) + "\n")
                all_children.append(c)
                children_per_parent[c["parent_id"]] += 1

    parent_word_counts = [p["word_count"] for p in all_parents]
    child_word_counts = [c["word_count"] for c in all_children]
    child_subword_counts = [
        c["subword_count"] for c in all_children if c["subword_count"] is not None
    ]

    parents_over_512 = sum(1 for wc in parent_word_counts if wc > config.parent_max_words)
    if config.child_length_mode == "subword":
        children_over_limit = sum(
            1 for sw in child_subword_counts if sw > config.child_max_subwords
        )
    else:
        children_over_limit = sum(
            1 for wc in child_word_counts if wc > config.child_max_words
        )

    children_with_broken_table_header = sum(
        1 for c in all_children if has_broken_table_header(c["text"])
    )
    orphan_children = sum(1 for c in all_children if c["parent_id"] not in parent_ids_set)
    childless_parents = sum(1 for pid in parent_ids_set if children_per_parent[pid] == 0)
    valid_parent_id_ratio = (
        round((len(all_children) - orphan_children) / len(all_children), 6)
        if all_children
        else 1.0
    )

    parents_with_single_child = sum(
        1 for pid in parent_ids_set if children_per_parent[pid] == 1
    )
    parents_with_multiple_children = sum(
        1 for pid in parent_ids_set if children_per_parent[pid] > 1
    )

    elapsed_seconds = round(time.perf_counter() - t0, 2)
    report: dict[str, Any] = {
        "child_length_mode": config.child_length_mode,
        "elapsed_seconds": elapsed_seconds,
        "config": asdict(config),
        "docs_total": len(results),
        "docs_by_source": dict(docs_by_source),
        "docs_by_type": dict(docs_by_type),
        "sections_processed": sections_processed,
        "skipped_empty_sections": skipped_empty_sections,
        "total_parents": len(all_parents),
        "total_children": len(all_children),
        "parents_with_single_child": parents_with_single_child,
        "parents_with_multiple_children": parents_with_multiple_children,
        "parent_word_count_stats": _compute_stats(parent_word_counts),
        "child_word_count_stats": _compute_stats(child_word_counts),
        "child_subword_count_stats": _compute_stats(child_subword_counts),
        "parents_over_512": parents_over_512,
        "children_over_limit": children_over_limit,
        "children_with_broken_table_header": children_with_broken_table_header,
        "orphan_children": orphan_children,
        "childless_parents": childless_parents,
        "valid_parent_id_ratio": valid_parent_id_ratio,
    }

    with report_path.open("w", encoding="utf-8") as rf:
        json.dump(report, rf, ensure_ascii=False, indent=2)

    return report


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Contextual Parent-Child KB JSONL files for Vietnamese medical RAG."
    )
    parser.add_argument("--guidelines", default=ChunkConfig.guidelines_path)
    parser.add_argument("--drugs", default=ChunkConfig.drugs_path)
    parser.add_argument("--out-dir", default=ChunkConfig.out_dir)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--child-length",
        choices=["subword", "words"],
        default="subword",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    config = ChunkConfig(
        guidelines_path=args.guidelines,
        drugs_path=args.drugs,
        out_dir=args.out_dir,
        workers=args.workers,
        child_length_mode=args.child_length,
    )
    report = build_kb(config, limit=args.limit)
    print(
        json.dumps(
            {
                "docs_total": report["docs_total"],
                "total_parents": report["total_parents"],
                "total_children": report["total_children"],
                "parents_over_512": report["parents_over_512"],
                "children_over_limit": report["children_over_limit"],
                "children_with_broken_table_header": report["children_with_broken_table_header"],
                "orphan_children": report["orphan_children"],
                "childless_parents": report["childless_parents"],
                "elapsed_seconds": report["elapsed_seconds"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
