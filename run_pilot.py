"""
Script chính chạy pilot.

Ví dụ dùng:
    python run_pilot.py --dataset vm14k --strategy zeroshot --sample_per_level 100 \
        --vm14k_path data/vm14k_test.jsonl

    python run_pilot.py --dataset vm14k --strategy rag --sample_per_level 100 \
        --vm14k_path data/vm14k_test.jsonl

    python run_pilot.py --dataset vimedaqa --strategy cot --sample 200 \
        --vimedaqa_path tmnam20/ViMedAQA

Mỗi lần chạy sẽ lặp qua TẤT CẢ model khai báo trong configs/models.yaml,
ghi kết quả ra results/<dataset>_<strategy>_<model_name>.jsonl
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Callable
from tqdm import tqdm

from data_loaders import (
    load_vm14k, load_vimedaqa, stratified_sample_by_difficulty, random_sample,
)
from prompts import extract_option_pairs, format_mcq_prompt, format_abstractive_prompt
from model_wrappers import load_model_configs, LLMClient
from parse_utils import parse_mcq_answer

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def _is_error_record(rec: dict[str, Any]) -> bool:
    raw = rec.get("raw_output")
    pred = rec.get("pred_answer")
    return (isinstance(raw, str) and raw.startswith("[ERROR]")) or (
        isinstance(pred, str) and pred.startswith("[ERROR]")
    )


def _load_done_ids(out_path: Path) -> set:
    """Đọc kết quả hợp lệ đã có để resume, tự động bỏ qua dòng [ERROR] cũ."""
    if not out_path.exists():
        return set()
    done = set()
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if _is_error_record(rec):
                continue
            if "id" in rec and rec["id"] is not None:
                done.add(rec["id"])
    return done


def _batched(seq, batch_size):
    for i in range(0, len(seq), batch_size):
        yield seq[i: i + batch_size]


def _get_valid_letters(options) -> list[str]:
    return [k for k, _ in extract_option_pairs(options)]


def _build_rag_query(item: dict[str, Any], query_mode: str = "question") -> str:
    question = str(item.get("question") or "").strip()
    if query_mode == "question_options" and item.get("options"):
        opts = " ".join(f"{k}. {v}" for k, v in extract_option_pairs(item["options"]))
        return f"{question}\n{opts}".strip()
    return question


def _log_error_record(
    error_path: Path,
    item_id: Any,
    model_name: str,
    strategy: str,
    error: Exception,
) -> None:
    error_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "id": item_id,
        "model": model_name,
        "strategy": strategy,
        "error": str(error),
    }
    with open(error_path, "a", encoding="utf-8") as ef:
        ef.write(json.dumps(entry, ensure_ascii=False) + "\n")
        ef.flush()


def _generate_batch_resilient(
    client: LLMClient,
    batch_entries: list[Any],
    prompts: list[str],
    id_getter: Callable[[Any], Any],
    error_path: Path,
    model_name: str,
    strategy: str,
) -> list[tuple[Any, dict[str, Any]]]:
    """Gọi generate_batch; nếu lỗi thì fallback từng câu, chỉ ghi log lỗi sang file riêng."""
    try:
        results = client.generate_batch(prompts)
        return list(zip(batch_entries, results))
    except Exception as batch_exc:
        if len(batch_entries) <= 1:
            if batch_entries:
                _log_error_record(
                    error_path, id_getter(batch_entries[0]), model_name, strategy, batch_exc
                )
            return []

        rescued: list[tuple[Any, dict[str, Any]]] = []
        for entry, prompt in zip(batch_entries, prompts):
            try:
                res = client.generate(prompt)
                rescued.append((entry, res))
            except Exception as item_exc:
                _log_error_record(
                    error_path, id_getter(entry), model_name, strategy, item_exc
                )
        return rescued


def run_vm14k(
    items,
    model_name,
    client,
    strategy,
    out_path,
    retriever=None,
    rag_top_k: int = 3,
    rag_query_mode: str = "question",
):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    error_path = out_path.with_suffix(".errors.jsonl")

    done_ids = _load_done_ids(out_path)
    remaining = [it for it in items if it.get("id") not in done_ids]
    if done_ids:
        print(f"[RESUME] Bỏ qua {len(done_ids)} câu đã chạy trước đó, "
              f"còn lại {len(remaining)} câu")

    if strategy == "rag" and retriever is None:
        raise ValueError("Retriever is required when strategy='rag'")

    bs = getattr(client.cfg, "batch_size", 1)
    with open(out_path, "a", encoding="utf-8") as f:
        for batch in tqdm(list(_batched(remaining, bs)),
                           desc=f"{model_name}/{strategy}/vm14k"):
            if strategy == "rag":
                queries = [_build_rag_query(it, query_mode=rag_query_mode) for it in batch]
                ret_results = retriever.retrieve_batch(queries, top_k=rag_top_k)
                prompts = [
                    format_mcq_prompt(
                        it["question"],
                        it["options"],
                        strategy,
                        context=rr.format_context(),
                    )
                    for it, rr in zip(batch, ret_results)
                ]
                batch_entries = list(zip(batch, ret_results))
            else:
                prompts = [format_mcq_prompt(it["question"], it["options"], strategy)
                           for it in batch]
                batch_entries = [(it, None) for it in batch]

            succeeded = _generate_batch_resilient(
                client=client,
                batch_entries=batch_entries,
                prompts=prompts,
                id_getter=lambda entry: entry[0].get("id"),
                error_path=error_path,
                model_name=model_name,
                strategy=strategy,
            )

            for (item, rr), result in succeeded:
                valid_letters = _get_valid_letters(item["options"])
                pred, parse_method = parse_mcq_answer(
                    result["text"], valid_letters=valid_letters
                )
                record = {
                    "id": item.get("id"),
                    "difficulty": item.get("difficulty_level", "UNKNOWN"),
                    "specialty": item.get("specialty"),
                    "gold_answer": item.get("answer"),
                    "pred_answer": pred,
                    "parse_method": parse_method,
                    "correct": (pred == item.get("answer")) if pred is not None else None,
                    "raw_output": result["text"],
                    "latency_sec": result["latency_sec"],
                    "prompt_tokens": result["prompt_tokens"],
                    "completion_tokens": result["completion_tokens"],
                    "model": model_name,
                    "strategy": strategy,
                }
                if rr is not None:
                    record["retrieved_ids"] = rr.retrieved_ids
                    record["retrieved_child_ids"] = rr.retrieved_child_ids
                    record["top_retrieval_score"] = rr.top_retrieval_score
                    record["top_rerank_score"] = rr.top_rerank_score
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()


def run_vimedaqa(
    items,
    model_name,
    client,
    strategy,
    out_path,
    retriever=None,
    rag_top_k: int = 3,
    rag_query_mode: str = "question",
):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    error_path = out_path.with_suffix(".errors.jsonl")

    done_ids = _load_done_ids(out_path)
    # Lấy id ưu tiên key "id", nếu không có thì lấy "question_idx" (cho data từ HuggingFace)
    remaining = [it for it in items if it.get("id", it.get("question_idx")) not in done_ids]
    if done_ids:
        print(f"[RESUME] Bỏ qua {len(done_ids)} câu đã chạy trước đó, "
              f"còn lại {len(remaining)} câu")

    if strategy == "rag" and retriever is None:
        raise ValueError("Retriever is required when strategy='rag'")

    bs = getattr(client.cfg, "batch_size", 1)
    with open(out_path, "a", encoding="utf-8") as f:
        for batch in tqdm(list(_batched(remaining, bs)),
                           desc=f"{model_name}/{strategy}/vimedaqa"):
            if strategy == "rag":
                queries = [_build_rag_query(it, query_mode=rag_query_mode) for it in batch]
                ret_results = retriever.retrieve_batch(queries, top_k=rag_top_k)
                prompts = [
                    format_abstractive_prompt(
                        it["question"],
                        rr.format_context() if rr.contexts else it.get("context", ""),
                        strategy,
                    )
                    for it, rr in zip(batch, ret_results)
                ]
                batch_entries = list(zip(batch, ret_results))
            else:
                prompts = [format_abstractive_prompt(
                    it["question"], it.get("context", ""), strategy) for it in batch]
                batch_entries = [(it, None) for it in batch]

            succeeded = _generate_batch_resilient(
                client=client,
                batch_entries=batch_entries,
                prompts=prompts,
                id_getter=lambda entry: entry[0].get("id", entry[0].get("question_idx")),
                error_path=error_path,
                model_name=model_name,
                strategy=strategy,
            )

            for (item, rr), result in succeeded:
                record = {
                    "id": item.get("id", item.get("question_idx")),  # Tương thích HF dataset
                    "question": item["question"],
                    "gold_answer": item.get("answer"),
                    "pred_answer": result["text"],
                    "latency_sec": result["latency_sec"],
                    "prompt_tokens": result["prompt_tokens"],
                    "completion_tokens": result["completion_tokens"],
                    "model": model_name,
                    "strategy": strategy,
                }
                if rr is not None:
                    record["retrieved_ids"] = rr.retrieved_ids
                    record["retrieved_child_ids"] = rr.retrieved_child_ids
                    record["top_retrieval_score"] = rr.top_retrieval_score
                    record["top_rerank_score"] = rr.top_rerank_score
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()


def _build_retriever(args):
    if args.strategy != "rag":
        return None
    from rag.config import RAGConfig
    from rag.retriever import DenseRetriever

    rag_cfg = RAGConfig(
        kb_dir=args.kb_dir,
        index_dir=args.index_dir,
        top_k=args.rag_top_k,
        context_unit=args.rag_context_unit,
        query_mode=args.rag_query_mode,
        device=args.rag_device,
    )
    return DenseRetriever(rag_cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["vm14k", "vimedaqa"], required=True)
    ap.add_argument("--strategy", choices=["zeroshot", "cot", "rag"], required=True)
    ap.add_argument("--vm14k_path", default="data/vm14k_test.jsonl")
    # Đổi default thành tên repo trên HuggingFace
    ap.add_argument("--vimedaqa_path", default="tmnam20/ViMedAQA")
    ap.add_argument("--sample_per_level", type=int, default=100,
                     help="Số câu lấy mỗi mức độ khó (chỉ áp dụng cho vm14k)")
    ap.add_argument("--sample", type=int, default=200,
                     help="Số câu lấy ngẫu nhiên (chỉ áp dụng cho vimedaqa)")
    ap.add_argument("--models_config", default="configs/models.yaml")
    ap.add_argument("--model_name", default=None,
                     help="Chỉ chạy 1 model theo 'name' trong models.yaml. "
                          "KHUYẾN NGHỊ dùng trên Colab/Kaggle: chạy từng model "
                          "một lần runtime, tránh load nhiều model 7B cùng lúc gây OOM.")
    ap.add_argument("--kb_dir", default="data/kb")
    ap.add_argument("--index_dir", default="data/kb/index")
    ap.add_argument("--rag_top_k", type=int, default=3)
    ap.add_argument("--rag_context_unit", choices=["parent", "child"], default="parent")
    ap.add_argument(
        "--rag_query_mode",
        choices=["question", "question_options"],
        default="question",
    )
    ap.add_argument("--rag_device", default="auto")
    args = ap.parse_args()

    model_cfgs = load_model_configs(args.models_config)
    if args.model_name:
        model_cfgs = [c for c in model_cfgs if c.name == args.model_name]
        if not model_cfgs:
            raise ValueError(f"Không tìm thấy model '{args.model_name}' trong "
                              f"{args.models_config}")

    if args.dataset == "vm14k":
        items = load_vm14k(args.vm14k_path)
        items = stratified_sample_by_difficulty(
            items,
            n_per_level=args.sample_per_level,
            difficulty_field="difficulty_level",
        )
    else:
        items = load_vimedaqa(args.vimedaqa_path)
        items = random_sample(items, n=args.sample)
        print(f"Đã lấy mẫu {len(items)} câu ViMedAQA (random)")

    retriever = _build_retriever(args)
    try:
        for cfg in model_cfgs:
            print(f"\n=== Model: {cfg.name} ===")
            try:
                client = LLMClient(cfg)
            except Exception as e:
                print(f"[SKIP] Không khởi tạo được model {cfg.name}: {e}")
                continue

            out_path = RESULTS_DIR / f"{args.dataset}_{args.strategy}_{cfg.name}.jsonl"
            if args.dataset == "vm14k":
                run_vm14k(
                    items,
                    cfg.name,
                    client,
                    args.strategy,
                    out_path,
                    retriever=retriever,
                    rag_top_k=args.rag_top_k,
                    rag_query_mode=args.rag_query_mode,
                )
            else:
                run_vimedaqa(
                    items,
                    cfg.name,
                    client,
                    args.strategy,
                    out_path,
                    retriever=retriever,
                    rag_top_k=args.rag_top_k,
                    rag_query_mode=args.rag_query_mode,
                )
            print(f"Đã lưu: {out_path}")

            if cfg.backend == "local_hf":
                client.unload()
    finally:
        if retriever is not None:
            retriever.unload()


if __name__ == "__main__":
    main()
