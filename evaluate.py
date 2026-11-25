"""
Gom toàn bộ results/*.jsonl, tính metric, in bảng so sánh, lưu CSV.

Chạy: python evaluate.py
"""
import glob
import json
from pathlib import Path
from typing import Any

import pandas as pd

RESULTS_DIR = Path("results")


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def is_error_record(rec: dict[str, Any]) -> bool:
    raw = rec.get("raw_output")
    pred = rec.get("pred_answer")
    return (isinstance(raw, str) and raw.startswith("[ERROR]")) or (
        isinstance(pred, str) and pred.startswith("[ERROR]")
    )


def clean_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Lọc bỏ bản ghi [ERROR] và khử trùng lặp theo id (giữ bản ghi cuối cùng)."""
    valid: list[dict[str, Any]] = []
    n_errors = 0
    for rec in records:
        if is_error_record(rec):
            n_errors += 1
            continue
        valid.append(rec)

    last_idx_by_id: dict[Any, int] = {}
    for idx, rec in enumerate(valid):
        rec_id = rec.get("id")
        if rec_id is not None:
            last_idx_by_id[rec_id] = idx

    deduped: list[dict[str, Any]] = []
    for idx, rec in enumerate(valid):
        rec_id = rec.get("id")
        if rec_id is None or last_idx_by_id.get(rec_id) == idx:
            deduped.append(rec)
    return deduped, n_errors


def evaluate_mcq(results_dir: Path = RESULTS_DIR):
    results_dir = Path(results_dir)
    files = sorted(
        fp
        for fp in glob.glob(str(results_dir / "vm14k_*.jsonl"))
        if not fp.endswith(".errors.jsonl")
    )
    if not files:
        print("Không tìm thấy kết quả VM14K trong results/")
        return None

    rows = []
    for fp in files:
        raw_records = load_jsonl(fp)
        records, n_errors = clean_records(raw_records)
        if not records:
            continue
        model = records[0]["model"]
        strategy = records[0]["strategy"]

        df = pd.DataFrame(records)
        is_parse_fail = df["pred_answer"].isna() | (
            df["pred_answer"].astype(str).str.strip() == ""
        )
        n_parse_fail = int(is_parse_fail.sum())
        parse_fail_rate = round(float(is_parse_fail.mean()), 4)

        is_correct = (~is_parse_fail) & (df["pred_answer"] == df["gold_answer"])
        overall_acc = round(float(is_correct.mean()), 4)
        parsed_mask = ~is_parse_fail
        parsed_acc = (
            round(float(is_correct[parsed_mask].mean()), 4)
            if bool(parsed_mask.any())
            else None
        )

        avg_latency = df["latency_sec"].dropna().mean() if "latency_sec" in df else None
        if "prompt_tokens" in df and "completion_tokens" in df:
            avg_tokens = (
                df["prompt_tokens"].dropna().mean()
                + df["completion_tokens"].dropna().mean()
            )
        else:
            avg_tokens = None

        row = {
            "model": model,
            "strategy": strategy,
            "overall_accuracy": overall_acc,
            "parsed_accuracy": parsed_acc,
            "parse_fail_rate": parse_fail_rate,
            "n_parse_fail": n_parse_fail,
            "n_errors": n_errors,
            "avg_latency_sec": round(float(avg_latency), 2) if pd.notna(avg_latency) else None,
            "avg_tokens_per_q": round(float(avg_tokens), 1) if pd.notna(avg_tokens) else None,
            "n_questions": len(df),
        }
        # Accuracy breakdown theo difficulty — cốt lõi để sau này validate router
        for level, sub in df.groupby("difficulty"):
            row[f"acc_{level}"] = round(float(is_correct.loc[sub.index].mean()), 4)
        rows.append(row)

    if not rows:
        print("Không có bản ghi VM14K hợp lệ trong results/")
        return None

    out = pd.DataFrame(rows).sort_values(
        ["strategy", "overall_accuracy"], ascending=[True, False]
    )
    out.to_csv(results_dir / "summary_mcq.csv", index=False)
    print("\n=== VM14K — Accuracy tổng quan + theo độ khó ===")
    print(out.to_string(index=False))
    return out


def clean_cot_prediction(text: str) -> str:
    """Tách lấy phần câu trả lời cuối cùng sau tiền tố 'Trả lời:' của chiến lược CoT."""
    if "Trả lời:" in text:
        # Lấy phần văn bản nằm sau chữ 'Trả lời:' cuối cùng
        return text.split("Trả lời:")[-1].strip()
    return text.strip()


def evaluate_abstractive(results_dir: Path = RESULTS_DIR):
    results_dir = Path(results_dir)
    files = sorted(
        fp
        for fp in glob.glob(str(results_dir / "vimedaqa_*.jsonl"))
        if not fp.endswith(".errors.jsonl")
    )
    if not files:
        print("Không tìm thấy kết quả ViMedAQA trong results/")
        return None

    try:
        import sacrebleu
        from rouge_score import rouge_scorer
        from bert_score import score as bertscore
    except ImportError as exc:
        print(f"[SKIP] Thiếu thư viện đánh giá abstractive ({exc}); bỏ qua summary_abstractive.csv.")
        return None

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    rows = []
    for fp in files:
        raw_records = load_jsonl(fp)
        records, n_errors = clean_records(raw_records)
        records = [r for r in records if r.get("gold_answer") and r.get("pred_answer")]
        if not records:
            continue
        model = records[0]["model"]
        strategy = records[0]["strategy"]

        preds = [clean_cot_prediction(r["pred_answer"]) for r in records]
        golds = [r["gold_answer"] for r in records]

        bleu = sacrebleu.corpus_bleu(preds, [golds]).score
        rouge_l = (
            sum(scorer.score(g, p)["rougeL"].fmeasure for g, p in zip(golds, preds))
            / len(records)
        )

        # BERTScore multilingual — dùng model mặc định đa ngôn ngữ (bert-base-multilingual-cased)
        # Đổi lang="vi" nếu bert-score hỗ trợ checkpoint tiếng Việt riêng trong môi trường của bạn.
        P, R, F1 = bertscore(preds, golds, lang="vi", verbose=False)
        bertscore_f1 = F1.mean().item()

        df = pd.DataFrame(records)
        avg_latency = df["latency_sec"].dropna().mean()
        avg_tokens = (
            df["prompt_tokens"].dropna().mean()
            + df["completion_tokens"].dropna().mean()
        )

        rows.append(
            {
                "model": model,
                "strategy": strategy,
                "BLEU": round(bleu, 2),
                "ROUGE_L": round(rouge_l, 4),
                "BERTScore_F1": round(bertscore_f1, 4),
                "n_errors": n_errors,
                "avg_latency_sec": round(float(avg_latency), 2) if pd.notna(avg_latency) else None,
                "avg_tokens_per_q": round(float(avg_tokens), 1) if pd.notna(avg_tokens) else None,
                "n_questions": len(records),
            }
        )

    if not rows:
        return None

    out = pd.DataFrame(rows).sort_values(
        ["strategy", "BERTScore_F1"], ascending=[True, False]
    )
    out.to_csv(results_dir / "summary_abstractive.csv", index=False)
    print("\n=== ViMedAQA — BLEU / ROUGE-L / BERTScore ===")
    print(out.to_string(index=False))
    print(
        "\n[Lưu ý] BLEU/ROUGE/BERTScore chỉ đo độ trùng khớp bề mặt/ngữ nghĩa, "
        "KHÔNG đo factual consistency. Để đánh giá đúng mức độ 'bịa' hay 'bỏ sót "
        "bằng chứng', cần bổ sung NLI-based hoặc LLM-as-judge faithfulness ở vòng sau."
    )
    return out


if __name__ == "__main__":
    evaluate_mcq()
    evaluate_abstractive()
