"""
Data loaders cho VM14K và ViMedAQA.

LƯU Ý: Hai dataset này thường phát hành qua HuggingFace `datasets` hoặc file
JSON/CSV tải từ GitHub của tác giả. Điền đúng `path_or_name` bên dưới theo
nguồn thật bạn tải được (kiểm tra README của từng bộ để biết field chính xác —
tên field dưới đây dựa trên mô tả public, có thể cần đổi lại cho khớp).

Mỗi loader trả về list[dict] với format thống nhất để pipeline dùng chung:
  - VM14K item: {id, question, options: {A,B,C,D,...}, answer, difficulty, specialty}
  - ViMedAQA item: {id, question, context, answer}
"""
import json
import random
from pathlib import Path
from collections import defaultdict


def load_vm14k(
    path_or_name: str,
    hf_config_name: str = "test",
    hf_split: str = "vietnamese",
) -> list[dict]:
    """
    Tải VM14K. Xử lý đọc thư mục chứa nhiều file JSONL, 
    hoặc đọc 1 file cụ thể, hoặc tải từ HuggingFace Hub.
    """
    # SỬA LỖI: Sử dụng path_or_name thay vì chuỗi hardcode
    p = Path(path_or_name)
    
    items = []
    
    # Kịch bản 1: Nếu đường dẫn là một thư mục (chứa 3 file test của bạn)
    if p.is_dir():
        for file_path in p.glob("*"): # Quét tất cả các file trong thư mục
            if file_path.is_file():
                with open(file_path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        items.append(json.loads(line))
        return items
        
    # Kịch bản 2: Nếu đường dẫn là một file cụ thể
    if p.exists() and p.is_file():
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                items.append(json.loads(line))
        return items
        
    # Kịch bản 3: Fallback tải trực tiếp từ HuggingFace Hub nếu không có ở local
    from datasets import load_dataset
    # Chú ý: lúc này path_or_name đóng vai trò là repo_id trên Hub
    ds = load_dataset(path_or_name, hf_config_name, split=hf_split)
    return [dict(x) for x in ds]


def load_vimedaqa(path_or_name: str) -> list[dict]:
    """Tương tự load_vm14k nhưng cho ViMedAQA (abstractive, open-book)."""
    p = Path(path_or_name)
    
    # 1. Thử đọc từ file local trước
    if p.exists():
        items = []
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                items.append(json.loads(line))
        return items
        
    # 2. Nếu file không tồn tại, tự động kéo từ HuggingFace
    from datasets import load_dataset
    # Thêm tham số "all" để tương thích với config_name của tmnam20/ViMedAQA
    ds = load_dataset("tmnam20/ViMedAQA", "all", split="test") 
    return [dict(x) for x in ds]


def stratified_sample_by_difficulty(
    items: list[dict], n_per_level: int = 100, seed: int = 42,
    difficulty_field: str = "difficulty",
) -> list[dict]:
    """
    Lấy mẫu đều theo từng mức độ khó (VD: LOW/MEDIUM/HIGH trong VM14K).
    Quan trọng: nếu không stratify, mẫu random sẽ lệch về nhóm chiếm đa số
    (MEDIUM/HIGH ~88% theo phân bố CAMEC Table 4) và không đủ mẫu LOW để
    so sánh accuracy theo độ khó một cách tin cậy.
    """
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for it in items:
        level = it.get(difficulty_field, "UNKNOWN")
        buckets[level].append(it)

    sampled = []
    for level, bucket in buckets.items():
        rng.shuffle(bucket)
        take = bucket[: min(n_per_level, len(bucket))]
        sampled.extend(take)
        if len(bucket) < n_per_level:
            print(f"[WARN] mức '{level}' chỉ có {len(bucket)} câu, "
                  f"lấy hết thay vì {n_per_level}")
    rng.shuffle(sampled)
    return sampled


def random_sample(items: list[dict], n: int = 200, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    items = items.copy()
    rng.shuffle(items)
    return items[: min(n, len(items))]
