import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset

# Import từ thư viện local của bạn
from model_wrappers import load_model_configs, LLMClient

# 1. Định nghĩa tập nhãn VM14K
VM14K_SPECIALTIES = [
    "Allergy and Immunology", "Anesthesiology", "Cardiology", "Dermatology",
    "Endocrinology", "Gastroenterology", "Geriatrics", "Hematology",
    "Infectious Diseases", "Internal Medicine", "Nephrology", "Neurology",
    "Nuclear Medicine", "Obstetrics and Gynecology", "Oncology", "Ophthalmology",
    "Orthopedics", "Otolaryngology", "Palliative Medicine", "Pathology",
    "Pediatrics", "Physical Medicine and Rehabilitation", "Psychiatry",
    "Pulmonology", "Radiology", "Rheumatology", "Sports Medicine",
    "Surgery", "Urology", "General Medicine", "Eastern Medicine",
    "Public Health", "Preventive Healthcare", "Emergency Medicine"
]

# Lọc riêng các chuyên khoa có chứa chữ "Medicine"
MEDICINE_SPECIALTIES = [s for s in VM14K_SPECIALTIES if "Medicine" in s]

SPECIALTIES_LIST_STR = "\n".join([f"- {s}" for s in VM14K_SPECIALTIES])
MEDICINE_LIST_STR = "\n".join([f"- {s}" for s in MEDICINE_SPECIALTIES])

# 2. Định nghĩa các luồng Prompt động (Dynamic Prompts)
PROMPT_DEFAULT = """Bạn là một chuyên gia y tế. Hãy phân loại bài báo y khoa này vào ĐÚNG 1 trong 34 chuyên khoa dưới đây.
Danh sách chuyên khoa:
{specialties_list}

Thông tin bài báo:
- Tiêu đề: {title}
- Từ khóa: {keyword}

Chỉ trả về ĐÚNG MỘT TÊN CHUYÊN KHOA bằng tiếng Anh từ danh sách. Không giải thích.
Chuyên khoa:"""

PROMPT_DRUG = """Bạn là một dược sĩ lâm sàng. Dựa vào nội dung chi tiết về một loại thuốc dưới đây, hãy phân tích nhóm tác dụng của hoạt chất/cơ chế dược lý để phân loại nó vào ĐÚNG 1 trong 34 chuyên khoa y tế dưới đây.
Danh sách chuyên khoa:
{specialties_list}

Nội dung chi tiết của thuốc (Context):
{context}

Chỉ trả về ĐÚNG MỘT TÊN CHUYÊN KHOA bằng tiếng Anh từ danh sách, tương ứng với chuyên khoa sử dụng loại thuốc này nhiều nhất. Không giải thích.
Chuyên khoa:"""

PROMPT_MEDICINE = """Bạn là chuyên gia y khoa và dược liệu. Hãy phân loại bài viết này vào ĐÚNG 1 chuyên khoa trong không gian thu hẹp (chỉ bao gồm các phân ngành Medicine/Dược liệu) dưới đây.
Danh sách chuyên khoa thu hẹp:
{medicine_list}

Thông tin bài viết:
- Tiêu đề: {title}
- Từ khóa: {keyword}
- Nội dung trích lược: {context}

Chỉ trả về ĐÚNG MỘT TÊN CHUYÊN KHOA DƯỢC LIỆU bằng tiếng Anh từ danh sách thu hẹp trên. Không giải thích.
Chuyên khoa:"""

def build_prompt(row: dict) -> str:
    """Định tuyến Prompt dựa trên topic gốc của ViMedAQA."""
    topic_val = row.get("topic", "")
    # Xử lý trường hợp HuggingFace trả về int (0, 1, 2, 3) hoặc string
    topic_map = {0: 'body-part', 1: 'disease', 2: 'drug', 3: 'medicine'}
    topic_str = topic_map.get(topic_val, topic_val) if isinstance(topic_val, int) else str(topic_val).lower()

    if topic_str == 'drug':
        return PROMPT_DRUG.format(
            specialties_list=SPECIALTIES_LIST_STR,
            context=row.get("context", "")
        )
    elif topic_str == 'medicine':
        return PROMPT_MEDICINE.format(
            medicine_list=MEDICINE_LIST_STR,
            title=row.get("title", ""),
            keyword=row.get("keyword", ""),
            # Cắt ngắn context để tránh tràn token nếu kết hợp cả title/keyword
            context=str(row.get("context", ""))[:1500] 
        )
    else:
        return PROMPT_DEFAULT.format(
            specialties_list=SPECIALTIES_LIST_STR,
            title=row.get("title", ""),
            keyword=row.get("keyword", "")
        )

def extract_specialty(output_text: str, is_medicine_restricted: bool = False) -> str:
    """Lọc kết quả dựa trên không gian nhãn cho phép."""
    output_text = output_text.strip()
    valid_list = MEDICINE_SPECIALTIES if is_medicine_restricted else VM14K_SPECIALTIES
    
    for specialty in valid_list:
        if specialty.lower() in output_text.lower():
            return specialty
    return "Unclassified"

def load_sample_vimedaqa(num_samples=1000, seed=42):
    print("Đang tải dữ liệu ViMedAQA từ Hugging Face...")
    dataset = load_dataset("tmnam20/ViMedAQA", "all", split="test")
    df = dataset.to_pandas()
    # Giữ lại context để dùng cho nhánh Drug và Medicine
    df = df.dropna(subset=['title', 'keyword', 'context', 'topic'])
    if len(df) > num_samples:
        df = df.sample(n=num_samples, random_state=seed)
    return df

def plot_specialty_distribution(df_results, output_path="specialty_distribution.png"):
    plt.figure(figsize=(16, 12))
    sns.set_theme(style="whitegrid")
    
    counts = df_results['assigned_specialty'].value_counts().reset_index()
    counts.columns = ['Specialty', 'Count']
    
    ax = sns.barplot(x='Count', y='Specialty', data=counts, palette="viridis")
    ax.bar_label(ax.containers[0], padding=3)
    
    plt.title("Phân bổ ViMedAQA (Dynamic Prompts) vào 34 chuyên khoa VM14K", fontsize=16, pad=20)
    plt.xlabel("Số lượng mẫu (Count)", fontsize=12)
    plt.ylabel("Chuyên khoa y tế (Medical Specialty)", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Đã lưu biểu đồ phân bổ tại: {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_size", type=int, default=500)
    parser.add_argument("--models_config", type=str, default="configs/models.yaml")
    parser.add_argument("--model_name", type=str, default="qwen2.5-7b-instruct")
    parser.add_argument("--output_csv", type=str, default="results/vimedaqa_dynamic_classified.csv")
    args = parser.parse_args()

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)

    print(f"Khởi tạo LLM Client cho: {args.model_name}")
    model_cfgs = load_model_configs(args.models_config)
    cfg = next((c for c in model_cfgs if c.name == args.model_name), None)
    client = LLMClient(cfg)
    batch_size = getattr(cfg, "batch_size", 8)

    df_data = load_sample_vimedaqa(num_samples=args.sample_size)
    records = df_data.to_dict('records')
    results_list = []

    def _batched(seq, bs):
        for i in range(0, len(seq), bs):
            yield seq[i: i + bs]

    print("Bắt đầu gọi LLM để phân loại...")
    for batch in tqdm(list(_batched(records, batch_size))):
        prompts = [build_prompt(item) for item in batch]
        try:
            outputs = client.generate_batch(prompts)
        except Exception as e:
            print(f"[ERROR] Batch failed: {e}")
            outputs = [{"text": "Unclassified"} for _ in batch]
            
        for item, out in zip(batch, outputs):
            # Xác định xem có phải đang ở nhánh medicine để lọc nhãn gắt gao hơn không
            topic_val = item.get("topic", "")
            is_med = (topic_val == 3 or str(topic_val).lower() == 'medicine')
            
            raw_text = out['text']
            specialty = extract_specialty(raw_text, is_medicine_restricted=is_med)
            
            results_list.append({
                "question_idx": item.get("question_idx", ""),
                "original_topic": topic_val,
                "title": item["title"],
                "keyword": item["keyword"],
                "assigned_specialty": specialty,
                "raw_llm_output": raw_text
            })

    df_results = pd.DataFrame(results_list)
    df_results.to_csv(args.output_csv, index=False, encoding='utf-8-sig')
    
    plot_specialty_distribution(df_results, output_path="results/dynamic_specialty_distribution.png")

    if hasattr(client, "unload"):
        client.unload()

if __name__ == "__main__":
    main()