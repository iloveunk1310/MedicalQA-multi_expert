
# KẾ HOẠCH TRIỂN KHAI CONTEXTUAL PARENT-CHILD CHUNKING VÀ PIPELINE RAG ĐÁNH GIÁ NHANH CHẤT LƯỢNG TÀI LIỆU

---

## PHẦN 1: OUTPUT SAU BƯỚC CHUNKING & KỸ THUẬT PARENT-CHILD CHUNKING NHẤT QUÁN

### 1.1. Định dạng và Cấu trúc Output chuẩn (`.jsonl`)

Để tương thích hoàn toàn với kiến trúc truy xuất lai (Hybrid Retrieval: BM25 + Dense Vector) và cơ chế ánh xạ ngữ cảnh của đồ án khóa trước, đầu ra của bước Chunking nên được xuất thành **2 tệp `.jsonl` liên kết khóa ngoại với nhau**:

1. **`kb_parents.jsonl` (Kho ngữ cảnh cung cấp cho LLM sinh câu trả lời):** Lưu các đoạn văn bản cha có độ dài vừa đủ ($\le 512$ từ/token) để bảo toàn điều kiện lâm sàng, chống chỉ định và liều lượng đi kèm.
2. **`kb_children.jsonl` (Kho đơn vị tìm kiếm dùng để đánh chỉ mục FAISS & BM25):** Lưu các đoạn văn bản con nhỏ gọn ($\le 256$ từ/token) giúp tăng độ chính xác khi so khớp từ khóa và ngữ nghĩa.

#### Ví dụ Output thực tế của `kb_parents.jsonl`:
```json
{
  "parent_id": "BYT_HUYETHOC_1832_ART07_SEC03_P01",
  "doc_id": "BYT_HUYETHOC_1832_ART07",
  "source": "Phác đồ Bộ Y tế (QĐ 1832/QĐ-BYT)",
  "effective_date": "2022-07-04",
  "status": "active",
  "specialty_vm14k": ["Hematology"],
  "entity_name": "HỘI CHỨNG EVANS",
  "section_key": "treatment_procedure",
  "section_name": "Điều trị",
  "contextual_prefix": "[Nguồn: Phác đồ BYT (1832/QĐ-BYT) | Chuyên khoa: Hematology | Bệnh: HỘI CHỨNG EVANS | Mục: Điều trị]",
  "text_for_llm": "[Nguồn: Phác đồ BYT (1832/QĐ-BYT) | Chuyên khoa: Hematology | Bệnh: HỘI CHỨNG EVANS | Mục: Điều trị]\n3.1. Methylprednisolon: Liều dùng 1 - 2 mg/kg/ngày. Khi có đáp ứng (huyết sắc tố > 80g/L và tiểu cầu > 50G/L) thì giảm liều dần (30% liều/tuần). Trường hợp cơn tan máu rầm rộ hoặc xuất huyết nhiều nơi, nguy cơ đe dọa tính mạng có thể dùng methylprednisolone liều cao (bolus): Liều 1g/ngày trong 3 ngày; sau đó dùng liều 3-4mg/kg/ngày trong 3-5 ngày...",
  "token_count": 342
}
```

#### Ví dụ Output thực tế của `kb_children.jsonl`:
```json
{
  "child_id": "BYT_HUYETHOC_1832_ART07_SEC03_P01_C01",
  "parent_id": "BYT_HUYETHOC_1832_ART07_SEC03_P01",
  "doc_id": "BYT_HUYETHOC_1832_ART07",
  "specialty_vm14k": ["Hematology"],
  "entity_name": "HỘI CHỨNG EVANS",
  "section_key": "treatment_procedure",
  "text_for_dense": "[Nguồn: Phác đồ BYT (1832/QĐ-BYT) | Chuyên khoa: Hematology | Bệnh: HỘI CHỨNG EVANS | Mục: Điều trị] 3.1. Methylprednisolon: Liều dùng 1 - 2 mg/kg/ngày. Khi có đáp ứng (huyết sắc tố > 80g/L và tiểu cầu > 50G/L) thì giảm liều dần (30% liều/tuần)...",
  "tokens_for_bm25": ["nguồn", "phác_đồ", "byt", "1832/qđ-byt", "chuyên_khoa", "hematology", "bệnh", "hội_chứng", "evans", "mục", "điều_trị", "methylprednisolon", "liều_dùng", "1", "2", "mg/kg/ngày", "đáp_ứng", "huyết_sắc_tố", "80g/l", "tiểu_cầu", "50g/l", "giảm", "liều", "30%", "liều/tuần"],
  "token_count": 185
}
```

> **Lưu ý quan trọng:** Nhờ có `contextual_prefix` nằm ngay đầu `text_for_dense` và `tokens_for_bm25`, mỗi đoạn con dù bị cắt ra ở giữa bài vẫn mang đầy đủ thông tin định danh (Tên thuốc/Tên bệnh + Tên đề mục), loại bỏ hoàn toàn hiện tượng "mơ hồ cục bộ" (local ambiguity).

---

### 1.2. Bốn quy tắc Kỹ thuật Parent-Child Chunking để đảm bảo tính nhất quán

Do dữ liệu của bạn đến từ 3 nguồn có cấu trúc khác nhau (*Dược thư 19 mục*, *Phác đồ Nội khoa 6 nhóm chuẩn*, *Quy trình Ngoại khoa*) và cần hợp nhất với kho *114.384 tài liệu Vinmec/Thuốc Biệt Dược* của khóa trước, bộ Chunker cần tuân thủ nghiêm ngặt 4 quy tắc sau:

1. **Quy tắc 1 — Cô lập ranh giới đề mục (Strict Section-Boundary Isolation):**
   * Tuyệt đối **không** gộp toàn bộ các mục của một bài thành một chuỗi văn bản dài rồi mới cắt 512 token.
   * Quá trình cắt Parent-Child chỉ thực hiện **độc lập bên trong từng trường `sections[key]`** của mỗi JSON object. Điều này ngăn chặn triệt để việc đoạn cuối của mục *Chỉ định* bị dính sang đoạn đầu của mục *Chống chỉ định* hoặc *Tác dụng phụ*.
2. **Quy tắc 2 — Đồng nhất đơn vị đếm Token và Ngân sách độ dài (Token Budget):**
   * Sử dụng **từ ghép tiếng Việt tách bởi `underthesea`** (đã chuẩn hóa Unicode NFC) làm đơn vị đếm token chuẩn thay vì đếm khoảng trắng thông thường.
   * **Kích thước chuẩn:**
     * **Parent Chunk:** Tối đa `512` word-tokens (trong đó dành khoảng `35–40` tokens cho tiền tố ngữ cảnh và `470` tokens cho nội dung; bước trượt `stride = 384`, chồng lấp `overlap = 128` tokens nếu mục đó dài hơn 512 tokens).
     * **Child Chunk:** Tối đa `256` word-tokens (trong đó dành `35–40` tokens cho tiền tố ngữ cảnh và `215` tokens cho nội dung; chồng lấp `overlap = 64` tokens).
   * Nếu một đề mục ngắn hơn 256 tokens (ví dụ mục *Bảo quản* hoặc *Chống chỉ định* ngắn), hệ thống tạo đúng **1 Parent = 1 Child** (không nhồi đệm/padding).
3. **Quy tắc 3 — Bảo toàn tính nguyên vẹn của Bảng Markdown và Danh sách bước:**
   * **Đối với Bảng biểu (`| ... |`):** Trong Phác đồ Huyết học/Dị ứng và Dược thư có rất nhiều bảng liều lượng và phân độ. Không bao giờ cắt ngang giữa một hàng (`\n| ... |\n`). Nếu một bảng quá dài vượt quá 256/512 tokens và buộc phải tách thành 2 chunk, bộ chunker phải **tự động sao chép 2 dòng tiêu đề cột (`| Cột 1 | Cột 2 |\n|---|---|`) gắn lên đầu chunk thứ hai** để bảng không bị mất nghĩa.
   * **Đối với Quy trình Ngoại khoa (`Bước 1:`, `Bước 2:`...):** Ưu tiên ngắt đoạn tại ranh giới xuống dòng giữa các bước kỹ thuật thay vì cắt giữa câu mô tả thao tác phẫu thuật.
4. **Quy tắc 4 — Chuẩn hóa Mẫu Tiền tố Ngữ cảnh (Contextual Prefix Templates):**
   * **Dược thư Quốc gia:**
     `[Nguồn: Dược thư QGVN | Thuốc: {drug_name_vn} ({inn_name}) | Nhóm: {drug_class} | Mục: {section_name}]`
   * **Phác đồ Nội khoa / Dược thư chung:**
     `[Nguồn: Phác đồ BYT ({decision_number}) | Chuyên khoa: {specialty} | Bệnh: {entity_name} | Mục: {section_name}]`
   * **Quy trình Kỹ thuật Ngoại khoa:**
     `[Nguồn: Quy trình BYT ({decision_number}) | Chuyên khoa: {specialty} | Kỹ thuật: {entity_name} | Mục: {section_name}]`

---

## PHẦN 2: THIẾT KẾ TỔNG QUAN & CÔNG CỤ TEST NHANH RAG (GIỮ NGUYÊN PIPELINE ZERO-SHOT)

Mục tiêu của bước kiểm thử này là **đo lường thuần túy giá trị đóng góp của kho dữ liệu mới (Dược thư + Phác đồ BYT)** bằng cách đặt cấu hình `LLM + Naive Hybrid RAG` lên cùng bàn cân với cấu hình `LLM + Zero-shot` bạn đã chạy, sử dụng lại 100% file `run_pilot.py` và `evaluate.py`.

### 2.1. Sơ đồ Kiến trúc Pipeline RAG Đánh giá Nhanh

```text
[GIAI ĐOẠN 1: OFFLINE INDEXING - CHẠY 1 LẦN (~5-10 PHÚT)]
  kb_children.jsonl ──┬──► ViHealthBERT / BGE-M3 ──► FAISS IndexFlatIP (Dense)
                      └──► underthesea tokens ─────► BM25Okapi Pickle (Sparse)
  kb_parents.jsonl  ──────► Dictionary Map (parent_id -> text_for_llm)

[GIAI ĐOẠN 2: ONLINE RETRIEVAL & GENERATION - CHẠY TRONG run_pilot.py]
  Câu hỏi q (VM14K / ViMedAQA)
       │
       ├──► BM25 (Top-30 Child) ──┐
       └──► FAISS (Top-30 Child) ─┴──► RRF Fusion (0.6 BM25 + 0.4 Dense)
                                              │
                                       Map sang Parent ID (Khử trùng lặp)
                                              │
                                    Cross-Encoder Reranker (Top-3 hoặc Top-5 Parents)
                                              │
                                Ghép vào Prompt MCQ_RAG_ZEROSHOT
                                              │
                           LLMClient (Vistral-7B / Qwen2.5-7B - giữ nguyên tham số)
                                              │
                               results/vm14k_rag_<model_name>.jsonl
                                              │
                                    python evaluate.py (So sánh trực tiếp với zeroshot)
```

### 2.2. Bộ Công cụ Triển khai Nhanh Nhất (Lightweight & Local/Colab Ready)

Để chạy xong toàn bộ thử nghiệm trong vòng **1 buổi chiều** mà không tốn chi phí API hay phải dựng server phức tạp, bạn nên dùng bộ công cụ gọn nhẹ sau:

| Thành phần | Công cụ / Mô hình đề xuất | Lý do lựa chọn cho bước Test nhanh |
| :--- | :--- | :--- |
| **Tách từ tiếng Việt** | `underthesea` + `unicodedata` (NFC) | Đồng bộ 100% với bộ từ điển và index của đồ án khóa trước. |
| **Sparse Retriever** | `rank_bm25` (`BM25Okapi`) | Cài đặt 1 dòng (`pip install rank-bm25`), chạy trên RAM cực nhanh với quy mô vài chục nghìn chunk. |
| **Dense Embedding** | `demdecuong/vihealthbert-base-word` (hoặc `BAAI/bge-m3`) | `ViHealthBERT` nhẹ (768 chiều), chuyên biệt y tế Việt Nam, encode 50.000 chunk trên GPU T4 Colab chỉ mất ~2 phút. |
| **Vector Database** | `faiss-cpu` / `faiss-gpu` (`IndexFlatIP`) | Tìm kiếm cosine similarity chính xác tuyệt đối, lưu trực tiếp ra 1 file `.bin` gọn nhẹ. |
| **Neural Reranker** | `BAAI/bge-reranker-v2-m3` (qua `FlagEmbedding` hoặc `sentence-transformers`) | Chạy local miễn phí trên GPU T4, hỗ trợ tiếng Việt rất mạnh, thay thế cho Vertex AI/Cohere API tốn phí của khóa trước. |
| **LLM Generator** | `LLMClient` hiện có (`Qwen2.5-7B-Instruct` / `Vistral-7B`) | Giữ nguyên batching, `temperature`, `max_tokens` giống hệt lúc chạy Zero-shot để đảm bảo tính công bằng thực nghiệm. |

---

### 2.3. Cách tích hợp vào `prompts.py`, `run_pilot.py` và `evaluate.py` hiện có

Để không phải viết lại luồng chấm điểm, bạn chỉ cần thêm chiến lược `"rag"` vào 3 vị trí nhỏ sau:

#### 1. Thêm Template Prompt vào `prompts.py`
Giữ nguyên yêu cầu định dạng đầu ra `"Đáp án: <chữ cái>"` để hàm `extract_mcq_answer()` trong `parse_utils.py` hoạt động trơn tru không cần sửa đổi:

```python
MCQ_RAG_ZEROSHOT = """Bạn là một bác sĩ chuyên khoa. Dựa vào các tài liệu y khoa chuẩn dưới đây và kiến thức chuyên môn, hãy trả lời câu hỏi trắc nghiệm sau.

Tài liệu tham khảo:
{context}

Câu hỏi: {question}

Các lựa chọn:
{options_block}

Chỉ trả lời bằng một chữ cái duy nhất tương ứng với đáp án đúng (A, B, C, D...).
Định dạng bắt buộc: "Đáp án: <chữ cái>"
"""
```

#### 2. Thêm tham số `--strategy rag` vào `run_pilot.py`
Trong hàm `main()` của `run_pilot.py`, mở rộng lựa chọn chiến lược:
```python
ap.add_argument("--strategy", choices=["zeroshot", "cot", "rag"], required=True)
```
Trong hàm `run_vm14k()`, nếu `strategy == "rag"`, gọi bộ truy xuất `retriever.retrieve(it["question"], top_k=3)` để lấy `context` ghép vào prompt, đồng thời lưu thêm 2 trường phụ trợ vào file `.jsonl` để phục vụ phân tích lỗi (Error Analysis):
* `"retrieved_ids"`: Danh sách `parent_id` đã truy xuất được (giúp kiểm tra xem hệ thống có bốc trúng đúng bài phác đồ/thuốc hay không).
* `"top_rerank_score"`: Điểm số của đoạn văn top 1 (giúp kiểm tra độ phủ của tài liệu mới trên từng chuyên khoa VM14K).

#### 3. Chạy `evaluate.py` tự động xuất bảng so sánh
Vì file `evaluate.py` của bạn đã được thiết kế để tự động quét mọi file `results/vm14k_*.jsonl` và nhóm kết quả theo `["strategy", "overall_accuracy"]` kèm theo từng mức độ khó (`acc_Easy`, `acc_Moderate`, `acc_Challenging`, `acc_Hard`), nên ngay khi chạy xong `python run_pilot.py --dataset vm14k --strategy rag`, bạn chỉ cần gõ:
```bash
python evaluate.py
```
File `results/summary_mcq.csv` sẽ tự động xuất hiện thêm dòng `rag` đặt ngay cạnh dòng `zeroshot` và `cot`, giúp bạn nhìn thấy ngay mức tăng điểm trên từng chuyên khoa và từng mức độ khó của bộ dữ liệu vừa thu thập!