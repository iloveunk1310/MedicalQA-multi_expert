"""
Template prompt tiếng Việt cho 2 loại tác vụ x 2 chiến lược = 4 template.
Giữ format output có cấu trúc rõ ràng để parse_utils.py trích đáp án dễ dàng.
"""

MCQ_ZEROSHOT = """Bạn là một bác sĩ chuyên khoa. Hãy trả lời câu hỏi trắc nghiệm y khoa sau.

Câu hỏi: {question}

Các lựa chọn:
{options_block}

Chỉ trả lời bằng một chữ cái duy nhất tương ứng với đáp án đúng (A, B, C, D...).
Định dạng bắt buộc: "Đáp án: <chữ cái>"
"""

MCQ_RAG_ZEROSHOT = """Bạn là một bác sĩ chuyên khoa. Dựa vào các tài liệu y khoa chuẩn dưới đây và kiến thức chuyên môn, hãy trả lời câu hỏi trắc nghiệm sau.

Tài liệu tham khảo:
{context}

Câu hỏi: {question}

Các lựa chọn:
{options_block}

Chỉ trả lời bằng một chữ cái duy nhất tương ứng với đáp án đúng (A, B, C, D...).
Định dạng bắt buộc: "Đáp án: <chữ cái>"
"""

MCQ_COT = """Bạn là bác sĩ chuyên khoa đang làm bài thi y khoa. Hãy suy luận NGẮN GỌN rồi chọn đúng một đáp án.
 
Câu hỏi: {question}
 
Các lựa chọn:
{options_block}
 
Cách làm bài:
1. Xác định đề yêu cầu chọn đáp án ĐÚNG hay SAI / KHÔNG ĐÚNG / NGOẠI TRỪ (chú ý các từ phủ định).
2. Nêu kiến thức y khoa then chốt liên quan (định nghĩa, cơ chế, tiêu chuẩn chẩn đoán, phác đồ, liều dùng...) và áp dụng vào dữ kiện trong câu hỏi.
3. Đối chiếu với các lựa chọn; chỉ nhắc đến một lựa chọn khi cần loại trừ nó.
Kết luận bằng chữ cái tương ứng với đáp án đúng (A, B, C, D...).
Phần suy luận không quá khoảng 150 từ.

Quy tắc đầu ra (bắt buộc):
- Dòng cuối cùng phải đúng định dạng: FINAL ANSWER: <chữ cái>
- Không in đậm, không viết thêm gì sau dòng đó.
- Luôn chọn đúng một đáp án, kể cả khi không chắc chắn.
"""

ABSTRACTIVE_ZEROSHOT = """Bạn là một bác sĩ chuyên khoa. Dựa vào đoạn tài liệu y khoa dưới đây, hãy trả lời câu hỏi của bệnh nhân một cách chính xác, ngắn gọn.

Tài liệu tham khảo:
{context}

Câu hỏi: {question}

Trả lời (chỉ dựa trên tài liệu trên, không suy diễn thêm thông tin không có trong tài liệu):
"""

ABSTRACTIVE_COT = """Bạn là một bác sĩ chuyên khoa. Dựa vào đoạn tài liệu y khoa dưới đây, hãy trả lời câu hỏi của bệnh nhân.

Tài liệu tham khảo:
{context}

Câu hỏi: {question}

Trước tiên, hãy trích ra (1-2 câu) phần thông tin trong tài liệu liên quan trực tiếp đến câu hỏi.
Sau đó, viết câu trả lời cuối cùng, bắt đầu bằng "Trả lời:".
"""


def extract_option_pairs(options) -> list[tuple[str, str]]:
    """Chuẩn hóa options dạng list hoặc dict thành danh sách cặp (chữ cái, nội dung)."""
    if isinstance(options, dict):
        return [(str(k).strip().upper(), str(v)) for k, v in options.items()]
    return [(chr(ord("A") + i), str(v)) for i, v in enumerate(options)]


def format_mcq_prompt(question: str, options, strategy: str, context: str = "") -> str:
    pairs = extract_option_pairs(options)
    options_block = "\n".join(f"{k}. {v}" for k, v in pairs)

    if strategy == "rag":
        return MCQ_RAG_ZEROSHOT.format(
            context=context or "",
            question=question,
            options_block=options_block,
        )
    template = MCQ_COT if strategy == "cot" else MCQ_ZEROSHOT
    return template.format(question=question, options_block=options_block)


def format_abstractive_prompt(question: str, context: str, strategy: str) -> str:
    template = ABSTRACTIVE_COT if strategy == "cot" else ABSTRACTIVE_ZEROSHOT
    return template.format(question=question, context=context)
