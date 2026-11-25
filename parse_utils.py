"""
Trích đáp án từ output thô của model.

Thiết kế lại (v2) để sửa các lỗi của bản cũ:
  1. Bản cũ dùng re.search() -> lấy lần khớp ĐẦU TIÊN. Trong CoT, model hay viết
     "Đáp án A sai vì..." khi loại trừ từng phương án -> parser lấy nhầm chữ A.
     Bản mới lấy lần khớp CUỐI CÙNG hợp lệ.
  2. Bản cũ có fallback "chữ cái đứng riêng cuối cùng" -> bắt nhầm "vitamin C",
     "viêm gan B", "nhóm máu A". Bản mới BỎ fallback này; nếu không trích được
     thì trả None (parse_fail) để pipeline xử lý riêng (forced answer), và để
     evaluate.py báo cáo parse_fail_rate thay vì lặng lẽ tính là sai.
  3. Chỉ nhận chữ cái thuộc tập lựa chọn hợp lệ của từng câu (valid_letters).

Các tầng (tier), ưu tiên từ trên xuống; mỗi tầng lấy khớp CUỐI CÙNG hợp lệ:
  "marker" : "FINAL ANSWER: B" / "ĐÁP ÁN CUỐI CÙNG: B" / "... là B"
  "loose"  : "Đáp án: B" / "Đáp án đúng là B" / "Answer: B"  (bắt buộc có ':' hoặc 'là',
             nên "Đáp án A sai vì..." KHÔNG khớp)
  "bare"   : toàn bộ output chỉ là một chữ cái ("B", "(B)", "B."), hoặc output ngắn
             bắt đầu bằng "B. ..." / "B) ..."
"""
import re
import unicodedata

# Marker mặc định (không phân biệt hoa/thường). Marker của prompt hiện tại là
# ASCII "FINAL ANSWER:" để stop-string trong HF generate không phụ thuộc tokenizer.
DEFAULT_MARKERS = [
    r"final\s*answer",
    r"đáp\s*án\s*cuối\s*cùng",
]

_WRAP = r"(?:[\*\(\[\{\"'`_]|\s)*"      # bọc quanh chữ cái: **B**, (B), "B", `B`
_LETTER = r"([A-Ha-h])"                   # bỏ I/J: tránh nhầm đại từ tiếng Anh "I"; valid_letters sẽ lọc tiếp
# Sau chữ cái phải là: (dấu câu/đóng ngoặc/bold)* rồi hết dòng, hoặc mở ngoặc, gạch, hoặc từ nối giải thích.
# => chặn "FINAL ANSWER: A patient...", "Anh", "Ăn" nhưng vẫn nhận "B.", "**B**", "B (Viêm gan)", "B vì ...".
_TAIL = (r"(?=[\*\)\]\}\"'`_\.:,;]*[ \t]*"
         r"(?:\r?\n|$|[\(\[\-–—]|(?:vì|do|bởi|because|since)\b))")
_NOT_WORD = _TAIL                         # giữ tên cũ cho các regex bên dưới
_SEP = r"(?:là\s*[:：]?|[:：\-–—])"       # bắt buộc có ':' hoặc 'là' sau marker


def _norm(text: str) -> str:
    return unicodedata.normalize("NFC", text or "")


def _build_marker_regex(marker: str | None):
    parts = list(DEFAULT_MARKERS)
    if marker:
        m = marker.strip().rstrip(":：").strip()
        if m:
            parts.insert(0, r"\s*".join(re.escape(tok) for tok in m.split()))
    return re.compile(
        rf"(?:{'|'.join(parts)})\s*{_SEP}\s*{_WRAP}{_LETTER}{_NOT_WORD}",
        re.IGNORECASE,
    )


_LOOSE = re.compile(
    rf"(?:đáp\s*án(?:\s*đúng)?\s*(?:là\s*[:：]?|[:：])"
    rf"|answer\s*(?:is\s*[:：]?|[:：]))\s*{_WRAP}{_LETTER}{_NOT_WORD}",
    re.IGNORECASE,
)
_BARE_WHOLE = re.compile(rf"^\s*{_WRAP}{_LETTER}{_WRAP}[\.\):]?\s*$")
_BARE_START = re.compile(rf"^\s*{_WRAP}{_LETTER}\s*[\.\)]\s+\S")


def _last_valid(regex, text, valid):
    letter = None
    for m in regex.finditer(text):
        c = m.group(1).upper()
        if valid is None or c in valid:
            letter = c
    return letter


def parse_mcq_answer(raw_text, valid_letters=None, marker=None):
    """Trả về (letter | None, method). method ∈ {"marker","loose","bare","parse_fail"}."""
    text = _norm(raw_text)
    valid = {v.upper() for v in valid_letters} if valid_letters else None

    letter = _last_valid(_build_marker_regex(marker), text, valid)
    if letter:
        return letter, "marker"

    letter = _last_valid(_LOOSE, text, valid)
    if letter:
        return letter, "loose"

    stripped = text.strip()
    m = _BARE_WHOLE.match(stripped)
    if m and (valid is None or m.group(1).upper() in valid):
        return m.group(1).upper(), "bare"
    if len(stripped) <= 300:
        m = _BARE_START.match(stripped)
        if m and (valid is None or m.group(1).upper() in valid):
            return m.group(1).upper(), "bare"

    return None, "parse_fail"


def extract_mcq_answer(raw_text, valid_letters=None, marker=None):
    """Tương thích ngược với bản cũ: chỉ trả về chữ cái (hoặc None)."""
    return parse_mcq_answer(raw_text, valid_letters, marker)[0]


def split_at_marker(text: str, marker: str | None):
    """
    Cắt text tại marker đầu tiên. Trả về (phần_suy_luận, tìm_thấy_marker, phần_sau_marker).
    Dùng để (a) bỏ phần model tự viết sau marker khi chấm bằng logit, (b) tính
    'written_letter' để so với đáp án chấm bằng logit.
    """
    text = _norm(text)
    parts = list(DEFAULT_MARKERS)
    if marker:
        m = marker.strip().rstrip(":：").strip()
        if m:
            parts.insert(0, r"\s*".join(re.escape(tok) for tok in m.split()))
    rx = re.compile(rf"(?:{'|'.join(parts)})\s*[:：]", re.IGNORECASE)
    m = rx.search(text)
    if not m:
        return text, False, ""
    return text[: m.start()], True, text[m.end():]


# ---------------------------------------------------------------------------
# ViMedAQA (abstractive)
# ---------------------------------------------------------------------------
_ANSWER_TAG = re.compile(r"(?:^|\n)[ \t]*[\*#]*[ \t]*trả\s*lời[ \t]*[:：][ \t]*\**", re.IGNORECASE)


def extract_abstractive_answer(raw_text: str) -> str:
    """
    Prompt CoT của ViMedAQA yêu cầu model trích bằng chứng rồi viết "Trả lời: ...".
    Khi chấm BLEU/ROUGE/BERTScore phải so phần SAU "Trả lời:" với đáp án chuẩn,
    không phải toàn bộ output (gồm cả phần trích bằng chứng).
    """
    text = _norm(raw_text).strip()
    matches = list(_ANSWER_TAG.finditer(text))
    if matches:
        return text[matches[-1].end():].strip()
    return text
