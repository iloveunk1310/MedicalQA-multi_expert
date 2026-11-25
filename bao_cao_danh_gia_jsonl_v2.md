# Đánh giá lại `clinical_guidelines_unified.jsonl` (v2)

**Phạm vi:** quét tự động toàn bộ **1.425 dòng** (chạy lại các script của lần trước, thêm `readiness.py` và `v2_extra.py`). Sau đó đọc tay **18 bản ghi** trải đều 12/13 file. Chăm sóc giảm nhẹ chỉ được xem qua 1 bản ghi đại diện.

## 1. So sánh trước / sau

| Chỉ số | Bản cũ | Bản mới | |
|---|---|---|---|
| Schema 6 trường | 1424/1424 | 1425/1425 | ✅ (xem lưu ý `title_inferred` ở §3.6) |
| Tiêu đề bài sau dính vào cuối bài trước | ~822 | **≤9** (phần lớn là báo động giả của heuristic) | ✅ đã giải quyết |
| Tiêu đề sinh tự động / bị cắt | ~160 | **40** (`title_inferred`) | ✅ cải thiện mạnh |
| Heading bị dính dòng | 38 bản ghi | **11** | ✅ |
| Watermark trong nội dung | 477 | **0** | ✅ |
| HTML entity (`&lt;`…) | ~2.500 | **0** | ✅ |
| `<!-- image -->` / ký tự `�` / dòng số trang | 125 / 109 / 91 | 9 / 0 / 0 | ✅ |
| Lỗi mất ký tự (CĐHA) | ~1.600 token | ~790 token | 🟡 giảm một nửa |
| Tách âm tiết (Dị ứng) | 157 | 157 | ❌ chưa sửa |
| Bản ghi chỉ có `overview` | 7 | **14** | ❌ **xấu đi** (Dị ứng) |

**Tổng thể:** **1.270/1.425 dòng (89,1%) không có lỗi cấu trúc nào**. Nếu tính cả lỗi nhẹ (OCR, tiêu đề trùng hoặc được suy ra) thì có 1.183 dòng (83%) hoàn toàn sạch.

| File | Số bài | Lỗi cấu trúc | Ghi chú |
|---|---|---|---|
| phau_thuat | 89 | 0 | 6 tiêu đề suy ra |
| Y học hạt nhân | 22 | 0 | ❌ mojibake 22/22 (§3.3) |
| CĐHA | 350 | 3 | còn lỗi OCR |
| GPB-TB | 134 | 3 | |
| Hồi sức | 34 | 3 | đọc mẫu: rất tốt |
| Mắt | 29 | 4 | |
| PHCN | 108 | 4 | |
| CXK | 47 | 4 | |
| Thận | 30 | 5 | |
| Chăm sóc giảm nhẹ | 10 | 4 | độ hạt quá thô |
| Huyết học | 48 | 18 | dính dòng, watermark trong tiêu đề |
| **Gây mê** | 503 | **94** | tiêu đề có số thứ tự đứng đầu, khối tiêu đề lặp lại trong overview |
| **Dị ứng** | 21 | **13** | **regression** |

## 2. Kết quả đọc mẫu

- ✅ **Tốt, sẵn sàng chunk:**
  - `hoi_suc_tich_cuc_008`, `_017`: đủ 5 trường, nội dung đúng trường.
  - `co_xuong_khop_020`, `phau_thuat_002`.
  - `gay_me_003`: tiêu đề đã đầy đủ "QUI TRÌNH GÂY TÊ TỦY SỐNG PHẪU THUẬT BONG HAY ĐỨT…".
  - `chan_doan_hinh_anh_042`: tiêu đề giờ khớp với nội dung (bản cũ bị lệch sang bài khác).
  - `than_tiet_nieu_010`: đúng trường, chỉ thiếu overview.
- 🟡 **Cải thiện một phần:**
  - `mat_015`: lâm sàng đã tách ra được, nhưng `4. ĐIỀU TRỊ …` vẫn nằm trong `diagnosis`, còn `treatment_procedure` rỗng. Lý do: không có dấu câu đứng trước heading nên regex tách dính dòng không bắt được.
  - `huyet_hoc_013`: overview 18,6k ký tự, vẫn có `1. ĐẠI CƯƠNG …` và `3. XẾP LOẠI …` dính trong dòng.
- ❌ **Còn lỗi:**
  - `di_ung_004`: cả bài 9,6k ký tự nằm trong `overview`, 5 trường còn lại rỗng.
  - `phau_thuat_072`: tiêu đề suy ra thành "PHẪU THUẬT CHIẾM 60 GÃY XƯƠNG TỤ CỐT". Tiêu đề thật `## PHẪU THUẬT KHX GÃY XƯƠNG GÓT` lại nằm giữa phần các bước tiến hành do ngắt trang. Ngoài ra `## CHUẨN BỊ` (không có số La Mã) bị rơi vào `indication`.
  - `y_hoc_hat_nhan_010`: chữ bị hỏng mã hóa (`vi rœt`, `vˆng da`, `l‡ch to`).
  - `cham_soc_giam_nhe_005`: mục 5.4 "bệnh phổi giai đoạn cuối" bị xếp vào `diagnosis`, chỉ vì tiêu đề có cụm "GIAI ĐOẠN".

## 3. Lỗi còn lại (theo mức ưu tiên)

### 3.1 Regression ở Dị ứng (13/21 bài)
[`split_level1_sections`](file:///d:/tai_lieu_hoc%20(1)/thesis/markdown_byt/convert_jsonl.py#L477-L481) và [`detect_numbering_regime`](file:///d:/tai_lieu_hoc%20(1)/thesis/markdown_byt/convert_jsonl.py#L450) ở regime `ARABIC_L1` giờ yêu cầu `{VN_UPPER}{2,}`, tức heading phải in hoa toàn bộ. Dị ứng viết heading dạng câu (`## 3. Chẩn đoán`, `## 4. Điề u trị`), nên không tách được → cả bài vào `overview`, hoặc rơi sang regime DECIMAL.
**Sửa:** chấp nhận cả heading viết hoa đầu câu, với điều kiện dòng bắt đầu bằng `#` *và* chứa từ khóa chuẩn (không phân biệt hoa thường), *và* dài ≤ 120 ký tự.

### 3.2 Gây mê: nhiễu tiêu đề
- **75 tiêu đề có số thứ tự đứng đầu** (`06 QUI TRÌNH GÂY MÊ…`) → cần bỏ `^\d+\s+`.
- **128 overview (cùng 9 bài CĐHA) mở đầu bằng chính khối tiêu đề** (`## QUI TRÌNH…`, `## BONG HAY ĐỨT…`) → bỏ khối này khỏi overview vì đã có trong `title`.
- **19 overview còn chứa `CHỈ ĐỊNH/CHUẨN BỊ/CÁC BƯỚC`** và 16 bài có `CÁC BƯỚC` nằm ngoài `treatment`. Đây là những bài heading không có số La Mã.
- 8 tiêu đề fallback vẫn trùng nhau: `QUI TRÌNH GÂY MÊ GÂY MÊ NỘI KHÍ QUẢN GÂY`.

### 3.3 Mojibake ở Y học hạt nhân (22/22 bài, 367 ký tự)
Lỗi font kiểu Mac-Roman: `œ→ú`, `ˆ→à`, `‡→á`, `‹→ã`… Lỗi này có từ bản cũ nhưng lần trước mình chưa phát hiện. Một bảng thay thế ký tự trong `clean_text` là xử lý được.

### 3.4 Heading dính dòng (11 bài, chủ yếu Huyết học và Mắt)
Mở rộng regex: không bắt buộc có dấu câu phía trước, chỉ cần `\s(\d|[IVX]+)\.\s+<TỪ KHÓA IN HOA>` và trước đó có ít nhất khoảng 20 ký tự văn bản.

### 3.5 Chính tả / OCR
- Dị ứng: tách âm tiết (`m ột`, `xuấ t hiệ n`, `Đị nh nghĩ a`) vẫn còn, 157 token trong 6 bài.
- CĐHA: còn khoảng 790 token (`v ng`, `b ng`, `Vật tư y t`, `c p tóc`).
- Còn 4 tiêu đề Huyết học dính watermark: `THIẾU MÁU THIẾU SẮT Ngoc_04/07/2022`, `… Le Van`, `… ngoctlv.kcb_Truong` (×2). Bộ lọc đã làm sạch nội dung nhưng chưa áp dụng cho `title`.
- 10 bài còn "TÀI LIỆU THAM KHẢO" chưa bị cắt (ví dụ viết dính `TÀI LIỆUTHAM KHẢO`) → dùng `TÀI\s*LIỆU\s*THAM\s*KHẢO`.

### 3.6 Nhỏ
- `title_inferred` chỉ có ở 40 dòng → nên thêm `false` cho mọi dòng để schema đồng nhất.
- Chăm sóc giảm nhẹ (10 bài, tới 110k ký tự): ánh xạ 6 trường không có nhiều ý nghĩa. Khi chunk nên tách theo mục `x.y`, đừng tách theo trường.
- GPB-TB vẫn thiếu 12 bài so với mục lục.

## 4. Kết luận: đã sẵn sàng cho chunking chưa?

**Phần lớn đã sẵn sàng (~89% dòng không có lỗi cấu trúc).** Các nhóm sau có thể chunk ngay:
- Phẫu thuật, CĐHA, GPB-TB, PHCN, Hồi sức, Thận, CXK, Mắt: lỗi rải rác, không ảnh hưởng lớn tới việc chunk theo trường.
- Khoảng 400/503 bài Gây mê.

**Nên sửa trước khi chunk** (đều là sửa nhanh bằng regex/bảng thay thế, chạy lại pipeline vài phút):
1. Regression Dị ứng (§3.1). Bắt buộc sửa, vì cả bài nằm trong 1 trường thì metadata trường không còn đúng.
2. Bỏ số thứ tự đầu tiêu đề và khối tiêu đề lặp lại trong overview của Gây mê (§3.2). Hai lỗi này làm bẩn metadata và embedding.
3. Mojibake Y học hạt nhân (§3.3) và watermark trong tiêu đề (§3.5).

**Chấp nhận được, xử lý sau:** OCR còn sót ở CĐHA và Dị ứng, 11 bài dính dòng, độ hạt của Chăm sóc giảm nhẹ. Khi chunk nên:
- Ghi `doc_id` và `section` vào metadata của từng chunk.
- Gắn cờ `needs_review` cho danh sách bản ghi lỗi do `readiness.py` xuất ra, để lọc hoặc đánh giá riêng khi kiểm tra retrieval.
