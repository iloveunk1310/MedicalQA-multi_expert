"""Unit tests for chunking modules (implemented with stdlib unittest)."""

import json
from pathlib import Path
import tempfile
import unicodedata
import unittest
from unittest.mock import patch

from chunking.build_kb import build_kb, has_broken_table_header

from chunking.config import ChunkConfig
from chunking.normalize import (
    compact_markdown_table_line,
    flatten_side_effect,
    is_table_line,
    is_table_separator,
    normalize_section_text,
    normalize_unicode,
)
from chunking.prefix import (
    build_contextual_prefix,
    build_short_prefix,
    clean_drug_class,
    extract_entity_name,
    get_section_name,
    truncate_heading,
)
from chunking.splitter import (
    split_parent_to_children,
    split_section_to_parents,
    split_text_recursive,
)
from chunking.tokenize_vi import ViHealthBertCounter, count_vi_words, segment_vi


class TestNormalize(unittest.TestCase):
    def test_unicode_nfc(self) -> None:
        nfd = unicodedata.normalize("NFD", "Tăng huyết áp điều trị")
        self.assertNotEqual(nfd, unicodedata.normalize("NFC", nfd))
        self.assertEqual(normalize_unicode(nfd), unicodedata.normalize("NFC", nfd))

    def test_flatten_side_effect_ordering_and_no_double_label(self) -> None:
        raw = {
            "huong_dan_xu_tri": "Ngừng thuốc nếu nặng.",
            "hiem_gap": "Sốc phản vệ.",
            "chung": "Dung nạp tốt.",
            "it_gap": "Ít gặp, 1/1000 < ADR < 1/100\nĐau đầu.",
            "thuong_gap": "Thường gặp, ADR > 1/100\nBuồn nôn.",
            "khac": "Phù nhẹ.",
        }
        out = flatten_side_effect(raw)
        blocks = out.split("\n\n")
        self.assertEqual(len(blocks), 6)
        self.assertTrue(blocks[0].startswith("Thường gặp, ADR > 1/100"))
        self.assertFalse(blocks[0].startswith("Thường gặp:\nThường gặp"))
        self.assertTrue(blocks[1].startswith("Ít gặp, 1/1000"))
        self.assertTrue(blocks[2].startswith("Hiếm gặp:\nSốc phản vệ."))
        self.assertTrue(blocks[3].startswith("Tổng quan ADR:\nDung nạp tốt."))
        self.assertTrue(blocks[4].startswith("Hướng dẫn xử trí ADR:\nNgừng thuốc"))
        self.assertTrue(blocks[5].startswith("khac:\nPhù nhẹ."))

    def test_markdown_table_padding_and_separator_compaction(self) -> None:
        header = "|   Thuốc                     |   Liều lượng                    |"
        sep = "|-----------------------------|---------------------------------|"
        row = "|   Paracetamol               |   500 mg   x   3 lần/ngày       |"
        raw = f"{header}\n{sep}\n{row}\n\n\n\nGhi chú   "
        normalized = normalize_section_text(raw)
        lines = normalized.splitlines()
        self.assertEqual(lines[0], "| Thuốc | Liều lượng |")
        self.assertEqual(lines[1], "|---|---|")
        self.assertEqual(lines[2], "| Paracetamol | 500 mg x 3 lần/ngày |")
        self.assertEqual(lines[3], "")
        self.assertEqual(lines[4], "Ghi chú")
        self.assertTrue(is_table_line(lines[0]))
        self.assertTrue(is_table_separator(lines[1]))
        self.assertEqual(compact_markdown_table_line(sep), "|---|---|")

    def test_empty_and_whitespace_sections(self) -> None:
        self.assertEqual(normalize_section_text(None), "")
        self.assertEqual(normalize_section_text("   \n\t  "), "")
        self.assertEqual(normalize_section_text({"thuong_gap": "  ", "it_gap": ""}, "side_effect"), "")


class TestPrefix(unittest.TestCase):
    def test_drug_monograph_prefix_dedup_and_fallbacks(self) -> None:
        doc_same_inn = {
            "doc_id": "DTQGVN2018_DRUG_0001",
            "doc_type": "drug_monograph",
            "drug_name_vn": "Abacavir",
            "inn_name": "abacavir",
            "drug_class": "Thuốc kháng virus",
        }
        p1 = build_contextual_prefix(doc_same_inn, "indication")
        self.assertEqual(
            p1,
            "[Nguồn: Dược thư QGVN 2018 | Thuốc: Abacavir | Nhóm: Thuốc kháng virus | Mục: Chỉ định]",
        )
        self.assertNotIn("(abacavir)", p1.lower())

        doc_diff_inn_no_class = {
            "doc_id": "DTQGVN2018_DRUG_0250",
            "doc_type": "drug_monograph",
            "drug_name_vn": "Natri clorid",
            "inn_name": "Sodium chloride",
            "drug_class": "",
        }
        p2 = build_contextual_prefix(doc_diff_inn_no_class, "side_effect")
        self.assertEqual(
            p2,
            "[Nguồn: Dược thư QGVN 2018 | Thuốc: Natri clorid (Sodium chloride) | Mục: Tác dụng không mong muốn (ADR)]",
        )
        self.assertNotIn("Nhóm:", p2)

        doc_group_fallback = {
            "doc_id": "DTQGVN2018_DRUG_0061",
            "doc_type": "drug_monograph",
            "drug_name_vn": "Atorvastatin",
            "inn_name": "Atorvastatin",
            "drug_class": "",
            "group_monograph": "Các statin",
        }
        p3 = build_contextual_prefix(doc_group_fallback, "pharmacology")
        self.assertIn("Nhóm: Các statin", p3)

    def test_guideline_and_procedure_prefixes(self) -> None:
        cg = {
            "doc_id": "byt_di_ung_001",
            "doc_type": "clinical_guideline",
            "specialty": "Dị ứng - Miễn dịch",
            "title": "DỊ ỨNG THUỐC",
        }
        p_cg = build_contextual_prefix(cg, "treatment_procedure")
        self.assertEqual(
            p_cg,
            "[Nguồn: Hướng dẫn chẩn đoán và điều trị BYT | Chuyên khoa: Dị ứng - Miễn dịch | Bệnh: DỊ ỨNG THUỐC | Mục: Điều trị]",
        )

        sp = {
            "doc_id": "byt_phau_thuat_001",
            "doc_type": "surgical_procedure",
            "specialty": "Ngoại khoa - Phẫu thuật",
            "title": "PHẪU THUẬT THÁO KHỚP VAI",
        }
        p_sp = build_contextual_prefix(sp, "treatment_procedure")
        self.assertEqual(
            p_sp,
            "[Nguồn: Quy trình kỹ thuật BYT | Chuyên khoa: Ngoại khoa - Phẫu thuật | Kỹ thuật: PHẪU THUẬT THÁO KHỚP VAI | Mục: Chuẩn bị và các bước tiến hành]",
        )
        self.assertNotIn("None", p_sp)
        self.assertNotIn("()", p_sp)

        tp = {
            "doc_id": "byt_gt_001",
            "doc_type": "technical_procedure",
            "specialty": "Giải phẫu bệnh",
            "title": "Nhuộm mô",
        }
        p_tp = build_contextual_prefix(tp, "monitoring_complications")
        self.assertIn("Mục: Theo dõi và xử trí tai biến", p_tp)

    def test_heading_suffix_dedup_and_truncation(self) -> None:
        self.assertEqual(get_section_name("clinical_guideline", "overview", "## Đại cương"), "Đại cương")
        self.assertEqual(
            get_section_name("clinical_guideline", "overview", "## 4.1. Đánh giá đau"),
            "Đại cương > 4.1. Đánh giá đau",
        )
        long_h = "## " + "Nguyên tắc điều trị triệu chứng đau khổ thể chất kéo dài ở người bệnh nặng"
        truncated = truncate_heading(long_h, max_chars=40)
        self.assertLessEqual(len(truncated), 40)
        self.assertTrue(truncated.endswith("..."))

        self.assertEqual(
            build_short_prefix(extract_entity_name({"doc_type": "drug_monograph", "drug_name_vn": "Abacavir"}), "Chỉ định"),
            "Abacavir - Chỉ định: ",
        )

    def test_oversized_leaked_drug_class_is_cleaned_and_capped(self) -> None:
        leaked = "Vitamin nhóm B. Dạng dùng và hàm lư ợng " + "Viên nén 10 mg. " * 200
        self.assertEqual(clean_drug_class(leaked), "Vitamin nhóm B")

        doc = {
            "doc_id": "DTQGVN2018_DRUG_0564",
            "doc_type": "drug_monograph",
            "drug_name_vn": "Pyridoxin hydroclorid",
            "inn_name": "Pyridoxine hydrochloride",
            "drug_class": leaked,
        }
        pfx = build_contextual_prefix(doc, "indication")
        self.assertIn("Nhóm: Vitamin nhóm B", pfx)
        self.assertNotIn("Viên nén 10 mg", pfx)

        parents = split_section_to_parents(
            doc,
            "indication",
            "Phòng và điều trị thiếu hụt vitamin B6 ở bệnh nhân.",
            ChunkConfig(),
        )
        self.assertEqual(len(parents), 1)
        self.assertLessEqual(parents[0]["word_count"], 512)


class _MockTokenizer:
    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        tokens = text.split()
        extra = 2 if add_special_tokens else 0
        return list(range(len(tokens) + extra))


class TestTokenizeVi(unittest.TestCase):
    def test_segment_vi_preserves_newlines_and_nfc(self) -> None:
        nfd = unicodedata.normalize("NFD", "tăng huyết áp\nbệnh nhân điều trị")
        seg = segment_vi(nfd)
        lines = seg.split("\n")
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0], "tăng huyết_áp")
        self.assertIn("bệnh_nhân", lines[1])
        self.assertIn("điều_trị", lines[1])
        self.assertEqual(seg, unicodedata.normalize("NFC", seg))
        self.assertEqual(count_vi_words(nfd), 4)

    def test_vihealthbert_counter_with_mock_tokenizer(self) -> None:
        counter = ViHealthBertCounter(tokenizer=_MockTokenizer())
        self.assertEqual(counter.count_subwords("tăng huyết áp", add_special_tokens=True), 4)
        self.assertEqual(counter.count_subwords("tăng huyết áp", add_special_tokens=False), 2)

    def test_vihealthbert_counter_raises_explicit_error_on_load_failure(self) -> None:
        with patch("transformers.AutoTokenizer.from_pretrained", side_effect=OSError("offline")):
            with self.assertRaises(RuntimeError) as ctx:
                ViHealthBertCounter(model_name="demdecuong/vihealthbert-base-word")
            self.assertIn("--child-length words", str(ctx.exception))


class TestSplitterRecursive(unittest.TestCase):
    @staticmethod
    def _ws_len(text: str) -> int:
        return len(text.split())

    def test_long_markdown_table_repeats_headers_and_respects_budget(self) -> None:
        header = "| Thuốc | Liều dùng |"
        sep = "|---|---|"
        rows = [f"| Thuốc_{i} | Dùng {i} mg mỗi ngày sau ăn |" for i in range(1, 25)]
        table_text = "\n".join([header, sep] + rows)

        chunks = split_text_recursive(table_text, budget=35, length_fn=self._ws_len)
        self.assertGreater(len(chunks), 2)
        for ch in chunks:
            self.assertLessEqual(self._ws_len(ch), 35)
            lines = ch.splitlines()
            self.assertEqual(lines[0], header)
            self.assertEqual(lines[1], sep)
            self.assertGreaterEqual(len(lines), 3)

    def test_oversized_single_table_row_and_degenerate_header(self) -> None:
        # Oversized single row
        header = "| Tiêu chí | Mô tả |"
        sep = "|---|---|"
        long_row = "| A | " + " ".join([f"triệu_chứng_{i}." for i in range(45)]) + " |"
        chunks = split_text_recursive(f"{header}\n{sep}\n{long_row}", budget=25, length_fn=self._ws_len)
        self.assertGreater(len(chunks), 1)
        for ch in chunks:
            self.assertLessEqual(self._ws_len(ch), 25)
            lines = ch.splitlines()
            self.assertEqual(lines[0], header)
            self.assertEqual(lines[1], sep)

        # Degenerate oversized header (> 60% of budget)
        big_header = "| " + " ".join([f"cột_{i}" for i in range(30)]) + " | giá_trị |"
        big_tbl = f"{big_header}\n|---|---|\n| nội_dung ngắn | 10 mg |"
        chunks_deg = split_text_recursive(big_tbl, budget=25, length_fn=self._ws_len)
        self.assertGreater(len(chunks_deg), 1)
        for ch in chunks_deg:
            self.assertLessEqual(self._ws_len(ch), 25)
            self.assertNotIn("|---|", ch)

    def test_procedure_steps_and_sentence_word_fallback(self) -> None:
        steps = "\n".join(
            [f"Bước {i}: Chuẩn bị dụng cụ và sát khuẩn vùng phẫu thuật số {i}." for i in range(1, 10)]
        )
        chunks = split_text_recursive(steps, budget=28, length_fn=self._ws_len)
        self.assertGreater(len(chunks), 2)
        for ch in chunks:
            self.assertLessEqual(self._ws_len(ch), 28)
            for ln in ch.splitlines():
                self.assertTrue(ln.startswith("Bước "))

        single_long_sentence = " ".join([f"từ_{i}" for i in range(60)])
        word_chunks = split_text_recursive(single_long_sentence, budget=15, length_fn=self._ws_len)
        self.assertEqual(len(word_chunks), 4)
        for ch in word_chunks:
            self.assertLessEqual(self._ws_len(ch), 15)


class _NonlinearMockTokenizer:
    """Mock tokenizer adding extra subwords when hyphen/colon appears, to exercise post-check repair."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        parts = text.replace(":", " : ").replace("-", " - ").split()
        extra = 2 if add_special_tokens else 0
        return list(range(len(parts) + extra))


class TestParentChildSplitter(unittest.TestCase):
    def test_short_section_produces_one_parent_and_one_child(self) -> None:
        doc = {
            "doc_id": "DTQGVN2018_DRUG_0001",
            "doc_type": "drug_monograph",
            "drug_name_vn": "Abacavir",
            "inn_name": "Abacavir",
            "drug_class": "Thuốc kháng virus",
        }
        text = "Điều trị nhiễm HIV ở người lớn và trẻ em phối hợp với các thuốc kháng retrovirus khác."
        cfg = ChunkConfig(parent_max_words=512, child_max_subwords=256, child_overlap_subwords=48)
        counter = ViHealthBertCounter(tokenizer=_MockTokenizer())

        parents = split_section_to_parents(doc, "indication", text, cfg)
        self.assertEqual(len(parents), 1)
        self.assertLessEqual(parents[0]["word_count"], 512)

        short_prefix = build_short_prefix("Abacavir", parents[0]["section_name"])
        children = split_parent_to_children(parents[0]["text"], short_prefix, cfg, counter=counter)
        self.assertEqual(len(children), 1)
        self.assertEqual(children[0]["text"], text)
        self.assertLessEqual(children[0]["subword_count"], 256)

    def test_long_section_non_overlapping_parents_and_bounded_children(self) -> None:
        doc = {
            "doc_id": "byt_test_001",
            "doc_type": "clinical_guideline",
            "specialty": "Nội khoa",
            "title": "Bệnh mẫu",
        }
        # Build ~750 words across two ## headings
        p1_body = " ".join([f"Câu số {i} mô tả triệu chứng lâm sàng rõ ràng." for i in range(1, 50)])
        p2_body = " ".join([f"Câu điều trị {i} nêu phác đồ chuẩn cho bệnh nhân." for i in range(1, 50)])
        section_text = f"## 1. Triệu chứng\n{p1_body}\n\n## 2. Điều trị cụ thể\n{p2_body}"

        cfg = ChunkConfig(parent_max_words=220, child_max_subwords=70, child_overlap_subwords=14)
        counter = ViHealthBertCounter(tokenizer=_NonlinearMockTokenizer())

        parents = split_section_to_parents(doc, "overview", section_text, cfg)
        self.assertGreater(len(parents), 2)
        for p in parents:
            self.assertLessEqual(p["word_count"], 220)
            self.assertLessEqual(count_vi_words(f"{p['contextual_prefix']}\n{p['text']}"), 220)

        # Verify parents do not overlap each other (each sentence index appears in only one parent)
        self.assertTrue(any("1. Triệu chứng" in p["section_name"] for p in parents))
        self.assertTrue(any("2. Điều trị cụ thể" in p["section_name"] for p in parents))
        # Verify "Câu số 1 " only appears once across all parents
        occurrences = sum(p["text"].count("Câu số 1 ") for p in parents)
        self.assertEqual(occurrences, 1)

        # Check children stay strictly inside parent and <= 70 subwords
        for p in parents:
            sp = build_short_prefix("Bệnh mẫu", p["section_name"])
            children = split_parent_to_children(p["text"], sp, cfg, counter=counter)
            self.assertGreaterEqual(len(children), 1)
            for ch in children:
                self.assertLessEqual(ch["subword_count"], 70)
                # Child content without overlap/newlines must come from parent text
                for line in ch["text"].splitlines():
                    if line.strip():
                        self.assertIn(line.strip()[:20], p["text"])

    def test_table_child_split_never_creates_broken_header_with_overlap(self) -> None:
        header = "| Thuốc | Liều dùng |"
        sep = "|---|---|"
        rows = [f"| Thuốc_{i} | Liều {i} mg/ngày |" for i in range(1, 20)]
        parent_text = "Danh mục thuốc điều trị:\n\n" + "\n".join([header, sep] + rows)
        cfg = ChunkConfig(child_max_subwords=45, child_overlap_subwords=9)
        counter = ViHealthBertCounter(tokenizer=_MockTokenizer())

        children = split_parent_to_children(parent_text, "Bệnh - Điều trị: ", cfg, counter=counter)
        self.assertGreater(len(children), 2)
        for ch in children:
            self.assertLessEqual(ch["subword_count"], 45)
            lines = ch["text"].splitlines()
            table_idx = [i for i, ln in enumerate(lines) if is_table_line(ln)]
            if table_idx:
                self.assertGreaterEqual(len(table_idx), 2)
                self.assertTrue(is_table_separator(lines[table_idx[1]]))


class TestBuildKBEndToEnd(unittest.TestCase):
    def test_determinism_section_isolation_and_fk_integrity(self) -> None:
        table_rows = "\n".join(
            [f"| Thuốc_{i} | Dùng {i} mg/ngày |" for i in range(1, 22)]
        )
        guidelines = [
            {
                "doc_id": "byt_cg_001",
                "source_file": "cg_001.md",
                "specialty": "Dị ứng",
                "doc_type": "clinical_guideline",
                "title": "DỊ ỨNG MẪU",
                "title_inferred": False,
                "sections": {
                    "overview": "Khái niệm bệnh dị ứng ngắn gọn.",
                    "clinical_paraclinical": "   ",
                    "treatment_procedure": f"| Thuốc | Liều |\n|---|---|\n{table_rows}",
                },
            },
            {
                "doc_id": "byt_sp_001",
                "source_file": "sp_001.md",
                "specialty": "Ngoại khoa",
                "doc_type": "surgical_procedure",
                "title": "PHẪU THUẬT MẪU",
                "title_inferred": False,
                "sections": {
                    "overview": "Đại cương phẫu thuật.",
                    "clinical_paraclinical": "",
                    "diagnosis": "",
                    "treatment_procedure": "Bước 1: Chuẩn bị bệnh nhân.\nBước 2: Tiến hành phẫu thuật.",
                },
            },
        ]
        drugs = [
            {
                "doc_id": "DTQGVN2018_DRUG_0001",
                "source": "Dược thư Quốc gia Việt Nam 2018",
                "decision_number": "2033/QĐ-BYT",
                "effective_date": "2015-05-29",
                "doc_type": "drug_monograph",
                "specialty_vm14k": ["Truyền nhiễm"],
                "drug_name_vn": "Abacavir",
                "inn_name": "Abacavir",
                "atc_code": ["J05AF06"],
                "drug_class": "Thuốc kháng virus",
                "source_page": 101,
                "sections": {
                    "indication": "Điều trị HIV phối hợp.",
                    "side_effect": {
                        "thuong_gap": "Thường gặp, ADR > 1/100\nBuồn nôn.",
                        "it_gap": "Đau khớp.",
                        "hiem_gap": "",
                    },
                    "storage": "   ",
                },
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            g_path = tmp / "guidelines.jsonl"
            d_path = tmp / "drugs.jsonl"
            g_path.write_text(
                "\n".join(json.dumps(x, ensure_ascii=False) for x in guidelines) + "\n",
                encoding="utf-8",
            )
            d_path.write_text(
                "\n".join(json.dumps(x, ensure_ascii=False) for x in drugs) + "\n",
                encoding="utf-8",
            )

            out_a = tmp / "out_a"
            out_b = tmp / "out_b"
            cfg_a = ChunkConfig(
                guidelines_path=str(g_path),
                drugs_path=str(d_path),
                out_dir=str(out_a),
                child_length_mode="words",
                child_max_words=40,
                child_overlap_words=8,
            )
            cfg_b = ChunkConfig(
                guidelines_path=str(g_path),
                drugs_path=str(d_path),
                out_dir=str(out_b),
                child_length_mode="words",
                child_max_words=40,
                child_overlap_words=8,
            )

            rep_a = build_kb(cfg_a)
            rep_b = build_kb(cfg_b)

            self.assertEqual(
                (out_a / "kb_parents.jsonl").read_bytes(),
                (out_b / "kb_parents.jsonl").read_bytes(),
            )
            self.assertEqual(
                (out_a / "kb_children.jsonl").read_bytes(),
                (out_b / "kb_children.jsonl").read_bytes(),
            )
            self.assertEqual(rep_a["docs_total"], 3)
            self.assertEqual(rep_a["skipped_empty_sections"], 4)
            self.assertEqual(rep_a["parents_over_512"], 0)
            self.assertEqual(rep_a["children_over_limit"], 0)
            self.assertEqual(rep_a["children_with_broken_table_header"], 0)
            self.assertEqual(rep_a["orphan_children"], 0)
            self.assertEqual(rep_a["childless_parents"], 0)
            self.assertEqual(rep_a["valid_parent_id_ratio"], 1.0)
            self.assertEqual(rep_b["total_children"], rep_a["total_children"])

            # Also verify subword mode with mock counter
            out_sw = tmp / "out_sw"
            cfg_sw = ChunkConfig(
                guidelines_path=str(g_path),
                drugs_path=str(d_path),
                out_dir=str(out_sw),
                child_length_mode="subword",
                child_max_subwords=50,
                child_overlap_subwords=10,
            )
            rep_sw = build_kb(cfg_sw, counter=ViHealthBertCounter(tokenizer=_MockTokenizer()))
            self.assertEqual(rep_sw["children_over_limit"], 0)
            self.assertEqual(rep_sw["children_with_broken_table_header"], 0)
            self.assertEqual(rep_sw["orphan_children"], 0)

            parents = [
                json.loads(ln)
                for ln in (out_sw / "kb_parents.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            children = [
                json.loads(ln)
                for ln in (out_sw / "kb_children.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            parent_ids = {p["parent_id"] for p in parents}
            for c in children:
                self.assertIn(c["parent_id"], parent_ids)
                self.assertTrue(c["text_for_dense_seg"])
                self.assertFalse(has_broken_table_header(c["text"]))


if __name__ == "__main__":
    unittest.main()
