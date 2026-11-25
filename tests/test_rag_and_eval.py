"""Unit tests for RAG encoder/index/retriever, prompts, run_pilot fixes, and evaluate.py."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unicodedata
import unittest

import faiss
import numpy as np
import torch

from evaluate import clean_records, evaluate_mcq
from prompts import extract_option_pairs, format_mcq_prompt
from rag.build_index import build_dense_index
from rag.config import RAGConfig, RetrievalResult
from rag.encoder import ViHealthBertEncoder
from rag.retriever import DenseRetriever, resolve_candidates_to_contexts
import run_pilot


class _MockBatchTokenizer:
    """Deterministic lightweight tokenizer supporting max_length, padding, and truncation."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(
        self,
        texts: list[str],
        max_length: int = 256,
        truncation: bool = True,
        padding: bool = True,
        return_tensors: str = "pt",
    ) -> dict[str, torch.Tensor]:
        self.calls.append({"texts": list(texts), "max_length": max_length, "truncation": truncation})
        token_lists: list[list[int]] = []
        for t in texts:
            ids = [1] + [((sum(ord(c) for c in tok) % 50) + 3) for tok in t.split()] + [2]
            if truncation and len(ids) > max_length:
                ids = ids[:max_length]
            token_lists.append(ids)

        max_len = max(len(x) for x in token_lists) if padding else len(token_lists[0])
        input_ids = []
        attention_mask = []
        for ids in token_lists:
            pad_n = max_len - len(ids)
            input_ids.append(ids + [0] * pad_n)
            attention_mask.append([1] * len(ids) + [0] * pad_n)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


class _MockTransformerModel:
    """Mock transformer returning distinct hidden states per token id and huge values on pad tokens."""

    def __init__(self, dim: int = 4) -> None:
        self.dim = dim

    def to(self, device: str):
        return self

    def eval(self):
        return self

    def __call__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        bsz, seqlen = input_ids.shape
        hidden = torch.zeros((bsz, seqlen, self.dim), dtype=torch.float32)
        for b in range(bsz):
            for s in range(seqlen):
                tid = int(input_ids[b, s].item())
                if tid == 0:
                    # Huge values on PAD tokens to ensure mean pooling masks them out
                    hidden[b, s] = torch.tensor([999.0] * self.dim)
                else:
                    hidden[b, s, tid % self.dim] = float(tid + 1)
                    hidden[b, s, 0] += 1.0
        return SimpleNamespace(last_hidden_state=hidden)


class TestEncoderAndBuildIndex(unittest.TestCase):
    def test_mean_and_cls_pooling_l2_normalized_and_padding_excluded(self) -> None:
        tok = _MockBatchTokenizer()
        model = _MockTransformerModel(dim=4)

        enc_mean = ViHealthBertEncoder(
            pooling="mean", max_length=8, device="cpu", model=model, tokenizer=tok
        )
        # Verify query preprocessing applies NFC + underthesea segmentation
        nfd_q = unicodedata.normalize("NFD", "Điều trị tăng huyết áp")
        seg_q = enc_mean.preprocess_query(nfd_q)
        self.assertEqual(seg_q, unicodedata.normalize("NFC", seg_q))
        self.assertIn("huyết_áp", seg_q)

        # Verify truncation and L2 normalization + padding exclusion
        vecs = enc_mean.encode_segmented_texts(
            ["tăng huyết_áp", "điều_trị bệnh tăng huyết_áp kéo_dài nhiều tháng liền nữa"],
            batch_size=2,
        )
        self.assertEqual(vecs.shape, (2, 4))
        self.assertEqual(vecs.dtype, np.float32)
        norms = np.linalg.norm(vecs, axis=1)
        np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-5)
        self.assertLess(np.max(np.abs(vecs)), 1.01)
        self.assertEqual(tok.calls[-1]["max_length"], 8)

        # Verify CLS pooling differs from mean pooling and is also L2-normalized
        enc_cls = ViHealthBertEncoder(
            pooling="cls", max_length=8, device="cpu", model=model, tokenizer=tok
        )
        cls_vecs = enc_cls.encode_segmented_texts(["tăng huyết_áp"], batch_size=1)
        np.testing.assert_allclose(np.linalg.norm(cls_vecs, axis=1), [1.0], atol=1e-5)
        self.assertFalse(np.allclose(vecs[0], cls_vecs[0]))

    def test_build_dense_index_writes_faiss_ids_and_manifest(self) -> None:
        tok = _MockBatchTokenizer()
        model = _MockTransformerModel(dim=4)
        enc = ViHealthBertEncoder(pooling="mean", max_length=16, device="cpu", model=model, tokenizer=tok)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            kb_dir = tmp / "kb"
            idx_dir = tmp / "index"
            kb_dir.mkdir()

            children = [
                {
                    "child_id": "doc1__overview__p01__c01",
                    "parent_id": "doc1__overview__p01",
                    "text_for_dense": "Bệnh A: tăng huyết áp",
                    "text_for_dense_seg": "Bệnh A : tăng huyết_áp",
                },
                {
                    "child_id": "doc1__overview__p01__c02",
                    "parent_id": "doc1__overview__p01",
                    "text_for_dense": "Bệnh A: suy tim",
                    "text_for_dense_seg": "Bệnh A : suy tim",
                },
                {
                    "child_id": "doc2__indication__p01__c01",
                    "parent_id": "doc2__indication__p01",
                    "text_for_dense": "Thuốc B: chỉ định",
                    "text_for_dense_seg": "Thuốc B : chỉ_định",
                },
            ]
            (kb_dir / "kb_children.jsonl").write_text(
                "\n".join(json.dumps(c, ensure_ascii=False) for c in children) + "\n",
                encoding="utf-8",
            )

            manifest = build_dense_index(
                kb_dir=str(kb_dir),
                index_dir=str(idx_dir),
                pooling="mean",
                max_length=16,
                batch_size=2,
                device="cpu",
                encoder=enc,
            )

            self.assertEqual(manifest["n_children"], 3)
            self.assertEqual(manifest["embedding_dim"], 4)
            self.assertTrue((idx_dir / "children_dense.faiss").exists())
            self.assertTrue((idx_dir / "child_ids.json").exists())
            self.assertTrue((idx_dir / "index_manifest.json").exists())

            loaded_idx = faiss.read_index(str(idx_dir / "children_dense.faiss"))
            self.assertEqual(loaded_idx.ntotal, 3)
            loaded_ids = json.loads((idx_dir / "child_ids.json").read_text(encoding="utf-8"))
            self.assertEqual(loaded_ids, [c["child_id"] for c in children])


class _FixedQueryEncoder:
    """Mock encoder mapping query strings directly to predetermined embedding vectors."""

    def __init__(self, query_vecs: dict[str, list[float]]) -> None:
        self.query_vecs = query_vecs

    def encode_segmented_texts(self, seg_texts, batch_size=64, show_progress=False):
        arr = np.array([self.query_vecs[t] for t in seg_texts], dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.clip(norms, 1e-9, None)

    def encode_queries(self, queries, batch_size=64):
        return self.encode_segmented_texts(queries, batch_size=batch_size)

    def unload(self):
        pass


class TestRetriever(unittest.TestCase):
    def test_parent_dedup_keeps_max_score_and_child_mode(self) -> None:
        children = {
            "p1__c01": {"child_id": "p1__c01", "parent_id": "p1", "text_for_dense": "C1"},
            "p1__c02": {"child_id": "p1__c02", "parent_id": "p1", "text_for_dense": "C2"},
            "p2__c01": {"child_id": "p2__c01", "parent_id": "p2", "text_for_dense": "C3"},
            "p3__c01": {"child_id": "p3__c01", "parent_id": "p3", "text_for_dense": "C4"},
        }
        parents = {
            "p1": {"parent_id": "p1", "text_for_llm": "Parent 1 full"},
            "p2": {"parent_id": "p2", "text_for_llm": "Parent 2 full"},
            "p3": {"parent_id": "p3", "text_for_llm": "Parent 3 full"},
        }
        candidates = [("p1__c01", 0.92), ("p1__c02", 0.88), ("p2__c01", 0.90), ("p3__c01", 0.70)]

        res_p = resolve_candidates_to_contexts(
            candidates, children, parents, top_k=2, context_unit="parent", query="q"
        )
        self.assertEqual(res_p.retrieved_ids, ["p1", "p2"])
        self.assertEqual(res_p.contexts, ["Parent 1 full", "Parent 2 full"])
        self.assertAlmostEqual(res_p.chunks[0].score, 0.92)
        self.assertEqual(res_p.chunks[0].child_ids, ["p1__c01", "p1__c02"])
        self.assertAlmostEqual(res_p.top_retrieval_score, 0.92)

        res_c = resolve_candidates_to_contexts(
            candidates, children, parents, top_k=3, context_unit="child", query="q"
        )
        self.assertEqual(res_c.retrieved_ids, ["p1__c01", "p2__c01", "p1__c02"])
        self.assertEqual(res_c.contexts, ["C1", "C3", "C2"])

    def test_dense_retriever_batch_and_manifest_mismatch(self) -> None:
        enc = _FixedQueryEncoder(
            {
                "c1": [1.0, 0.0, 0.0],
                "c2": [0.9, 0.4, 0.0],
                "c3": [0.0, 1.0, 0.0],
                "q_alpha": [1.0, 0.0, 0.0],
                "q_beta": [0.0, 1.0, 0.0],
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            kb_dir = tmp / "kb"
            idx_dir = tmp / "idx"
            kb_dir.mkdir()

            parents = [
                {"parent_id": "p1", "text_for_llm": "Parent One"},
                {"parent_id": "p2", "text_for_llm": "Parent Two"},
            ]
            children = [
                {"child_id": "p1__c01", "parent_id": "p1", "text_for_dense": "Child 1", "text_for_dense_seg": "c1"},
                {"child_id": "p1__c02", "parent_id": "p1", "text_for_dense": "Child 2", "text_for_dense_seg": "c2"},
                {"child_id": "p2__c01", "parent_id": "p2", "text_for_dense": "Child 3", "text_for_dense_seg": "c3"},
            ]
            (kb_dir / "kb_parents.jsonl").write_text(
                "\n".join(json.dumps(p) for p in parents) + "\n", encoding="utf-8"
            )
            (kb_dir / "kb_children.jsonl").write_text(
                "\n".join(json.dumps(c) for c in children) + "\n", encoding="utf-8"
            )

            build_dense_index(kb_dir=str(kb_dir), index_dir=str(idx_dir), encoder=enc)

            cfg = RAGConfig(kb_dir=str(kb_dir), index_dir=str(idx_dir), top_k=2, search_k=3, context_unit="parent")
            retriever = DenseRetriever(cfg, encoder=enc)
            batch_res = retriever.retrieve_batch(["q_alpha", "q_beta"], top_k=2)
            self.assertEqual(len(batch_res), 2)
            self.assertEqual(batch_res[0].retrieved_ids, ["p1", "p2"])
            self.assertEqual(batch_res[1].retrieved_ids, ["p2", "p1"])
            self.assertGreater(batch_res[0].top_retrieval_score, 0.99)

            # Mutate kb_children.jsonl to verify manifest mismatch raises ValueError
            with (kb_dir / "kb_children.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({"child_id": "extra", "parent_id": "p2", "text_for_dense": "X", "text_for_dense_seg": "c3"}) + "\n")
            with self.assertRaises(ValueError):
                DenseRetriever(cfg, encoder=enc)


class TestPrompts(unittest.TestCase):
    def test_rag_prompt_and_list_dict_options(self) -> None:
        p_list = format_mcq_prompt(
            "Thuốc đầu tay tăng huyết áp?",
            ["Amlodipin", "Paracetamol", "Amoxicillin", "Omeprazol"],
            strategy="rag",
            context="[Nguồn: BYT] Amlodipin là thuốc chẹn kênh canxi.",
        )
        self.assertIn("Tài liệu tham khảo:", p_list)
        self.assertIn("Amlodipin là thuốc chẹn kênh canxi", p_list)
        self.assertIn("A. Amlodipin", p_list)
        self.assertIn('Định dạng bắt buộc: "Đáp án: <chữ cái>"', p_list)

        p_dict = format_mcq_prompt(
            "Câu hỏi mẫu?",
            {"A": "Lựa chọn một", "B": "Lựa chọn hai"},
            strategy="zeroshot",
        )
        self.assertIn("A. Lựa chọn một", p_dict)
        self.assertIn("B. Lựa chọn hai", p_dict)
        self.assertEqual(extract_option_pairs({"A": "1", "B": "2"}), [("A", "1"), ("B", "2")])

        p_cot = format_mcq_prompt("Câu hỏi?", ["Một", "Hai"], strategy="cot")
        self.assertIn("FINAL ANSWER:", p_cot)


class _MockRetriever:
    def __init__(self) -> None:
        self.queries_seen: list[str] = []

    def retrieve_batch(self, queries, top_k=3):
        self.queries_seen.extend(queries)
        out = []
        for idx, q in enumerate(queries):
            out.append(
                RetrievalResult(
                    query=q,
                    contexts=[f"Ngữ cảnh chuẩn cho: {q}"],
                    chunks=[],
                    retrieved_ids=[f"parent_{idx}"],
                    retrieved_child_ids=[f"parent_{idx}__c01"],
                    top_retrieval_score=0.85,
                    top_rerank_score=None,
                )
            )
        return out


class _MockLLMClient:
    def __init__(self, batch_size: int = 3, fail_batch: bool = False, fail_indices: set[int] | None = None):
        self.cfg = SimpleNamespace(batch_size=batch_size)
        self.fail_batch = fail_batch
        self.fail_indices = fail_indices or set()
        self.prompts_seen: list[str] = []
        self._responses: list[str] = []

    def set_responses(self, responses: list[str]) -> None:
        self._responses = list(responses)

    def generate_batch(self, prompts: list[str]) -> list[dict]:
        self.prompts_seen.extend(prompts)
        if self.fail_batch:
            raise RuntimeError("simulated batch OOM")
        out = []
        for i, _ in enumerate(prompts):
            txt = self._responses[i] if i < len(self._responses) else "Đáp án: A"
            out.append({"text": txt, "latency_sec": 0.1, "prompt_tokens": 20, "completion_tokens": 5})
        return out

    def generate(self, prompt: str) -> dict:
        idx = len([p for p in self.prompts_seen if p == prompt])
        pos = self.prompts_seen.index(prompt) if prompt in self.prompts_seen else 0
        if pos in self.fail_indices:
            raise RuntimeError(f"item {pos} failed")
        txt = self._responses[pos] if pos < len(self._responses) else "Đáp án: A"
        return {"text": txt, "latency_sec": 0.2, "prompt_tokens": 20, "completion_tokens": 5}


class TestRunPilot(unittest.TestCase):
    def test_rag_metadata_and_parse_fail_correct_none(self) -> None:
        items = [
            {
                "id": "q1",
                "question": "Câu 1?",
                "options": ["Đúng", "Sai", "Khác", "Không"],
                "answer": "A",
                "difficulty_level": "Easy",
                "specialty": "Nội",
            },
            {
                "id": "q2",
                "question": "Câu 2?",
                "options": ["1", "2", "3", "4"],
                "answer": "B",
                "difficulty_level": "Hard",
                "specialty": "Nội",
            },
            {
                "id": "q3",
                "question": "Câu 3?",
                "options": {"A": "X", "B": "Y", "C": "Z", "D": "W"},
                "answer": "B",
                "difficulty_level": "Medium",
                "specialty": "Ngoại",
            },
        ]
        retriever = _MockRetriever()
        client = _MockLLMClient(batch_size=3)
        # q1: valid correct ("A"); q2: out-of-range ("E" for 4-option A-D) -> parse_fail; q3: valid wrong ("C")
        client.set_responses(["Đáp án: A", "Đáp án: E", "Đáp án: C"])

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "vm14k_rag_mock.jsonl"
            run_pilot.run_vm14k(
                items, "mock", client, "rag", out_path, retriever=retriever, rag_top_k=3
            )
            lines = [json.loads(ln) for ln in out_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(lines), 3)
            self.assertIn("Tài liệu tham khảo:", client.prompts_seen[0])
            self.assertEqual(lines[0]["retrieved_ids"], ["parent_0"])
            self.assertEqual(lines[0]["retrieved_child_ids"], ["parent_0__c01"])
            self.assertAlmostEqual(lines[0]["top_retrieval_score"], 0.85)
            self.assertIsNone(lines[0]["top_rerank_score"])

            self.assertEqual(lines[0]["pred_answer"], "A")
            self.assertTrue(lines[0]["correct"])
            self.assertIsNone(lines[1]["pred_answer"])
            self.assertIsNone(lines[1]["correct"])
            self.assertEqual(lines[1]["parse_method"], "parse_fail")
            self.assertEqual(lines[2]["pred_answer"], "C")
            self.assertIs(lines[2]["correct"], False)

    def test_vimedaqa_rag_retrieval(self) -> None:
        items = [{"id": "v1", "question": "Thuốc gì?", "answer": "Đáp án chuẩn"}]
        retriever = _MockRetriever()
        client = _MockLLMClient(batch_size=2)
        client.set_responses(["Thuốc mẫu."])
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "vimedaqa_rag_mock.jsonl"
            run_pilot.run_vimedaqa(items, "mock", client, "rag", out_path, retriever=retriever)
            recs = [json.loads(ln) for ln in out_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(recs), 1)
            self.assertIn("Ngữ cảnh chuẩn cho: Thuốc gì?", client.prompts_seen[0])
            self.assertEqual(recs[0]["retrieved_ids"], ["parent_0"])

    def test_batch_error_isolation_and_legacy_error_resume(self) -> None:
        items = [
            {"id": "q1", "question": "Q1", "options": ["A1", "B1"], "answer": "A"},
            {"id": "q2", "question": "Q2", "options": ["A2", "B2"], "answer": "B"},
            {"id": "q3", "question": "Q3", "options": ["A3", "B3"], "answer": "A"},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "vm14k_zeroshot_mock.jsonl"
            # Pre-populate q1 valid and q2 legacy [ERROR]
            out_path.write_text(
                json.dumps({"id": "q1", "pred_answer": "A", "raw_output": "Đáp án: A"}) + "\n"
                + json.dumps({"id": "q2", "pred_answer": None, "raw_output": "[ERROR] timeout"}) + "\n",
                encoding="utf-8",
            )
            done = run_pilot._load_done_ids(out_path)
            self.assertEqual(done, {"q1"})

            # Now remaining = [q2, q3]; simulate batch error where index 0 (q2) fails and index 1 (q3) succeeds
            client = _MockLLMClient(batch_size=2, fail_batch=True, fail_indices={0})
            client.set_responses(["", "Đáp án: A"])
            run_pilot.run_vm14k(items, "mock", client, "zeroshot", out_path)

            main_recs = [json.loads(ln) for ln in out_path.read_text(encoding="utf-8").splitlines()]
            # Legacy q1 + legacy [ERROR] q2 (untouched) + rescued q3 (no new [ERROR] written for q2)
            self.assertEqual([r["id"] for r in main_recs], ["q1", "q2", "q3"])
            err_path = out_path.with_suffix(".errors.jsonl")
            self.assertTrue(err_path.exists())
            err_recs = [json.loads(ln) for ln in err_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(err_recs), 1)
            self.assertEqual(err_recs[0]["id"], "q2")


class TestEvaluate(unittest.TestCase):
    def test_consistent_metrics_across_legacy_and_new_files_with_dedup_and_error_filter(self) -> None:
        raw_legacy = [
            {"id": "1", "difficulty": "Easy", "gold_answer": "A", "pred_answer": "A", "correct": True, "raw_output": "Đáp án: A", "latency_sec": 1.0, "prompt_tokens": 10, "completion_tokens": 2, "model": "m1", "strategy": "cot"},
            {"id": "2", "difficulty": "Easy", "gold_answer": "B", "pred_answer": None, "correct": False, "raw_output": "không rõ", "latency_sec": 1.0, "prompt_tokens": 10, "completion_tokens": 2, "model": "m1", "strategy": "cot"},
            {"id": "3", "difficulty": "Hard", "gold_answer": "C", "pred_answer": None, "correct": False, "raw_output": "[ERROR] boom", "latency_sec": None, "prompt_tokens": None, "completion_tokens": None, "model": "m1", "strategy": "cot"},
            {"id": "3", "difficulty": "Hard", "gold_answer": "C", "pred_answer": "C", "correct": True, "raw_output": "Đáp án: C", "latency_sec": 1.0, "prompt_tokens": 10, "completion_tokens": 2, "model": "m1", "strategy": "cot"},
            {"id": "4", "difficulty": "Hard", "gold_answer": "D", "pred_answer": "A", "correct": False, "raw_output": "Đáp án: A", "latency_sec": 1.0, "prompt_tokens": 10, "completion_tokens": 2, "model": "m1", "strategy": "cot"},
        ]
        raw_new = [
            {**r, "model": "m2", "correct": (None if r["pred_answer"] is None else r["correct"])}
            for r in raw_legacy
        ]
        cleaned, n_err = clean_records(raw_legacy)
        self.assertEqual(n_err, 1)
        self.assertEqual([r["id"] for r in cleaned], ["1", "2", "3", "4"])

        with tempfile.TemporaryDirectory() as tmpdir:
            res_dir = Path(tmpdir)
            (res_dir / "vm14k_cot_m1.jsonl").write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in raw_legacy) + "\n",
                encoding="utf-8",
            )
            (res_dir / "vm14k_cot_m2.jsonl").write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in raw_new) + "\n",
                encoding="utf-8",
            )
            # Ensure .errors.jsonl is ignored
            (res_dir / "vm14k_cot_m1.errors.jsonl").write_text('{"id":"x"}\n', encoding="utf-8")

            df = evaluate_mcq(results_dir=res_dir)
            self.assertEqual(len(df), 2)
            for _, row in df.iterrows():
                self.assertEqual(row["n_questions"], 4)
                self.assertEqual(row["n_parse_fail"], 1)
                self.assertEqual(row["n_errors"], 1)
                self.assertAlmostEqual(row["parse_fail_rate"], 0.25)
                self.assertAlmostEqual(row["overall_accuracy"], 0.5)
                self.assertAlmostEqual(row["parsed_accuracy"], round(2 / 3, 4))


if __name__ == "__main__":
    unittest.main()
