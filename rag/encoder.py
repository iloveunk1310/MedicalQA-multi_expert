"""ViHealthBERT dense encoder with attention-masked mean/CLS pooling and L2 normalization."""

import gc
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from chunking.tokenize_vi import segment_vi


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


class ViHealthBertEncoder:
    """Dense encoder wrapping demdecuong/vihealthbert-base-word (or compatible HF encoder)."""

    def __init__(
        self,
        model_name: str = "demdecuong/vihealthbert-base-word",
        pooling: str = "mean",
        max_length: int = 256,
        device: str = "auto",
        model: Any = None,
        tokenizer: Any = None,
    ) -> None:
        if pooling not in ("mean", "cls"):
            raise ValueError(f"Invalid pooling={pooling!r}; expected 'mean' or 'cls'.")
        self.model_name = model_name
        self.pooling = pooling
        self.max_length = max_length
        self.device = _resolve_device(device)

        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if model is not None:
            self.model = model
        else:
            from transformers import AutoModel

            self.model = AutoModel.from_pretrained(model_name)

        if hasattr(self.model, "to"):
            self.model.to(self.device)
        if hasattr(self.model, "eval"):
            self.model.eval()

    def preprocess_query(self, query: str) -> str:
        """Normalize Unicode NFC and segment Vietnamese words to match KB indexing format."""
        seg = segment_vi(query or "")
        return " ".join(seg.split())

    def encode_segmented_texts(
        self,
        seg_texts: Sequence[str],
        batch_size: int = 64,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode already word-segmented texts into L2-normalized float32 embeddings."""
        if not seg_texts:
            return np.zeros((0, 0), dtype=np.float32)

        batches = [
            list(seg_texts[i : i + batch_size])
            for i in range(0, len(seg_texts), batch_size)
        ]
        iterator = tqdm(batches, desc="Encoding dense chunks") if show_progress else batches

        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for batch in iterator:
                encoded = self.tokenizer(
                    batch,
                    max_length=self.max_length,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                )
                model_inputs = {
                    k: v.to(self.device) if hasattr(v, "to") else v
                    for k, v in encoded.items()
                }
                model_out = self.model(**model_inputs)
                hidden = (
                    model_out.last_hidden_state
                    if hasattr(model_out, "last_hidden_state")
                    else model_out[0]
                )

                if self.pooling == "mean":
                    attn_mask = model_inputs["attention_mask"]
                    mask = attn_mask.unsqueeze(-1).expand(hidden.size()).float()
                    summed = torch.sum(hidden.float() * mask, dim=1)
                    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
                    emb = summed / counts
                else:
                    emb = hidden[:, 0, :].float()

                emb = F.normalize(emb, p=2, dim=1)
                outputs.append(emb.detach().cpu().numpy().astype(np.float32))

        return np.ascontiguousarray(np.vstack(outputs), dtype=np.float32)

    def encode_queries(
        self,
        queries: Sequence[str],
        batch_size: int = 64,
    ) -> np.ndarray:
        """Preprocess raw Vietnamese queries and encode into L2-normalized float32 vectors."""
        seg_queries = [self.preprocess_query(q) for q in queries]
        return self.encode_segmented_texts(seg_queries, batch_size=batch_size, show_progress=False)

    def unload(self) -> None:
        """Release model weights and clear CUDA cache if available."""
        if getattr(self, "model", None) is not None:
            del self.model
            self.model = None
        if getattr(self, "tokenizer", None) is not None:
            del self.tokenizer
            self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
