"""
Interface thống nhất để gọi bất kỳ model nào khai báo trong configs/models.yaml.
Hỗ trợ 2 backend:
  - "openai_compatible": dùng chung cho OpenAI / OpenRouter / Together / Groq /
    vLLM-serve / Ollama (chỉ cần base_url tuân chuẩn /v1/chat/completions)
  - "local_hf": chạy model local qua transformers — PHÙ HỢP khi chạy trên
    Colab/Kaggle GPU. Hỗ trợ quantization 4-bit/8-bit (bitsandbytes) để vừa
    bộ nhớ GPU của Colab (T4 16GB / L4 24GB / A100 40GB tùy gói) và batch
    inference để tận dụng GPU thay vì gọi tuần tự từng câu.

Mỗi lần gọi trả về dict: {text, latency_sec, prompt_tokens, completion_tokens}
để evaluate.py tổng hợp chi phí/latency.
"""
import os
import gc
import time
from dataclasses import dataclass
from tenacity import retry, stop_after_attempt, wait_exponential


@dataclass
class ModelConfig:
    name: str
    backend: str
    model_id: str
    max_tokens: int = 1024
    temperature: float = 0.0
    base_url: str | None = None
    api_key_env: str | None = None
    device: str = "auto"          # "auto" để transformers tự chọn GPU/offload phù hợp
    quantization: str | None = None   # None | "4bit" | "8bit" — quan trọng trên Colab/Kaggle
    hf_token_env: str | None = "HF_TOKEN"   # cho model gated (vd: Vistral cần accept license)
    batch_size: int = 1            # >1 để chạy batch trên GPU, tăng tốc đáng kể trên Colab


class LLMClient:
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        if cfg.backend == "openai_compatible":
            from openai import OpenAI
            api_key = os.environ.get(cfg.api_key_env, "")
            if not api_key:
                raise ValueError(
                    f"Thiếu biến môi trường {cfg.api_key_env} cho model {cfg.name}"
                )
            self.client = OpenAI(api_key=api_key, base_url=cfg.base_url)
        elif cfg.backend == "local_hf":
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            hf_token = os.environ.get(cfg.hf_token_env, None) if cfg.hf_token_env else None

            quant_kwargs = {}
            if cfg.quantization in ("4bit", "8bit"):
                # Bắt buộc trên Colab free/T4 16GB cho model 7B; khuyến nghị dùng
                # ngay cả trên Colab Pro để dành VRAM cho batch_size > 1.
                from transformers import BitsAndBytesConfig
                if cfg.quantization == "4bit":
                    quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                    )
                else:
                    quant_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

            self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, token=hf_token)
            if self.tokenizer.pad_token is None:
                # Nhiều checkpoint tiếng Việt/Qwen không có pad_token riêng —
                # cần set để batch inference (padding) không lỗi.
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"  # bắt buộc cho batch generate với causal LM

            self.model = AutoModelForCausalLM.from_pretrained(
                cfg.model_id,
                torch_dtype=torch.bfloat16,
                device_map=cfg.device,
                token=hf_token,
                **quant_kwargs,
            )
            self.model.eval()
        else:
            raise ValueError(f"Backend không hỗ trợ: {cfg.backend}")

    def unload(self):
        """Giải phóng VRAM — gọi giữa các model khi chạy nhiều model local
        tuần tự trong CÙNG một phiên Colab/Kaggle, tránh OOM ở model sau."""
        if self.cfg.backend == "local_hf":
            import torch
            del self.model
            gc.collect()
            torch.cuda.empty_cache()

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, min=2, max=30))
    def _call_openai_compatible(self, prompt: str) -> dict:
        t0 = time.time()
        resp = self.client.chat.completions.create(
            model=self.cfg.model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.cfg.max_tokens,
            temperature=self.cfg.temperature,
        )
        latency = time.time() - t0
        usage = resp.usage
        return {
            "text": resp.choices[0].message.content,
            "latency_sec": latency,
            "prompt_tokens": usage.prompt_tokens if usage else None,
            "completion_tokens": usage.completion_tokens if usage else None,
        }

    def _call_local_hf(self, prompt: str) -> dict:
        return self._call_local_hf_batch([prompt])[0]

    def _call_local_hf_batch(self, prompts: list[str]) -> list[dict]:
        """Batch generate — dùng chat template đúng chuẩn của từng model
        (Qwen/Vistral đều hỗ trợ apply_chat_template) thay vì nhét prompt thô,
        vì model instruct được train với format chat, nhét thẳng text dễ ra
        kết quả kém hơn nhiều so với lúc gọi qua API (API tự lo phần này)."""
        import torch
        t0 = time.time()
        chat_prompts = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False, add_generation_prompt=True,
            )
            for p in prompts
        ]
        inputs = self.tokenizer(
            chat_prompts, return_tensors="pt", padding=True, truncation=True,
        ).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.cfg.max_tokens,
                do_sample=self.cfg.temperature > 0,
                temperature=max(self.cfg.temperature, 1e-5),
                pad_token_id=self.tokenizer.pad_token_id,
            )
        latency = time.time() - t0
        input_len = inputs["input_ids"].shape[1]
        results = []
        for i in range(len(prompts)):
            text = self.tokenizer.decode(out[i][input_len:], skip_special_tokens=True)
            n_prompt_tok = int(inputs["attention_mask"][i].sum())
            n_completion_tok = out.shape[1] - input_len
            results.append({
                "text": text,
                "latency_sec": latency / len(prompts),  # chia đều cho batch để so sánh công bằng
                "prompt_tokens": n_prompt_tok,
                "completion_tokens": n_completion_tok,
            })
        return results

    def generate(self, prompt: str) -> dict:
        if self.cfg.backend == "openai_compatible":
            return self._call_openai_compatible(prompt)
        return self._call_local_hf(prompt)

    def generate_batch(self, prompts: list[str]) -> list[dict]:
        """Dùng cho local_hf khi batch_size > 1. Với openai_compatible, vẫn
        gọi tuần tự (API không có khái niệm batch phía client)."""
        if self.cfg.backend == "local_hf" and self.cfg.batch_size > 1:
            return self._call_local_hf_batch(prompts)
        return [self.generate(p) for p in prompts]


def load_model_configs(yaml_path: str) -> list[ModelConfig]:
    import yaml
    with open(yaml_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return [ModelConfig(**{k: v for k, v in m.items() if k != "note"})
            for m in raw["models"]]
