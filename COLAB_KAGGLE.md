# Chạy pipeline trên Google Colab (Pro) / Kaggle

## Những gì đã được chuẩn bị sẵn trong code cho 2 môi trường này

- `device="auto"` trong `model_wrappers.py` — transformers tự chọn GPU/offload,
  không cần biết trước Colab cấp T4/L4/A100.
- `quantization: 4bit` (bitsandbytes NF4) — bắt buộc nếu rơi vào T4 (16GB VRAM);
  nếu Colab Pro cấp A100/L4 (VRAM lớn), có thể bỏ để chạy bf16 đầy đủ, chất
  lượng cao hơn một chút.
- `batch_size` trong config — tận dụng GPU thay vì chạy tuần tự từng câu.
  Với T4: batch_size=4 an toàn. Với A100/L4 (Colab Pro): thử 8-16.
- **Resume tự động**: `run_pilot.py` đọc file kết quả cũ (nếu có), bỏ qua các
  câu đã chạy, chỉ chạy tiếp phần còn thiếu — quan trọng vì Colab free/Pro đều
  có giới hạn thời gian phiên và có thể ngắt giữa chừng.
- Flag `--model_name` — chạy từng model một lần runtime thay vì load nhiều
  model 7B cùng lúc trong 1 phiên (tránh OOM, và nếu phiên ngắt giữa chừng chỉ
  mất tiến độ của 1 model, không phải toàn bộ).

## Checklist trước khi chạy

1. **Bật GPU**: Colab → Runtime → Change runtime type → GPU (A100/L4 nếu Colab
   Pro cho chọn). Kaggle → Settings → Accelerator → GPU, và **Settings → Internet: On**
   (bắt buộc để `pip install` và tải model từ HuggingFace).
2. **Mount Google Drive** (Colab) hoặc dùng `/kaggle/working/` (Kaggle) để
   `results/` không mất khi hết phiên:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   # trỏ RESULTS_DIR trong run_pilot.py / evaluate.py ra thư mục trong Drive,
   # hoặc đơn giản hơn: chạy toàn bộ project trong /content/drive/MyDrive/...
   ```
3. **HF token** (nếu model gated, ví dụ cần accept license trước khi tải):
   ```python
   import os
   os.environ["HF_TOKEN"] = "hf_xxx..."
   ```
4. **API key** cho model qua API (OpenAI/OpenRouter):
   ```python
   import os
   os.environ["OPENAI_API_KEY"] = "sk-..."
   os.environ["OPENROUTER_API_KEY"] = "sk-or-..."
   ```

## Lệnh chạy mẫu (1 cell Colab/Kaggle)

```bash
!pip install -q -r requirements.txt

# Chạy model local (vd: Vistral) trên GPU Colab — 1 model/lần
!python run_pilot.py --dataset vm14k --strategy zeroshot \
    --sample_per_level 100 --model_name vistral-7b-chat

!python run_pilot.py --dataset vm14k --strategy cot \
    --sample_per_level 100 --model_name vistral-7b-chat

# Model qua API không cần GPU — có thể chạy cùng lúc, không cần --model_name
!python run_pilot.py --dataset vm14k --strategy zeroshot --sample_per_level 100

# Sau khi đủ các lần chạy cần thiết
!python evaluate.py
```

## Lưu ý thêm

- **Colab Pro vs free**: khác biệt chính là loại GPU cấp (A100/L4 thay vì T4)
  và thời gian phiên dài hơn — code không cần biết trước đang chạy trên loại
  nào nhờ `device="auto"`, chỉ cần bạn chỉnh `quantization`/`batch_size` trong
  YAML cho phù hợp sau khi biết GPU thực tế được cấp (`!nvidia-smi` để kiểm tra).
- **Kaggie giới hạn GPU quota** (thường ~30h/tuần) — nên ưu tiên chạy các model
  local trước, để dành model API (không tốn quota GPU) chạy sau hoặc chạy ở
  môi trường khác (máy cá nhân/Colab) nếu gần hết quota.
- Nếu Colab ngắt kết nối giữa chừng: chỉ cần chạy lại ĐÚNG lệnh cũ — cơ chế
  resume sẽ tự bỏ qua phần đã xong.
- Chạy `evaluate.py` sau khi tải `results/` về máy cá nhân cũng được — không
  bắt buộc chạy trên Colab/Kaggle vì bước này không cần GPU.
