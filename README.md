# Gemma 4 E2B Router Model - Daily Fine-Tuning Pipeline

## Kiến Trúc

```
Daily Pipeline
├── Generate dataset (local)
├── Push dataset → GitHub (version control)
├── Colab pull dataset từ GitHub
├── Fine-tune trên T4 GPU (30-45 min)
├── Upload model → Hugging Face Hub (diepxuan/dx-gemma4)
└── Local download model từ HF Hub
```

## Setup

### 1. Hugging Face Hub (model storage)

Đã setup sẵn token `dx-gemma4` trên server.
Model repo: https://huggingface.co/diepxuan/dx-gemma4

Trên Colab, cần thêm bước upload model sau khi train xong:

```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_file(
    path_or_fileobj="/content/output/gguf-model-q4_k_m.gguf",
    path_in_repo="models/gguf-model-q4_k_m.gguf",
    repo_id="diepxuan/dx-gemma4",
    repo_type="model",
    commit_message=f"model {datetime.now().strftime('%Y%m%d')}"
)
```

### 2. GitHub (code + dataset)

```bash
# Setup SSH key hoặc GitHub token
ssh-keygen -t ed25519 -C "hermes-agent"
# Add public key vào GitHub Settings → SSH keys

# Init repo (pipeline tự động làm)
cd ~/gemma4-router-finetune
git init
git add .
git commit -m "init"
git remote add origin git@github.com:diepxuan/dx-gemma4.git
git push -u origin main
```

Colab pull dataset:
```python
!git clone https://github.com/diepxuan/dx-gemma4.git /content/dataset-repo
```

Kaggle pull dataset:
```python
!git clone https://github.com/diepxuan/dx-gemma4.git /kaggle/working/dataset-repo
```

### 3. Cài dependencies

```bash
# Trên local server
cd ~/gemma4-router-finetune
pip install huggingface_hub  # đã có trong .hf-venv

# Trên Colab notebook
!pip install huggingface_hub transformers accelerate bitsandbytes
```

## Usage

### Fine-tune trên Kaggle khi Colab hết quota

Dùng notebook:

```text
notebooks/gemma4_e2b_router_finetune_kaggle.ipynb
```

Thiết lập trên Kaggle:

```text
1. Settings -> Accelerator -> GPU
2. Settings -> Internet -> On
3. Add-ons / Secrets -> tạo secret HF_TOKEN
4. Run cells 1-12 theo thứ tự
```

Notebook Kaggle dùng `/kaggle/working`, `kaggle_secrets.UserSecretsClient`, clone dataset từ GitHub bằng HTTPS, rồi upload GGUF/LoRA/manifest lên Hugging Face Hub.

Nếu thiếu VRAM, giảm Cell 7:

```python
per_device_train_batch_size=1
gradient_accumulation_steps=8
```

```bash
# Generate dataset mới
python3 ~/gemma4-router-finetune/pipeline.py --step generate --count 2000

# Push dataset lên GitHub
python3 ~/gemma4-router-finetune/pipeline.py --step push

# Download model mới từ HF Hub
python3 ~/gemma4-router-finetune/pipeline.py --step download

# Full pipeline (dừng ở bước 3 chờ Colab)
python3 ~/gemma4-router-finetune/pipeline.py --step all
```

## Cấu Trúc Project

```
~/gemma4-router-finetune/
├── pipeline.py                      # Main orchestrator
├── README.md                        # Tài liệu dự án
├── .gitignore                       # Bỏ qua models/, logs/
├── dataset/
│   └── training_data.jsonl          # Generated dataset (push lên GitHub)
├── notebooks/
│   ├── gemma4_e2b_router_finetune.ipynb         # Colab notebook
│   └── gemma4_e2b_router_finetune_kaggle.ipynb  # Kaggle notebook fallback
├── scripts/
│   ├── generate_dataset.py          # Dataset generator
│   └── create_notebook.py           # Notebook generator
├── models/                          # Downloaded từ HF Hub
│   └── latest/                      # Model hiện tại
│       └── manifest.json
└── logs/                            # Pipeline logs
```

## Model Specs

- **Base Model:** Gemma 4 E2B (2B params)
- **Fine-tuning:** QLoRA rank 16, 3 epochs
- **Quantization:** Q4_K_M (~1.5GB GGUF file)
- **Inference:** llama.cpp hoặc Ollama trên CPU
- **VRAM Needed:** 16GB (Colab free T4)
- **Training Time:** 30-45 phút trên T4
- **Dataset Size:** 2000 examples (~1MB)

## Storage

| Thành phần | Nơi lưu | Công cụ |
|-----------|---------|---------|
| Code, scripts | GitHub | git |
| Dataset | GitHub | git (file nhỏ ~1MB) |
| Model GGUF (~1.5GB) | Hugging Face Hub | huggingface_hub CLI |
| Logs | Local server | filesystem |

## Environment Variables

| Biến | Giá trị | Mục đích |
|------|---------|----------|
| `HF_TOKEN` | Đã set trong `.env` | Upload/download model từ HF Hub |
| `GEMMA4_GITHUB_URL` | Default: git@github.com:diepxuan/dx-gemma4 | URL repo GitHub cho dataset |
