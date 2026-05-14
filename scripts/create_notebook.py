#!/usr/bin/env python3
"""Generate a proper Jupyter notebook (.ipynb) for fine-tuning Gemma 4 E2B on Google Colab."""

import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Gemma 4 E2B - AI Router Model Fine-Tuning\n",
                "## Google Colab Free Tier (T4 GPU, 16GB VRAM)\n",
                "\n",
                "**Mục đích:** Fine-tune Gemma 4 E2B làm router model — hiểu intent của Sếp, trích xuất thông tin, tạo enriched prompt cho model lớn xử lý.\n",
                "\n",
                "**Framework:** Unsloth + QLoRA\n",
                "**Output:** GGUF model quantized Q4_K_M (~1.5GB) cho CPU inference"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 1: Install Dependencies\n",
                "!pip install unsloth --quiet\n",
                "!pip install \"gguf>=0.10.0\" --quiet\n",
                "!pip install scikit-learn --quiet\n",
                "print(\"Dependencies installed!\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 2: Load Gemma 4 E2B Model\n",
                "from unsloth import FastLanguageModel\n",
                "import torch\n",
                "\n",
                "model, tokenizer = FastLanguageModel.from_pretrained(\n",
                "    model_name=\"unsloth/gemma-4-e2b-unsloth-bnb-4bit\",\n",
                "    max_seq_length=2048,\n",
                "    dtype=None,\n",
                "    load_in_4bit=True,\n",
                ")\n",
                "print(\"Model loaded successfully!\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 3: Add LoRA Adapters\n",
                "model = FastLanguageModel.get_peft_model(\n",
                "    model,\n",
                "    r=16,\n",
                "    target_modules=[\"q_proj\", \"k_proj\", \"v_proj\", \"o_proj\",\n",
                "                     \"gate_proj\", \"up_proj\", \"down_proj\"],\n",
                "    lora_alpha=32,\n",
                "    lora_dropout=0.1,\n",
                "    bias=\"none\",\n",
                "    use_gradient_checkpointing=True,\n",
                "    random_state=42,\n",
                ")\n",
                "print(\"LoRA adapters configured!\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 4: Setup Chat Template\n",
                "from unsloth.chat_templates import get_chat_template\n",
                "\n",
                "tokenizer = get_chat_template(\n",
                "    tokenizer,\n",
                "    chat_template=\"gemma-3\",\n",
                ")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 5: Mount Google Drive & Load Dataset\n",
                "from google.colab import drive\n",
                "drive.mount(\"/content/drive\")\n",
                "\n",
                "import json\n",
                "import os\n",
                "\n",
                "def load_jsonl(filepath):\n",
                "    dataset = []\n",
                "    with open(filepath, \"r\", encoding=\"utf-8\") as f:\n",
                "        for line in f:\n",
                "            data = json.loads(line)\n",
                "            dataset.append({\"text\": data[\"text\"]})\n",
                "    return dataset\n",
                "\n",
                "# Look for dataset in multiple locations\n",
                "dataset_paths = [\n",
                "    \"/content/drive/MyDrive/gemma4-router/training_data.jsonl\",\n",
                "    \"/content/training_data.jsonl\",\n",
                "]\n",
                "\n",
                "dataset = None\n",
                "for path in dataset_paths:\n",
                "    if os.path.exists(path):\n",
                "        print(f\"Loading dataset from: {path}\")\n",
                "        dataset = load_jsonl(path)\n",
                "        break\n",
                "\n",
                "if dataset is None:\n",
                "    # Fallback: generate minimal synthetic data\n",
                "    print(\"No dataset found, generating minimal synthetic data...\")\n",
                "    dataset = [\n",
                "        {\"text\": f\"<|user|>\\nSếp cần viết API REST cho user management bằng FastAPI\\n<|assistant|>\\n{{\\\"task_type\\\":\\\"code_generation\\\",\\\"domain\\\":\\\"user_management\\\",\\\"framework\\\":\\\"FastAPI\\\",\\\"complexity\\\":\\\"medium\\\",\\\"enriched_prompt\\\":\\\"Bạn là senior backend developer. Tạo FastAPI RESTful API cho user management. Bao gồm: endpoints, models, error handling. Code production-ready.\\\",\\\"needs_search\\\":false,\\\"recommended_model\\\":\\\"coding-model\\\"}}<|end|>\"},\n",
                "    ] * 100\n",
                "\n",
                "print(f\"Loaded {len(dataset)} examples\")\n",
                "\n",
                "# Train/eval split\n",
                "from sklearn.model_selection import train_test_split\n",
                "train_data, eval_data = train_test_split(dataset, test_size=0.1, random_state=42)\n",
                "print(f\"Train: {len(train_data)}, Eval: {len(eval_data)}\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 6: Create Hugging Face Dataset Format\n",
                "from datasets import Dataset\n",
                "\n",
                "train_ds = Dataset.from_list(train_data)\n",
                "eval_ds = Dataset.from_list(eval_data)\n",
                "\n",
                "print(f\"Datasets created: {len(train_ds)} train, {len(eval_ds)} eval\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 7: Training Configuration\n",
                "from trl import SFTConfig\n",
                "from unsloth import is_bfloat16_supported\n",
                "\n",
                "training_args = SFTConfig(\n",
                "    per_device_train_batch_size=2,\n",
                "    per_device_eval_batch_size=2,\n",
                "    gradient_accumulation_steps=4,\n",
                "    num_train_epochs=3,\n",
                "    learning_rate=2e-4,\n",
                "    weight_decay=0.01,\n",
                "    warmup_ratio=0.1,\n",
                "    optim=\"adamw_8bit\",\n",
                "    fp16=True,\n",
                "    bf16=is_bfloat16_supported(),\n",
                "    eval_strategy=\"steps\",\n",
                "    eval_steps=100,\n",
                "    logging_steps=20,\n",
                "    save_steps=200,\n",
                "    save_total_limit=2,\n",
                "    output_dir=\"/content/gemma4-router-output\",\n",
                "    report_to=\"none\",\n",
                "    gradient_checkpointing=True,\n",
                "    max_grad_norm=0.3,\n",
                "    lr_scheduler_type=\"cosine\",\n",
                "    dataset_text_field=\"text\",\n",
                "    max_seq_length=2048,\n",
                "    dataset_num_proc=2,\n",
                "    packing=False,\n",
                ")\n",
                "print(\"Training args configured!\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 8: Initialize SFTTrainer\n",
                "from trl import SFTTrainer\n",
                "\n",
                "trainer = SFTTrainer(\n",
                "    model=model,\n",
                "    tokenizer=tokenizer,\n",
                "    train_dataset=train_ds,\n",
                "    eval_dataset=eval_ds,\n",
                "    args=training_args,\n",
                ")\n",
                "print(\"Trainer initialized!\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 9: Train the Model\n",
                "print(\"Starting training...\")\n",
                "trainer_stats = trainer.train()\n",
                "print(f\"Training complete!\")\n",
                "print(f\"Train steps: {trainer_stats.training_steps}\")\n",
                "print(f\"Final loss: {trainer_stats.training_loss:.4f}\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 10: Save LoRA Adapter\n",
                "model.save_pretrained(\"/content/gemma4-router-lora\")\n",
                "tokenizer.save_pretrained(\"/content/gemma4-router-lora\")\n",
                "print(\"LoRA adapter saved to /content/gemma4-router-lora\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 11: Export to GGUF (Q4_K_M Quantization)\n",
                "model.save_pretrained_gguf(\n",
                "    \"/content/gemma4-router-gguf\",\n",
                "    tokenizer,\n",
                "    quantization_method=\"q4_k_m\",\n",
                ")\n",
                "print(\"GGUF export complete!\")\n",
                "print(\"Model: /content/gemma4-router-gguf/gguf-model-q4_k_m.gguf\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 12: Upload to Google Drive for Daily Download\n",
                "import shutil\n",
                "from datetime import datetime\n",
                "\n",
                "timestamp = datetime.now().strftime(\"%Y%m%d_%H%M%S\")\n",
                "drive_path = f\"/content/drive/MyDrive/gemma4-router/models/{timestamp}\"\n",
                "os.makedirs(drive_path, exist_ok=True)\n",
                "\n",
                "# Copy outputs\n",
                "shutil.copytree(\"/content/gemma4-router-gguf\", f\"{drive_path}/gguf\")\n",
                "shutil.copytree(\"/content/gemma4-router-lora\", f\"{drive_path}/lora\")\n",
                "\n",
                "# Create manifest\n",
                "with open(f\"{drive_path}/manifest.json\", \"w\") as f:\n",
                "    json.dump({\n",
                "        \"timestamp\": timestamp,\n",
                "        \"model\": \"gemma-4-e2b\",\n",
                "        \"quantization\": \"q4_k_m\",\n",
                "        \"epochs\": 3,\n",
                "        \"lora_rank\": 16,\n",
                "        \"dataset_size\": len(train_data),\n",
                "        \"final_loss\": trainer_stats.training_loss,\n",
                "    }, f, indent=2)\n",
                "\n",
                "# Create latest symlink for easy access\n",
                "latest_path = \"/content/drive/MyDrive/gemma4-router/models/latest\"\n",
                "if os.path.islink(latest_path):\n",
                "    os.remove(latest_path)\n",
                "os.symlink(drive_path, latest_path)\n",
                "\n",
                "print(f\"Model uploaded to: {drive_path}\")\n",
                "print(\"Symlink latest -> this version\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 13: Quick Inference Test\n",
                "from unsloth.chat_templates import get_chat_template\n",
                "\n",
                "test_prompts = [\n",
                "    \"Sếp cần viết API REST cho user management bằng FastAPI\",\n",
                "    \"Lỗi TypeError khi chạy migration, anh fix giúp\",\n",
                "    \"Tìm hiểu về vector databases, anh cần tổng quan chi tiết\",\n",
                "]\n",
                "\n",
                "FastLanguageModel.for_inference(model)\n",
                "\n",
                "print(\"Testing inference...\\n\")\n",
                "for prompt in test_prompts:\n",
                "    messages = [{\"role\": \"user\", \"content\": prompt}]\n",
                "    inputs = tokenizer.apply_chat_template(\n",
                "        messages,\n",
                "        tokenize=True,\n",
                "        add_generation_prompt=True,\n",
                "        return_tensors=\"pt\",\n",
                "    ).to(\"cuda\")\n",
                "    \n",
                "    outputs = model.generate(\n",
                "        input_ids=inputs,\n",
                "        max_new_tokens=256,\n",
                "        temperature=0.7,\n",
                "        top_p=0.95,\n",
                "        do_sample=True,\n",
                "    )\n",
                "    \n",
                "    response = tokenizer.decode(outputs[0], skip_special_tokens=True)\n",
                "    print(f\"Input: {prompt}\")\n",
                "    print(f\"Output: {response}\")\n",
                "    print(\"-\" * 60)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Done!\n",
                "\n",
                "**Model location:** `/content/drive/MyDrive/gemma4-router/models/latest/`\n",
                "**GGUF file:** `gguf/gguf-model-q4_k_m.gguf`\n",
                "\n",
                "Download với script local hàng ngày, deploy dùng llama.cpp hoặc ollama."
            ]
        }
    ],
    "metadata": {
        "accelerator": "GPU",
        "colab": {
            "gpuType": "T4",
            "provenance": [],
            "name": "Gemma4-E2B-Router-Finetune"
        },
        "kernelspec": {
            "display_name": "Python 3",
            "name": "python3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 0
}

with open("/root/gemma4-router-finetune/notebooks/gemma4_e2b_router_finetune.ipynb", "w") as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print("Notebook created successfully!")
print(f"File: /root/gemma4-router-finetune/notebooks/gemma4_e2b_router_finetune.ipynb")
