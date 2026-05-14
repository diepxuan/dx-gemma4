#!/usr/bin/env python3
"""
Pipeline orchestrator for daily Gemma 4 E2B router model fine-tuning.

Architecture:
    Dataset ──push──> GitHub (version control)
    Colab pull dataset from GitHub ──train──> model
    Model ──upload──> Hugging Face Hub (diepxuan/dx-gemma4)
    Local pull model from HF Hub

Steps:
1. Generate fresh synthetic dataset
2. Commit + push dataset to GitHub
3. Trigger Colab notebook (manual or via Kaggle)
4. Wait for Colab to complete
5. Upload trained model from Colab to HF Hub (diepxuan/dx-gemma4)
6. Download model from HF Hub to local
7. Notify completion

Usage:
    python pipeline.py --step generate    # Step 1: generate dataset
    python pipeline.py --step push        # Step 2: commit + push to GitHub
    python pipeline.py --step download    # Step 5: download latest model from HF Hub
    python pipeline.py --step all         # Full pipeline (interactive)
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ============================================================
# Configuration
# ============================================================
BASE_DIR = Path(os.path.expanduser("~/gemma4-router-finetune"))
DATASET_DIR = BASE_DIR / "dataset"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
SCRIPTS_DIR = BASE_DIR / "scripts"

# Hugging Face Hub repo for model storage
HF_REPO_ID = "diepxuan/dx-gemma4"

# GitHub repo for dataset/code version control (set by user)
GITHUB_REPO_URL = os.getenv(
    "GEMMA4_GITHUB_URL",
    "git@github.com:diepxuan/dx-gemma4.git"
)

# Ensure directories exist
for d in [DATASET_DIR, MODELS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ============================================================
# Helpers
# ============================================================
def run(cmd, capture=True):
    """Run shell command, return result."""
    if isinstance(cmd, str):
        cmd = cmd.split()
    kw = dict(capture_output=True, text=True) if capture else {}
    return subprocess.run(cmd, **kw)


def get_hf_token():
    """Get HF token from env or cached file."""
    token = os.getenv("HF_TOKEN", "")
    if token:
        return token
    token_path = Path(os.path.expanduser("~/.cache/huggingface/token"))
    if token_path.exists():
        return token_path.read_text().strip()
    return ""


# ============================================================
# Step 1: Generate Dataset
# ============================================================
def generate_dataset(count=2000):
    """Generate fresh synthetic dataset."""
    print("[1/5] Generating synthetic dataset...")

    script_path = SCRIPTS_DIR / "generate_dataset.py"
    output_path = DATASET_DIR / "training_data.jsonl"

    cmd = [
        sys.executable, str(script_path),
        "--output", str(output_path),
        "--count", str(count)
    ]

    result = run(cmd)
    if result.returncode != 0:
        print(f"ERROR: Dataset generation failed")
        print(result.stderr)
        return False

    if not output_path.exists():
        print(f"ERROR: Dataset file not created at {output_path}")
        return False

    with open(output_path) as f:
        line_count = sum(1 for _ in f)

    file_size = output_path.stat().st_size / 1024
    print(f"  Generated {line_count} examples ({file_size:.0f} KB)")
    print(f"  Saved to: {output_path}")
    return True


# ============================================================
# Step 2: Push Dataset to GitHub
# ============================================================
def push_dataset_to_github(message=None):
    """Commit and push dataset to GitHub."""
    print("[2/5] Pushing dataset to GitHub...")

    if not shutil.which("git"):
        print("  ERROR: git not found")
        return False

    # Init repo if needed
    if not (BASE_DIR / ".git").exists():
        print("  Initializing git repo...")
        r = run(["git", "init", str(BASE_DIR)])
        if r.returncode != 0:
            print(f"  git init failed: {r.stderr}")
            return False

        # .gitignore for large files
        gitignore = BASE_DIR / ".gitignore"
        if not gitignore.exists():
            gitignore.write_text(
                "# Models are stored on HF Hub, not in git\n"
                "models/\n"
                "*.gguf\n"
                "*.bin\n"
                "*.pt\n"
                "__pycache__/\n"
                ".hf-venv/\n"
                "logs/\n"
            )

        # Initial commit
        run(["git", "add", str(BASE_DIR / ".gitignore")])
        run(["git", "-C", str(BASE_DIR), "add", "pipeline.py", "scripts/", "notebooks/", "dataset/"])
        run(["git", "-C", str(BASE_DIR), "commit", "-m", "init: pipeline, dataset, notebook"])

    # Add remote if needed
    r = run(["git", "-C", str(BASE_DIR), "remote", "get-url", "origin"])
    if r.returncode != 0:
        print(f"  Adding remote: {GITHUB_REPO_URL}")
        run(["git", "-C", str(BASE_DIR), "remote", "add", "origin", GITHUB_REPO_URL])

    # Commit dataset changes
    dataset_path = DATASET_DIR / "training_data.jsonl"
    run(["git", "-C", str(BASE_DIR), "add", "dataset/training_data.jsonl"])
    r = run(["git", "-C", str(BASE_DIR), "diff", "--cached", "--quiet"])
    if r.returncode == 0:
        print("  No dataset changes to commit")
        return True

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    msg = message or f"update dataset {ts}"
    run(["git", "-C", str(BASE_DIR), "commit", "-m", msg])

    # Push
    print(f"  Pushing to: {GITHUB_REPO_URL}")
    r = run(["git", "-C", str(BASE_DIR), "push", "origin", "main"])
    if r.returncode != 0:
        print(f"  Push failed: {r.stderr}")
        print("  Hint: setup SSH key or GitHub token for auth")
        return False

    print("  Push successful")
    return True


# ============================================================
# Step 3: Trigger Colab Fine-Tuning
# ============================================================
def trigger_colab():
    """Trigger Colab notebook execution."""
    print("[3/5] Colab fine-tuning trigger...")

    if shutil.which("kaggle"):
        print("  Kaggle CLI found!")
        print("  Can trigger kernel run:")
        print("  kaggle kernels push -p /path/to/notebook")
        return True

    print("\n  MANUAL TRIGGER REQUIRED:")
    print("  1. Open notebook in Colab:")
    print("     Pull dataset from GitHub in Colab:")
    print("     !git clone https://github.com/diepxuan/dx-gemma4.git /content/dataset")
    print("  2. Connect to T4 GPU")
    print("  3. Runtime -> Run all")
    print("  4. Wait ~30-45 minutes for training")
    print("  5. Model auto-uploads to HF Hub: diepxuan/dx-gemma4")
    print("     (Colab notebook must include HF upload step)")
    return False


# ============================================================
# Step 4: Upload Model to HF Hub (from Colab)
# ============================================================
def upload_model_to_hf(local_model_path, commit_message=None):
    """Upload model file to Hugging Face Hub.

    This step is meant to run on Colab after training.
    Local pipeline calls download_from_hf instead.
    """
    print("[4/5] Uploading model to Hugging Face Hub...")

    token = get_hf_token()
    if not token:
        print("  ERROR: HF_TOKEN not set")
        return False

    try:
        from huggingface_hub import HfApi
        api = HfApi()

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        msg = commit_message or f"model update {ts}"

        api.upload_file(
            path_or_fileobj=str(local_model_path),
            path_in_repo=f"models/{Path(local_model_path).name}",
            repo_id=HF_REPO_ID,
            repo_type="model",
            commit_message=msg,
        )
        print(f"  Uploaded: {local_model_path}")
        print(f"  Repo: https://huggingface.co/{HF_REPO_ID}")
        return True
    except ImportError:
        print("  ERROR: huggingface_hub not installed")
        print("  pip install huggingface_hub")
        return False
    except Exception as e:
        print(f"  Upload failed: {e}")
        return False


# ============================================================
# Step 5: Download Model from HF Hub
# ============================================================
def download_model_from_hf():
    """Download latest model from Hugging Face Hub."""
    print("[4/5] Downloading model from Hugging Face Hub...")

    token = get_hf_token()
    if not token:
        print("  ERROR: HF_TOKEN not set")
        return False

    try:
        from huggingface_hub import HfApi, hf_hub_download
        api = HfApi()

        # List files in repo
        files = api.list_repo_files(HF_REPO_ID, repo_type="model")
        model_files = [f for f in files if f.endswith(".gguf") or f.endswith(".bin")]

        if not model_files:
            print("  No model files found in repo")
            print(f"  Check: https://huggingface.co/{HF_REPO_ID}/tree/main")
            return False

        # Get latest model file (by name sort, assumes timestamp naming)
        latest = sorted(model_files)[-1]
        print(f"  Found: {latest}")

        # Archive old model
        local_latest = MODELS_DIR / "latest"
        if local_latest.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive = MODELS_DIR / f"backup_{timestamp}"
            local_latest.rename(archive)
            print(f"  Archived old model to: {archive}")

        local_latest.mkdir(parents=True, exist_ok=True)

        # Download
        downloaded = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=latest,
            local_dir=str(MODELS_DIR / "latest"),
            repo_type="model",
            token=token,
        )
        print(f"  Downloaded: {downloaded}")
        print(f"  Location: {MODELS_DIR / 'latest'}")

        # Save manifest
        manifest = {
            "timestamp": datetime.now().isoformat(),
            "source": HF_REPO_ID,
            "filename": latest,
            "local_path": str(downloaded),
        }
        (local_latest / "manifest.json").write_text(json.dumps(manifest, indent=2))

        return True

    except ImportError:
        print("  ERROR: huggingface_hub not installed")
        print("  pip install huggingface_hub")
        return False
    except Exception as e:
        print(f"  Download failed: {e}")
        return False


# ============================================================
# Get Latest Model Info
# ============================================================
def get_latest_model_info():
    """Get info about latest trained model."""
    print("Checking for latest trained model...")

    manifest_path = MODELS_DIR / "latest" / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        print(f"  Found local model from: {manifest.get('timestamp', 'unknown')}")
        print(f"  Source: {manifest.get('source', 'unknown')}")
        return manifest

    print("  No local model found")
    return None


# ============================================================
# Main Pipeline
# ============================================================
def run_pipeline():
    """Run complete pipeline."""
    print("=" * 60)
    print("Gemma 4 E2B Router Model - Daily Fine-Tuning Pipeline")
    print("=" * 60)
    print(f"Model repo: https://huggingface.co/{HF_REPO_ID}")
    print(f"Code repo:  {GITHUB_REPO_URL}")
    print(f"Timestamp:  {datetime.now().isoformat()}")
    print()

    # Step 1: Generate
    if not generate_dataset(count=2000):
        print("Pipeline FAILED at dataset generation")
        return False

    # Step 2: Push to GitHub
    if not push_dataset_to_github():
        print("Pipeline FAILED at GitHub push")
        return False

    # Step 3: Trigger Colab
    trigger_colab()

    print("\n" + "=" * 60)
    print("Pipeline PAUSED - Waiting for Colab to complete")
    print("=" * 60)
    print("\nColab sẽ:")
    print("  1. Pull dataset từ GitHub")
    print("  2. Fine-tune model trên T4 GPU")
    print("  3. Upload model lên HF Hub: diepxuan/dx-gemma4")
    print("\nSau khi Colab xong, chạy lệnh sau để download model:")
    print(f"  python {sys.argv[0]} --step download")

    return True


def download_model():
    """Download latest model from HF Hub."""
    if download_model_from_hf():
        print("\nModel downloaded successfully!")
        get_latest_model_info()
    else:
        print("\nDownload failed.")


# ============================================================
# CLI Interface
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gemma 4 Router Model Pipeline")
    parser.add_argument(
        "--step",
        choices=["generate", "push", "colab", "download", "all"],
        default="all",
        help="Pipeline step to run"
    )
    parser.add_argument("--count", type=int, default=2000, help="Dataset size")

    args = parser.parse_args()

    if args.step == "generate":
        generate_dataset(args.count)
    elif args.step == "push":
        push_dataset_to_github()
    elif args.step == "colab":
        trigger_colab()
    elif args.step == "download":
        download_model()
    elif args.step == "all":
        run_pipeline()
