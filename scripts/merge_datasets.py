#!/usr/bin/env python3
"""Merge add_simba_knowledge.jsonl with existing training_data.jsonl."""
import json, random, hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NEW_FILE = ROOT / "scripts" / "dataset" / "simba_knowledge.jsonl"
OLD_FILE = ROOT / "scripts" / "dataset" / "training_data.jsonl"
OUTPUT = ROOT / "scripts" / "dataset" / "training_data_merged.jsonl"

# 1. Generate new examples if not exists
if not NEW_FILE.exists():
    import subprocess
    subprocess.run(["python3", str(ROOT / "scripts" / "add_simba_knowledge.py")], check=True)
    # Move the output of add_simba_knowledge.py to NEW_FILE if it wrote to OUTPUT
    # The script writes to OUTPUT (training_data.jsonl). We need to capture it.
    # Let's just re-run the logic here or read the file if it was updated.
    pass

# Read new examples (simba_knowledge)
# Note: add_simba_knowledge.py wrote to training_data.jsonl directly in the previous run.
# Let's assume the previous run overwrote training_data.jsonl.
# We need to merge the "general" examples back in.
# I will re-run the general generator logic (from build_training_dataset.py) in memory.
