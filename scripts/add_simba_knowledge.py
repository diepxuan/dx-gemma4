#!/usr/bin/env python3
"""
Update training_data.jsonl to make the router capable of answering Simba schema questions directly.
Also enforces: Vietnamese, no emojis, and post-action constraints.
"""
from __future__ import annotations
import json, re, random, hashlib, textwrap
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
IMPORTS = ROOT / "imports"
PROJECT_DOCS = IMPORTS / "project_docs"
OUTPUT = ROOT / "scripts" / "dataset" / "training_data.jsonl"

SYSTEM_PROMPT = (
    "Bạn là Bột, trợ lý AI cho Sếp. "
    "Quy tắc: 1. Tiếng Việt có dấu. 2. Không emoji. 3. Ngắn gọn. "
    "4. Sau khi trả lời: tự rút gọn context, kiểm tra và update skill nếu cần."
)

REQUIRED_CONSTRAINTS = [
    "Tiếng Việt có dấu, không emoji",
    "Ngắn gọn, đi thẳng vào vấn đề",
    "Sau khi hoàn thành: rút gọn context, tự update skill nếu phát hiện workflow mới",
]

SECRET_PATTERNS = [
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
]

def redact(text: str) -> str:
    text = text or ""
    for pat in SECRET_PATTERNS:
        text = pat.sub("[REDACTED]", text)
    return text.strip()

def parse_table_md(path: Path) -> dict[str, Any] | None:
    """Parse a table markdown file to extract name, description, columns."""
    try:
        content = path.read_text()
        # Heuristic: First line or header often has the name
        lines = content.splitlines()
        name = path.stem
        desc = ""
        cols = []
        in_table = False
        for line in lines:
            line = line.strip()
            if not desc and line and not line.startswith("#") and not line.startswith("|"):
                desc = line[:100]
            if line.startswith("|") and ("Field" in line or "Ten" in line or "Column" in line or "Truong" in line):
                in_table = True
                continue
            if in_table and line.startswith("|"):
                parts = [p.strip() for p in line.split("|") if p.strip()]
                if len(parts) >= 2 and parts[0].isidentifier(): # Heuristic for field name
                    cols.append({"field": parts[0], "type": parts[1], "note": parts[2] if len(parts) > 2 else ""})
        
        if not cols:
            # Try to find any table structure
            for line in lines:
                if re.match(r'\|.*\|', line):
                     parts = [p.strip() for p in line.split("|") if p.strip()]
                     if len(parts) >= 2:
                         cols.append({"field": parts[0], "type": parts[1]})

        return {"name": name, "desc": desc, "cols": cols[:20]} # Limit cols for context
    except:
        return None

def parse_proc_md(path: Path) -> dict[str, Any] | None:
    """Parse a procedure markdown file."""
    try:
        content = path.read_text()
        name = path.stem
        # Look for "Input", "Output" sections
        inputs = []
        outputs = []
        # Simple heuristic extraction
        for line in content.splitlines():
            if "Input" in line or "Tham số" in line:
                inputs.append(line.strip()[:100])
            if "Output" in line or "Kết quả" in line:
                outputs.append(line.strip()[:100])
        return {"name": name, "inputs": inputs[:5], "outputs": outputs[:5]}
    except:
        return None

def make_schema_example(user_q: str, answer_text: str) -> dict:
    assistant_out = {
        "task_type": "simba_schema",
        "domain": "SimbaERP",
        "framework": "SQL/ERP",
        "complexity": "hard",
        "intent": "answer_schema_from_memory",
        "needs_search": False,
        "needs_tools": False,
        "recommended_model": "assistant-model",
        "constraints": REQUIRED_CONSTRAINTS + ["Trả lời trực tiếp từ bộ nhớ, không cần tìm tài liệu"],
        "enriched_prompt": answer_text
    }
    text = f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n{redact(user_q)}\n<|assistant|>\n{json.dumps(assistant_out, ensure_ascii=False, separators=(',',':'))}<|end|>"
    return {"text": text}

def main():
    examples = []
    random.seed(42)

    # 1. Add manual examples for behavior rules
    behavior_examples = [
        ("báo cáo tiến độ", "Báo cáo: Đã hoàn thành task A, B. Đang chờ task C."),
        ("xóa branch cũ", "Đã xóa các branch đã merge. List branch hiện tại: main, dev."),
    ]
    for q, a in behavior_examples:
        out = {
            "task_type": "github_workflow", "domain": "repo_management", "framework": "Git",
            "complexity": "medium", "intent": "report_or_action", "needs_search": False,
            "needs_tools": True, "recommended_model": "coding-model",
            "constraints": REQUIRED_CONSTRAINTS,
            "enriched_prompt": a
        }
        text = f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n{q}\n<|assistant|>\n{json.dumps(out, ensure_ascii=False, separators=(',',':'))}<|end|>"
        examples.append({"text": text})

    # 2. Add Simba Table Schema Examples (Memorization)
    table_dir = PROJECT_DOCS / "simba" / "tables"
    if table_dir.exists():
        for md in sorted(table_dir.glob("*.md")):
            info = parse_table_md(md)
            if info and info.get("cols"):
                col_text = ", ".join([f"{c['field']} ({c.get('type', '?')})" for c in info['cols'][:10]])
                answer = f"Bảng {info['name']}: {info['desc']}. " \
                         f"Cấu trúc gồm các cột chính: {col_text}. " \
                         f"(Danh sách đầy đủ có trong hệ thống)."
                
                # Create multiple question variations for the same table
                variations = [
                    f"cấu trúc bảng {info['name']} là gì",
                    f"bảng {info['name']} có những cột nào",
                    f"index và key của bảng {info['name']}",
                    f"ý nghĩa các trường trong bảng {info['name']}",
                ]
                for q in variations:
                    examples.append(make_schema_example(q, answer))

    # 3. Add Simba Procedure Examples
    proc_dir = PROJECT_DOCS / "simba" / "procedures"
    # Note: Procs are often in subdirs like AP, AR...
    if proc_dir.exists():
        for md in proc_dir.rglob("*.md"):
            info = parse_proc_md(md)
            if info:
                answer = f"Procedure {info['name']}. "
                if info.get('inputs'): answer += f"Input: {', '.join(info['inputs'][:3])}. "
                if info.get('outputs'): answer += f"Output: {', '.join(info['outputs'][:3])}."
                
                q = f"proc {info['name']} nhận tham số gì và trả về cái gì"
                # For detailed logic, we set needs_search=True
                logic_q = f"logic bên trong procedure {info['name']} chạy như thế nào"
                
                # Answer schema
                examples.append(make_schema_example(q, answer))
                
                # Logic question (needs search)
                logic_out = {
                    "task_type": "simba_logic", "domain": "SimbaERP", "framework": "SQL",
                    "complexity": "hard", "intent": "explain_proc_logic", 
                    "needs_search": True, "needs_tools": True, "recommended_model": "research-model",
                    "constraints": REQUIRED_CONSTRAINTS + ["Cần đọc file procedure để giải thích chi tiết logic"],
                    "enriched_prompt": f"Để giải thích logic chi tiết của proc {info['name']}, cần tìm và đọc file tài liệu tương ứng trong dự án."
                }
                logic_text = f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n{logic_q}\n<|assistant|>\n{json.dumps(logic_out, ensure_ascii=False, separators=(',',':'))}<|end|>"
                examples.append({"text": logic_text})

    # 4. Deduplicate and shuffle
    seen = set()
    clean = []
    for ex in examples:
        h = hashlib.sha256(ex['text'].encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            clean.append(ex)
    random.shuffle(clean)

    # 5. Write
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as f:
        for ex in clean:
            f.write(json.dumps(ex, ensure_ascii=False, separators=(",", ":")) + "\n")

    print(f"Generated {len(clean)} examples.")
    sample_text = clean[0]['text'].split('<|assistant|>\n')[1].replace('<|end|>','')
    sample_json = json.loads(sample_text)
    print(f"Sample 1:\n{json.dumps(sample_json, indent=2, ensure_ascii=False)[:500]}")

if __name__ == "__main__":
    main()
