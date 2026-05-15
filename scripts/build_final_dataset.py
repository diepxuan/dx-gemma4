#!/usr/bin/env python3
"""Build comprehensive training_data.jsonl with Simba schema memorization and strict rules."""
import json, re, random, hashlib, argparse
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "imports" / "project_docs" / "simba"
OUTPUT = ROOT / "scripts" / "dataset" / "training_data.jsonl"

SYSTEM_PROMPT = (
    "Bạn là Bột, trợ lý AI cho Sếp. "
    "Quy tắc bắt buộc: 1. Tiếng Việt có dấu. 2. Không emoji. 3. Ngắn gọn. "
    "4. Sau khi hoàn thành task: tự rút gọn context, kiểm tra và update skill nếu cần."
)

CONSTRAINTS = [
    "Tiếng Việt có dấu, không emoji",
    "Ngắn gọn, đi thẳng vào vấn đề",
    "Sau khi hoàn thành: rút gọn context, tự update skill nếu phát hiện workflow mới",
]

def clean(text):
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def parse_table(path):
    try:
        txt = path.read_text()
        name = path.stem
        desc = ""
        # Try to get description from blockquote or first line
        m = re.search(r'>\s*(.+)', txt)
        if m: desc = m.group(1).strip()
        
        cols = []
        # Parse the structure table
        for line in txt.splitlines():
            if re.match(r'\|\s*\d+\s*\|', line):
                parts = [p.strip() for p in line.split('|') if p.strip()]
                if len(parts) >= 3:
                    # parts: ['1', 'ma_cty', 'NVARCHAR...', 'NO', '', 'Mã công ty']
                    cols.append(f"{parts[1]} ({parts[2]})")
        
        if cols:
            return {"name": name, "desc": desc, "cols": cols}
    except: pass
    return None

def parse_proc(path):
    try:
        txt = path.read_text()
        name = path.stem
        # Find Input/Output sections
        inputs = []
        outputs = []
        
        # Heuristic: lines under headers or containing "Input"/"Tham số"
        # Or looking for list items in those sections
        # Simple approach: just find lines with "Input" and subsequent list items
        
        for line in txt.splitlines():
            line = line.strip()
            if line.lower().startswith("input") or line.lower().startswith("tham số"):
                # Grab next few lines if they look like list items
                continue 
            # Better: find the table of params if exists
            if re.match(r'\|\s*\d+\s*\|', line):
                 parts = [p.strip() for p in line.split('|') if p.strip()]
                 if len(parts) >= 3 and ("IN" in line or "OUT" in line):
                     inputs.append(f"{parts[1]} ({parts[2]})")
        
        # Fallback: just use name
        return {"name": name, "summary": f"Procedure {name}."}
    except: pass
    return None

def make_ex(q, ans, task_type="simba_schema", needs_search=False, enriched_prompt=None):
    meta = {
        "task_type": task_type,
        "domain": "SimbaERP",
        "framework": "SQL/ERP",
        "complexity": "hard",
        "intent": "answer_from_memory",
        "needs_search": needs_search,
        "needs_tools": not needs_search, # If from memory, usually no search; if logic, needs search
        "recommended_model": "assistant-model",
        "constraints": list(CONSTRAINTS),
    }
    if needs_search:
        meta["intent"] = "find_and_explain_logic"
        meta["needs_tools"] = True
        meta["recommended_model"] = "research-model"
        meta["constraints"].append("Cần tìm tài liệu trong dự án để trả lời chi tiết logic")
        
    if not enriched_prompt:
        enriched_prompt = ans
        
    meta["enriched_prompt"] = clean(enriched_prompt)
    
    assistant = json.dumps(meta, ensure_ascii=False, separators=(",", ":"))
    text = f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n{clean(q)}\n<|assistant|>\n{assistant}<|end|>"
    return {"text": text}

def main():
    examples = []
    
    # 1. Simba Tables Schema
    tables = list((DOCS / "tables").glob("*.md"))
    for t_path in tables:
        info = parse_table(t_path)
        if info:
            col_str = ", ".join(info['cols'][:20])
            ans = f"Bảng {info['name']}: {info['desc']}. Cột: {col_str}..."
            
            for q_template in [
                "cấu trúc bảng {name} là gì",
                "bảng {name} gồm những cột nào",
                "ý nghĩa các trường trong bảng {name}",
                "cho anh xem schema của bảng {name}"
            ]:
                examples.append(make_ex(q_template.format(name=info['name']), ans, needs_search=False))

    # 2. Simba Procedures Logic
    procs = list((DOCS / "procedures").rglob("*.md"))
    for p_path in procs:
        info = parse_proc(p_path)
        if info:
            # Schema question
            examples.append(make_ex(
                f"proc {info['name']} dùng để làm gì",
                f"Procedure {info['name']} thuộc hệ thống Simba.",
                needs_search=False
            ))
            # Logic question (requires search)
            examples.append(make_ex(
                f"logic procedure {info['name']} chạy như thế nào",
                f"Cần đọc file procedure {info['name']} trong tài liệu dự án để giải thích chi tiết.",
                task_type="simba_logic",
                needs_search=True
            ))

    # 3. General / Behavioral / Mlops (Expanded)
    general = [
        ("em merge #1 đi", "Thực hiện merge PR #1 vào main. Kiểm tra conflict, build pass, squash merge. Sau đó xóa branch.", "github_workflow"),
        ("tạo PR cho branch feature/simba-login", "Tạo PR từ feature/simba-login sang main. Gán reviewer, kiểm tra CI.", "github_workflow"),
        ("kiểm tra git status", "Chạy git status để xem trạng thái repo. Báo cáo file nào bị modified/untracked.", "github_workflow"),
        ("push code lên GitHub", "Commit các file thay đổi. Push lên origin. Tạo tag nếu là release.", "github_workflow"),
        ("xóa branch cũ đã merge", "Liệt kê các branch đã merge. Xóa local và remote. Báo cáo kết quả.", "github_workflow"),
        ("em scan rồi dựng dataset", "Đã scan dữ liệu. Xóa dataset cũ. Dựng dataset mới từ session OpenClaw và tài liệu Simba, đảm bảo redact secret.", "dataset_engineering"),
        ("review training_data.jsonl", "Đang đọc file JSONL. Kiểm tra format, số lượng, và chất lượng dữ liệu. Báo cáo chi tiết.", "dataset_engineering"),
        ("pipeline.py có cần HF token không", "Có, pipeline.py cần HF_TOKEN để download/upload model từ Hugging Face Hub.", "mlops_pipeline"),
        ("hướng dẫn anh chạy notebook trên Colab", "Mở notebook. Cấu hình Secrets với HF_TOKEN. Chọn GPU runtime. Chạy các cell lần lượt.", "mlops_pipeline"),
        ("download model từ Hugging Face về local", "Dùng pipeline.py với step 'download'. Kiểm tra token HF. Chạy script và báo cáo tiến độ.", "mlops_pipeline"),
        ("sửa lỗi không chạy được", "Xác định nguyên nhân lỗi (log, traceback). Đề xuất cách fix. Kiểm tra lại sau khi sửa.", "debugging"),
        ("báo cáo tiến độ task", "Báo cáo: Đã hoàn thành task A, B. Đang xử lý task C. Không có issue block.", "github_workflow"),
        ("cập nhật README.md", "Cập nhật nội dung README. Bổ sung hướng dẫn cài đặt, cách dùng. Commit và push.", "documentation"),
        ("thêm chức năng mới vào hermes", "Viết code module mới. Viết test. Cập nhật config hermes. Thêm vào skill nếu cần.", "hermes_config"),
    ]
    for q, ans, ttype in general:
        m = {
            "task_type": ttype, "domain": "workspace", "framework": "Git/Hermes",
            "complexity": "medium", "intent": "execute_task",
            "needs_search": False, "needs_tools": True, "recommended_model": "coding-model",
            "constraints": CONSTRAINTS, "enriched_prompt": ans
        }
        text = f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n{clean(q)}\n<|assistant|>\n{json.dumps(m, ensure_ascii=False, separators=(',',':'))}<|end|>"
        examples.append({"text": text})
        # Add duplicates with slight variations to reinforce
        for suffix in [" nhanh lên", " giúp anh", " nhé"]:
            examples.append(make_ex(q + suffix, ans, ttype, needs_search=False))

    # 4. Simba Module General Questions
    for mod in ["GL", "AP", "AR", "SO", "PO", "IN", "FA", "HR", "CA", "SA", "SI", "CO", "Dash"]:
        examples.append(make_ex(
            f"module {mod} trong Simba làm gì",
            f"Module {mod} là một phần của SimbaERP, xử lý các nghiệp vụ liên quan. (Xem chi tiết trong tài liệu procedures/{mod}).",
            task_type="simba_schema", needs_search=False
        ))

    # Deduplicate & Shuffle
    seen = set()
    final = []
    for ex in examples:
        h = hashlib.sha256(ex['text'].encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            final.append(ex)
    random.shuffle(final)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as f:
        for ex in final:
            f.write(json.dumps(ex, ensure_ascii=False, separators=(",", ":")) + "\n")
            
    print(f"Done. {len(final)} examples.")
    print(json.dumps(json.loads(final[0]['text'].split('<|assistant|>\n')[1].replace('<|end|>','')), ensure_ascii=False, indent=2)[:300])

if __name__ == "__main__":
    main()
