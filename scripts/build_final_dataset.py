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

    # 5. Agent Failure Patterns (learned from session history)
    # These teach the agent to handle common mistakes and avoid repeating them
    agent_failure_patterns = [
        # Web provider config mistakes
        ("em config web search dùng nine-router",
         "Đã cấu hình nine-router cho web.search_backend. Lưu ý: nine-router chỉ hỗ trợ search, không dùng được cho extract. Giữ web.extract_backend là tavily.",
         "web_config"),
        ("dùng nine-router làm web extract được không",
         "Không được. nine-router chỉ hỗ trợ search endpoint. Để extract URL content, dùng tavily hoặc firecrawl làm extract_backend.",
         "web_config"),
        ("em cấu hình sai web backend rồi, sửa lại",
         "Đã nhận lỗi. Đang revert config web backend về tavily cho cả search và extract. Kiểm tra lại config.yaml sau khi sửa.",
         "error_recovery"),
        ("tại sao web extract bị lỗi với nine-router",
         "nine-router là search-only backend, không có endpoint extract. Cấu hình đúng: search_backend=tabily, extract_backend=tavily.",
         "web_config"),
        ("em set extract_backend = nine-router",
         "Sai. nine-router không hỗ trợ extract. Đã revert extract_backend về tavily. Chỉ dùng nine-router cho search.",
         "error_recovery"),

        # DNS / Network troubleshooting
        ("GitHub API bị timeout, em fix đi",
         "Đã kiểm tra DNS và API reachability. Nếu curl api.github.com timeout nhưng git push qua SSH ok, vấn đề là HTTPS/TLS route. Đã thử đổi DNS, retry sau 30s.",
         "network_debug"),
        ("em đổi DNS server sang 10.0.0.103",
         "Đã cập nhật /etc/resolv.conf. Đã kiểm tra curl api.github.com hoạt động. GitHub API đã phản hồi 200.",
         "network_debug"),
        ("tại sao gh pr merge bị timeout",
         "GraphQL API của GitHub bị TLS handshake timeout. Kiểm tra: curl -I api.github.com. Nếu API responsive thì retry. Nếu vẫn timeout, merge thủ công qua git + push.",
         "network_debug"),

        # Colab execution errors
        ("trainer is not defined trong Colab",
         "Lỗi chạy sai thứ tự cell. Cell 8 (Initialize SFTTrainer) phải chạy trước Cell 9 (trainer.train()). Hoặc dùng Run all để chạy tuần tự từ đầu.",
         "colab_debug"),
        ("lỗi NameError trong Colab khi train model",
         "Biến chưa được khai báo do nhảy cóc cell hoặc kernel bị reset. Chạy lại các cell trước theo thứ tự, không skip cell nào.",
         "colab_debug"),
        ("Colab báo lỗi NameError: name 'model' is not defined",
         "Cell load model chưa được chạy hoặc kernel bị restart. Chạy Cell 2 (Load Model) rồi mới chạy các cell sau.",
         "colab_debug"),

        # Git workflow mistakes
        ("em commit nhầm vào main rồi",
         "Đã dừng. Không push. Tạo branch mới từ main, cherry-pick commit đúng vào branch đó. Reset main về trạng thái gốc.",
         "error_recovery"),
        ("em force push lên main",
         "Dừng lại. Không force push lên main. Tạo branch mới cho thay đổi. Chỉ push lên branch feature.",
         "error_recovery"),
        ("em amend commit trên PR cũ",
         "Không sửa PR đã mở bằng amend. Tạo branch mới, commit thay đổi, mở PR mới. PR cũ giữ nguyên.",
         "error_recovery"),
        ("PR bị conflict, em resolve đi",
         "Đã fetch main mới nhất. Rebase branch feature lên main. Resolve conflict từng file. Force push branch (chỉ branch feature, không phải main).",
         "git_workflow"),

        # Token / Secret handling
        ("em hardcode HF token vào notebook",
         "Dừng. Không hardcode token. Dùng Colab Secrets (userdata.get('HF_TOKEN')) hoặc biến môi trường. Đã xóa token khỏi notebook.",
         "security"),
        ("token HF bị lộ trong git history",
         "Đã xóa token khỏi history. Tạo branch mới sạch từ origin. Không push branch chứa token. Dùng git filter-branch để xóa secret khỏi history cũ.",
         "security"),

        # Context hygiene
        ("em làm xong task rồi, giờ làm gì tiếp",
         "Đã rút gọn context. Đã kiểm tra và update skill nếu phát hiện workflow mới. Sẵn sàng cho task tiếp theo.",
         "context_hygiene"),
        ("context dài quá, em rút gọn đi",
         "Đã tóm tắt context: giữ lại task chính, kết quả, và các file quan trọng. Xóa các log không cần thiết.",
         "context_hygiene"),
    ]
    for q, ans, ttype in agent_failure_patterns:
        m = {
            "task_type": ttype, "domain": "workspace", "framework": "Hermes/Colab/Git",
            "complexity": "medium", "intent": "execute_task",
            "needs_search": False, "needs_tools": True, "recommended_model": "assistant-model",
            "constraints": list(CONSTRAINTS), "enriched_prompt": clean(ans)
        }
        text = f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n{clean(q)}\n<|assistant|>\n{json.dumps(m, ensure_ascii=False, separators=(',',':'))}<|end|>"
        examples.append({"text": text})

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
