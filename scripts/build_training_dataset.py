#!/usr/bin/env python3
"""Build concise router training_data.jsonl from current project docs and OpenClaw sessions.

Goal: teach dx-gemma4 to understand the user's project/workspace and produce clear
enriched prompts for input questions/commands. This script deliberately avoids the
old broad synthetic dataset and creates short, reviewed examples.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "scripts" / "dataset"
IMPORTS_DIR = ROOT / "imports"

SYSTEM = "Bạn là router cho Sếp. Phân tích input và viết prompt ngắn, đúng dự án, không lộ secret."

SECRET_PATTERNS = [
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"sk-or-[A-Za-z0-9_-]{20,}"),
    re.compile(r"tvly-[A-Za-z0-9_-]{20,}"),
    re.compile(r"(?i)authorization:\s*bearer\s+[^\s'\"]+"),
    re.compile(r"-----BEGIN OPENSSH PRIVATE KEY-----.*?-----END OPENSSH PRIVATE KEY-----", re.S),
]

BANNED_AFTER_REDACT = ["hf_", "ghp_", "github_pat_", "sk-or-", "tvly-", "OPENSSH PRIVATE KEY"]

TASK_TYPES = {
    "github_workflow",
    "dataset_engineering",
    "project_question",
    "code_generation",
    "code_review",
    "debugging",
    "documentation",
    "research",
    "mlops_pipeline",
    "hermes_config",
    "devops",
}


def redact(text: str) -> str:
    text = text or ""
    for pat in SECRET_PATTERNS:
        text = pat.sub("[REDACTED_SECRET]", text)
    # redact long high-entropy-ish tokens after key names
    text = re.sub(r"(?i)(api[_-]?key|token|password|secret)\s*[:=]\s*[^\s`'\"]+", r"\1=[REDACTED_SECRET]", text)
    # collapse paths that are too personal but keep project names when useful
    text = text.replace("/root/.openclaw/workspace/projects/", "[OPENCLAW_PROJECTS]/")
    text = text.replace("/root/dx-gemma4", "dx-gemma4")
    return text.strip()


def norm_space(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def short(text: str, max_len: int = 220) -> str:
    text = norm_space(redact(text))
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "…"


def classify(user: str) -> dict[str, Any]:
    u = user.lower()
    out = {
        "task_type": "project_question",
        "domain": "workspace",
        "framework": "N/A",
        "complexity": "medium",
        "intent": "answer_project_question",
        "needs_search": False,
        "needs_tools": False,
        "recommended_model": "assistant-model",
    }
    if any(k in u for k in ["pull request", "pr", "merge", "commit", "push", "branch", "github", "git "]):
        out.update(task_type="github_workflow", domain="repository_management", framework="GitHub/Git", intent="manage_github_workflow", needs_tools=True, recommended_model="coding-model")
    if any(k in u for k in ["training_data", "dataset", "jsonl", "fine tune", "fine-tune", "finetune"]):
        out.update(task_type="dataset_engineering", domain="dx-gemma4", framework="JSONL/Hugging Face", intent="build_or_review_training_dataset", needs_tools=True, recommended_model="coding-model")
    if any(k in u for k in ["colab", "notebook", "hf", "hugging face", "model", "pipeline.py"]):
        out.update(task_type="mlops_pipeline", domain="dx-gemma4", framework="Colab/Hugging Face Hub", intent="operate_mlops_pipeline", needs_tools=True, recommended_model="coding-model")
    if any(k in u for k in ["web search", "web fetch", "9router", "tavily", "hermes", "provider"]):
        out.update(task_type="hermes_config", domain="Hermes Agent", framework="Hermes config", intent="configure_or_verify_hermes", needs_tools=True, recommended_model="coding-model")
    if any(k in u for k in ["debug", "lỗi", "error", "fix", "sửa lỗi", "không chạy"]):
        out.update(task_type="debugging", intent="debug_issue", needs_tools=True, recommended_model="debugging-model")
    if any(k in u for k in ["readme", "tài liệu", "documentation", "hướng dẫn", "báo cáo"]):
        out.update(task_type="documentation", intent="write_or_report_documentation", recommended_model="writing-model")
    if any(k in u for k in ["tìm hiểu", "research", "so sánh", "phương án"]):
        out.update(task_type="research", intent="research_and_compare_options", needs_search=True, recommended_model="research-model")
    if any(k in u for k in ["code", "script", "module", "function", "api"]):
        out.update(task_type="code_generation", intent="implement_code", needs_tools=True, recommended_model="coding-model")
    if any(k in u for k in ["review", "kiểm tra"]):
        out.update(task_type="code_review", intent="review_or_validate", needs_tools=True, recommended_model="coding-model")
    return out


def constraints_for(meta: dict[str, Any]) -> list[str]:
    base = ["trả lời tiếng Việt, gọi user là Sếp", "ngắn gọn, rõ ràng", "không lộ secret"]
    t = meta["task_type"]
    if t == "github_workflow":
        base += ["không push/merge nếu chưa được phép", "kiểm tra trạng thái git trước khi hành động"]
    if t == "dataset_engineering":
        base += ["xóa dataset cũ nếu được yêu cầu", "mỗi dòng JSONL phải parse được", "dataset phải bám tài liệu dự án"]
    if t == "mlops_pipeline":
        base += ["không hardcode HF token", "model lưu ở Hugging Face Hub diepxuan/dx-gemma4"]
    if t == "hermes_config":
        base += ["chỉ sửa runtime config nếu Sếp yêu cầu", "nhắc restart/reset khi cần"]
    return base[:6]


def enriched_prompt(user: str, meta: dict[str, Any]) -> str:
    u = short(user, 180)
    t = meta["task_type"]
    if t == "dataset_engineering":
        return f"Xử lý yêu cầu dataset cho dx-gemma4: {u}. Dựa trên tài liệu dự án hiện tại, tạo prompt/output ngắn gọn, kiểm tra từng dòng JSONL, redact secret và báo cáo số liệu chất lượng."
    if t == "github_workflow":
        return f"Thực hiện workflow GitHub cho repo diepxuan/dx-gemma4 theo yêu cầu: {u}. Kiểm tra branch/status, commit đúng phần cần thiết, tránh secret, tạo hoặc xử lý PR theo quyền Sếp đã cấp."
    if t == "mlops_pipeline":
        return f"Hỗ trợ pipeline dx-gemma4 theo yêu cầu: {u}. Tôn trọng flow GitHub lưu code/dataset, Colab train notebook, Hugging Face Hub lưu model, local pipeline.py tải model về."
    if t == "hermes_config":
        return f"Hỗ trợ cấu hình Hermes theo yêu cầu: {u}. Kiểm tra config/env an toàn, không in secret, giữ fallback nếu có, và nhắc reload session khi cấu hình đổi."
    if t == "documentation":
        return f"Viết hoặc cập nhật tài liệu/báo cáo cho yêu cầu: {u}. Nội dung phải ngắn, đúng dự án, có bước thực hiện và lưu ý bảo mật nếu liên quan."
    if t == "research":
        return f"Nghiên cứu và so sánh phương án cho yêu cầu: {u}. Ưu tiên nguồn đáng tin, kết luận rõ lựa chọn phù hợp với workspace của Sếp."
    if t == "debugging":
        return f"Debug yêu cầu: {u}. Xác định hiện trạng, nguyên nhân gốc, cách sửa an toàn, bước kiểm chứng và báo cáo ngắn gọn."
    return f"Phân tích yêu cầu của Sếp: {u}. Viết prompt rõ mục tiêu, bám dự án hiện tại, nêu ràng buộc quan trọng và đầu ra mong muốn."


def make_example(user: str, source: str, meta_override: dict[str, Any] | None = None) -> dict[str, str] | None:
    user = short(user, 260)
    if len(user) < 8:
        return None
    if any(b in user for b in BANNED_AFTER_REDACT):
        return None
    meta = classify(user)
    if meta_override:
        meta.update(meta_override)
    meta["constraints"] = constraints_for(meta)
    meta["source"] = source
    meta["enriched_prompt"] = enriched_prompt(user, meta)
    # keep output concise and stable
    ordered = {
        "task_type": meta["task_type"],
        "domain": meta["domain"],
        "framework": meta["framework"],
        "complexity": meta["complexity"],
        "intent": meta["intent"],
        "needs_search": meta["needs_search"],
        "needs_tools": meta["needs_tools"],
        "recommended_model": meta["recommended_model"],
        "constraints": meta["constraints"],
        "enriched_prompt": meta["enriched_prompt"],
    }
    assistant = json.dumps(ordered, ensure_ascii=False, separators=(",", ":"))
    text = f"<|system|>\n{SYSTEM}\n<|user|>\n{user}\n<|assistant|>\n{assistant}<|end|>"
    return {"text": text}


def extract_text_from_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return ""


def session_users(source_dir: Path) -> Iterable[tuple[str, str]]:
    for path in sorted(source_dir.glob("*.jsonl*")):
        if ".trajectory" in path.name or ".checkpoint" in path.name:
            continue
        try:
            with path.open(errors="ignore") as fh:
                for line in fh:
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    if obj.get("type") != "message":
                        continue
                    msg = obj.get("message") or {}
                    if msg.get("role") != "user":
                        continue
                    text = extract_text_from_message_content(msg.get("content"))
                    text = redact(text)
                    if "System (untrusted):" in text or "Sender (untrusted metadata)" in text:
                        continue
                    text = re.sub(r'^\[[^\]]+\]\s*', '', text).strip()
                    if text:
                        yield text, path.name
        except Exception:
            continue


def project_doc_examples(project_docs: Path) -> list[dict[str, str]]:
    examples: list[dict[str, str]] = []
    # Curated examples from current docs and imported Simba docs structure.
    curated = [
        "Sếp hỏi dx-gemma4 là dự án gì và dùng để làm gì",
        "hướng dẫn anh chạy notebook dx-gemma4 trên Colab",
        "pipeline.py của dx-gemma4 chạy ở local hay GitHub Actions",
        "khi nào pipeline cần HF_TOKEN",
        "review training_data.jsonl và đề xuất cách cập nhật theo tài liệu dự án",
        "dựng dataset router từ session OpenClaw, phải redact secret",
        "xóa dataset cũ và tạo training_data.jsonl mới ngắn gọn theo tài liệu dự án",
        "tạo PR cập nhật dataset cho repo diepxuan/dx-gemma4",
        "merge PR trên GitHub sau khi đã kiểm tra không có secret",
        "kiểm tra web search 9router và web fetch Tavily trong Hermes",
        "hướng dẫn cấu hình Colab Secrets cho HF_TOKEN",
        "download model mới từ Hugging Face Hub về local bằng pipeline.py",
        "Simba có những module tài liệu nào trong project docs",
        "tra cứu bảng GlCt trong tài liệu Simba và viết prompt phân tích ngắn gọn",
        "giải thích stored procedure module AR trong tài liệu Simba",
        "tạo prompt để review tài liệu procedures GL của Simba",
        "tìm function afRound trong tài liệu Simba và nêu cách sử dụng",
        "viết prompt cho agent khác đọc tài liệu Simba trước khi sửa code",
    ]
    for c in curated:
        ex = make_example(c, "project_docs")
        if ex:
            examples.append(ex)
    # Add file-aware questions from imported docs, concise.
    for path in sorted(project_docs.rglob("*.md"))[:400]:
        rel = path.relative_to(project_docs)
        name = path.stem
        parts = rel.parts
        if len(parts) >= 2 and parts[0] == "simba":
            if parts[1] in {"tables", "functions"}:
                q = f"tra cứu {parts[1][:-1]} {name} trong tài liệu Simba và viết prompt trả lời cho Sếp"
            elif parts[1] == "procedures":
                q = f"đọc tài liệu procedures Simba {rel} và tóm tắt đúng trọng tâm"
            else:
                q = f"dùng tài liệu Simba {rel} để trả lời câu hỏi dự án"
            ex = make_example(q, "project_docs")
            if ex:
                examples.append(ex)
    return [e for e in examples if e]


def manual_core_examples() -> list[dict[str, str]]:
    prompts = [
        "em scan rồi dựng dataset",
        "xoá toàn bộ dataset cũ",
        "review từng dòng dataset, tuân thủ theo tài liệu dự án hiện tại",
        "dataset phải ngắn gọn, rõ ràng, mục tiêu là hiểu dự án và viết promt chuẩn xác cho câu hỏi input",
        "em có phương án lấy session của OpenClaw để dựng training_data.jsonl không",
        "báo cáo anh phương án sử dụng web search, web fetch custom provider 9router, không sửa code base",
        "kiểm tra web search và web fetch đã thành công hết chưa",
        "đổi tên folder gemma4-router-finetune thành dx-gemma4",
        "em commit những gì cần thiết rồi tạo PR",
        "em merge #1 đi",
        "đã có code trên github, bước tiếp theo cần làm gì",
        "hướng dẫn anh chạy trên notebook",
        "trên local đã có hf auth chưa",
    ]
    examples: list[dict[str, str]] = []
    for p in prompts:
        ex = make_example(p, "curated_session")
        if ex:
            examples.append(ex)
    return examples


def validate_examples(examples: list[dict[str, str]]) -> tuple[list[dict[str, str]], dict[str, Any]]:
    clean = []
    seen = set()
    errors = []
    counts = Counter()
    for idx, ex in enumerate(examples, 1):
        text = ex.get("text", "")
        h = hashlib.sha256(text.encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        err = None
        if any(b in text for b in BANNED_AFTER_REDACT):
            err = "secret_pattern"
        m = re.search(r"<\|assistant\|>\n(.*)<\|end\|>", text, re.S)
        if not m:
            err = "missing_assistant"
        else:
            try:
                out = json.loads(m.group(1))
                if out.get("task_type") not in TASK_TYPES:
                    err = "bad_task_type"
                counts[out.get("task_type")] += 1
            except Exception:
                err = "assistant_json_parse"
        if err:
            errors.append({"line": idx, "error": err, "preview": text[:160]})
        else:
            clean.append(ex)
    report = {"total_input": len(examples), "total_clean": len(clean), "errors": errors[:50], "task_type_counts": dict(counts)}
    return clean, report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=str(DATASET_DIR / "training_data.jsonl"))
    ap.add_argument("--max-session", type=int, default=450)
    args = ap.parse_args()

    random.seed(42)
    examples: list[dict[str, str]] = []
    examples.extend(manual_core_examples())
    examples.extend(project_doc_examples(IMPORTS_DIR / "project_docs"))

    # sample real user commands from sessions; prioritize project/Git/dataset/Hermes patterns.
    session_candidates = []
    for user, src in session_users(IMPORTS_DIR / "openclaw_sessions"):
        u = user.lower()
        if len(user) > 500 or len(user) < 8:
            continue
        score = 0
        for kw in ["github", "pr", "commit", "merge", "dataset", "training", "simba", "hermes", "colab", "hf", "báo cáo", "bao cao", "kiểm tra", "kiem tra", "hướng dẫn", "huong dan"]:
            if kw in u:
                score += 1
        if score:
            session_candidates.append((score, user, src))
    session_candidates.sort(key=lambda x: (-x[0], len(x[1])))
    for _, user, src in session_candidates[: args.max_session]:
        ex = make_example(user, f"openclaw:{src}")
        if ex:
            examples.append(ex)

    clean, report = validate_examples(examples)
    random.shuffle(clean)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    # delete old dataset by replacing whole file
    with out.open("w", encoding="utf-8") as f:
        for ex in clean:
            f.write(json.dumps(ex, ensure_ascii=False, separators=(",", ":")) + "\n")

    reports = IMPORTS_DIR / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    report.update({"output": str(out), "session_candidates": len(session_candidates)})
    (reports / "training_data_build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2)[:5000])


if __name__ == "__main__":
    main()
