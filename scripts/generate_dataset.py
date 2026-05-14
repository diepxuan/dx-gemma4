#!/usr/bin/env python3
"""Synthetic Dataset Generator for Gemma 4 E2B Router Model.

Generates ~2000 training examples for fine-tuning Gemma 4 E2B to act as a
request router: understand intent, extract key info, and produce enriched
prompts for larger downstream models.

Usage:
    python generate_dataset.py [--output dataset/training_data.jsonl] [--count 2000]
"""

import json
import random
import sys
import os
from datetime import datetime
from typing import Dict, List, Any

random.seed(42)

# ============================================================
# PATTERN TEMPLATES
# ============================================================

TASK_TYPES = {
    "code_generation": {
        "intent": "Code Generation",
        "model_hint": "coding-model",
        "templates": [
            ("viết một API REST cho {domain} bằng {framework}", 
             "Bạn là senior backend developer. Tạo {framework} RESTful API cho {domain}. Bao gồm: endpoints, models, error handling. Code production-ready, type-hinted đầy đủ.", "medium"),
            ("tạo function xử lý {task} trong {language}", 
             "Viết function {task} trong {language}. Handle edge cases, viết unit tests đi kèm, clean code.", "easy"),
            ("Sếp cần viết {task} cho {domain}", 
             "Triển khai {task} cho hệ thống {domain}. Đảm bảo best practices, security, performance. Viết rõ ràng, dễ maintain.", "medium"),
            ("cài đặt {algorithm} bằng {language}", 
             "Cài đặt thuật toán {algorithm} trong {language}. Giải thích complexity, include edge cases, test cases.", "easy"),
            ("viết script {task} trong {language}", 
             "Viết script {language} để {task}. Code chạy stable, handle errors, logging đầy đủ.", "easy"),
            ("xây dựng module {domain} dùng {framework}", 
             "Phát triển module {domain} sử dụng {framework}. Design pattern phù hợp, SOLID, testable, documented.", "medium"),
        ],
        "placeholders": {
            "domain": ["user management", "payment processing", "auth system", "notification service", 
                      "file upload system", "email service", "logging system", "cache layer",
                      "search functionality", "reporting system", "dashboard backend"],
            "framework": ["FastAPI", "Flask", "Django", "Express.js", "Spring Boot", "Gin"],
            "task": ["xử lý batch data", "gửi email hàng loạt", "crawl dữ liệu", 
                    "parse JSON/XML", "generate report", "sync data từ API"],
            "algorithm": ["binary search", "quick sort", "DFS", "BFS", "A* pathfinding",
                         "dynamic programming knapsack", "merge sort", "hash table"],
            "language": ["Python", "TypeScript", "Go", "Rust", "Java", "C++"],
        }
    },

    "code_review": {
        "intent": "Code Review",
        "model_hint": "coding-model",
        "templates": [
            ("review đoạn code {task} này, có gì cần cải thiện không", 
             "Review code cho {task}. Đánh giá: code quality, security vulnerabilities, performance issues, edge cases. Đưa ra cải thiện cụ thể với ví dụ code.", "medium"),
            ("check xem code {task} của em có bug tiềm ẩn không", 
             "Phân tích code {task} tìm: edge cases, memory leaks, race conditions, security risks, anti-patterns. Report chi tiết từng issue.", "hard"),
            ("so sánh approach A và B cho {task}", 
             "So sánh 2 cách tiếp cận cho {task}: pros/cons, performance, maintainability, scalability. Recommend approach tốt nhất với lý do.", "medium"),
        ],
        "placeholders": {
            "task": ["user authentication", "payment flow", "data pipeline", "API rate limiting",
                    "database migration", "real-time notification", "file processing"],
        }
    },

    "debugging": {
        "intent": "Debugging & Troubleshooting",
        "model_hint": "debugging-model",
        "templates": [
            ("lỗi {error} khi chạy {task}, fix giúp anh", 
             "Debug lỗi {error} khi thực hiện {task}. Phân tích root cause, đưa ra fix step-by-step, ngăn ngừa regression.", "medium"),
            ("anh gặp issue {problem} trong {context}, hướng dẫn xử lý", 
             "Trợ giúp xử lý {problem} trong {context}. Đưa ra: nguyên nhân có thể, cách debug, solution step-by-step, phòng ngừa.", "hard"),
            ("application crash khi {trigger}, log là {error}, xử lý sao", 
             "Debug ứng dụng crash khi {trigger}. Lỗi: {error}. Phân tích stack trace, tìm root cause, đề xuất fix và test case.", "hard"),
        ],
        "placeholders": {
            "error": ["TypeError: cannot read property", "Connection timeout", "Memory limit exceeded",
                     "Database deadlock", "CORS error", "500 Internal Server Error",
                     "Stack overflow", "Segmentation fault", "Null pointer exception"],
            "task": ["deploy production", "run migration", "process large file", "handle concurrent requests"],
            "problem": ["memory leak", "slow query", "connection pool exhausted", "race condition",
                       "state inconsistency", "data corruption"],
            "context": ["API server", "background worker", "data processing pipeline", "frontend component"],
            "trigger": ["load test", "peak traffic", "deploy new version", "run scheduled job"],
        }
    },

    "research": {
        "intent": "Research & Analysis",
        "model_hint": "research-model",
        "templates": [
            ("tìm hiểu về {topic}, anh cần tổng quan chi tiết", 
             "Nghiên cứu toàn diện về {topic}. Bao gồm: overview, key concepts, current landscape, pros/cons, use cases, comparison với alternatives, recommendations.", "hard"),
            ("so sánh {item_a} và {item_b} cho use case {usecase}", 
             "So sánh chi tiết {item_a} vs {item_b} cho {usecase}. Tiêu chí: performance, ease of use, cost, community, scalability, learning curve. Đưa ra recommendation rõ ràng.", "medium"),
            ("anh muốn học {topic} từ đầu, roadmap cho anh", 
             "Tạo learning roadmap cho {topic} từ beginner đến advanced. Chia phases rõ ràng, milestones, tài liệu tham khảo, bài tập thực hành cho mỗi phase.", "medium"),
        ],
        "placeholders": {
            "topic": ["LLM fine-tuning", "vector databases", "microservices architecture",
                     "event-driven design", "CI/CD best practices", "Docker optimization",
                     "Kubernetes fundamentals", "GraphQL vs REST", "Redis caching strategies"],
            "item_a": ["PostgreSQL", "MongoDB", "Redis", "GraphQL", "Docker", "Kubernetes"],
            "item_b": ["MySQL", "Cassandra", "Memcached", "REST API", "Docker Compose", "Docker Swarm"],
            "usecase": ["high-traffic API", "real-time analytics", "content management",
                       "microservices", "data pipeline", "caching layer"],
        }
    },

    "data_analysis": {
        "intent": "Data Analysis & Processing",
        "model_hint": "data-model",
        "templates": [
            ("xử lý file CSV có chứa {data_type}, cần {operation}", 
             "Viết script Python xử lý file CSV chứa {data_type}. Thực hiện: {operation}. Handle missing data, outliers, validate output.", "medium"),
            ("phân tích dataset về {topic}, tìm insights", 
             "Phân tích exploratory dataset {topic}. Tìm: patterns, correlations, anomalies, trends. Visualize key findings. Suggest next steps.", "hard"),
            ("tạo visualization cho data {data_type}", 
             "Tạo visualization cho dữ liệu {data_type}. Dùng matplotlib/seaborn. Charts phù hợp dữ liệu, readable, professional, exportable.", "easy"),
        ],
        "placeholders": {
            "data_type": ["transaction records", "user behavior logs", "sales data", "sensor readings",
                         "social media metrics", "API response times", "inventory data"],
            "operation": ["clean data", "aggregate by month", "detect anomalies", "merge with other dataset"],
            "topic": ["customer churn", "product performance", "market trends", "user engagement"],
        }
    },

    "system_design": {
        "intent": "System Design",
        "model_hint": "architecture-model",
        "templates": [
            ("thiết kế hệ thống cho {system_type}, handle khoảng {load}", 
             "Thiết kế kiến trúc cho {system_type} với tải {load}. Bao gồm: high-level architecture, component breakdown, data flow, scaling strategy, failure handling, technology choices với justification.", "hard"),
            ("anh cần design {system_type} sao cho scalable", 
             "Design scalable {system_type}. Focus trên: horizontal scaling, load balancing, database sharding, caching strategy, message queues, monitoring.", "hard"),
        ],
        "placeholders": {
            "system_type": ["real-time chat", "video streaming", "e-commerce platform", 
                           "social media feed", "collaborative document editor", "IoT dashboard"],
            "load": ["1 triệu requests/ngày", "10K concurrent users", "100GB data/ngày",
                    "1000 transactions/phút"],
        }
    },

    "writing": {
        "intent": "Content Writing",
        "model_hint": "writing-model",
        "templates": [
            ("viết README cho project {project_type}", 
             "Viết README.md cho project {project_type}. Bao gồm: overview, installation, quick start, API usage, architecture, contributing guide, license. Format markdown đẹp.", "medium"),
            ("viết commit message chuẩn cho thay đổi {change_type}", 
             "Tạo commit messages theo conventional commits cho thay đổi {change_type}. Format: type(scope): description. Bao gồm cả detailed body nếu cần.", "easy"),
            ("viết email cho {purpose}", 
             "Viết email cho {purpose}. Tone professional, concise, clear call-to-action.", "easy"),
            ("viết documentation cho {api_type} API", 
             "Viết API documentation cho {api_type}. Bao gồm: endpoints, request/response schemas, authentication, error codes, examples.", "medium"),
        ],
        "placeholders": {
            "project_type": ["Python library", "React frontend", "microservice", "CLI tool", "Docker compose setup"],
            "change_type": ["thêm feature mới", "fix security bug", "refactor database layer"],
            "purpose": ["báo cáo tiến độ dự án", "đề xuất meeting với client", "follow-up sau khi deploy"],
            "api_type": ["REST", "GraphQL", "Webhook", "gRPC"],
        }
    },

    "devops": {
        "intent": "DevOps & Deployment",
        "model_hint": "devops-model",
        "templates": [
            ("thiết lập CI/CD cho project {project_type}", 
             "Setup CI/CD pipeline cho {project_type}. Bao gồm: lint, test, build, deploy stages. Environment separation (staging/production), secrets management.", "medium"),
            ("deploy app lên {platform}, cần steps cụ thể", 
             "Hướng dẫn deploy application lên {platform}. Bước chi tiết: configure, build, deploy, verify, rollback plan. Security best practices.", "medium"),
            ("viết Dockerfile cho {project_type}", 
             "Viết Dockerfile tối ưu cho {project_type}. Multi-stage build, minimal image, non-root user, health check, proper CMD.", "easy"),
        ],
        "placeholders": {
            "project_type": ["Python FastAPI app", "React SPA", "Go microservice", "Node.js API",
                           "Next.js fullstack"],
            "platform": ["AWS EC2", "Digital Ocean", "VPS", "Docker + nginx", "Kubernetes cluster"],
        }
    },

    "general": {
        "intent": "General Assistant",
        "model_hint": "assistant-model",
        "templates": [
            ("giải thích {topic} cho người mới", 
             "Giải thích {topic} dễ hiểu cho người mới bắt đầu. Dùng analogies, ví dụ thực tế, tránh jargon. Summary cuối bài.", "easy"),
            ("anh có câu hỏi về {topic}", 
             "Trả lời chi tiết về {topic}. Bao gồm: khái niệm cơ bản, cách hoạt động, best practices, common pitfalls, references.", "medium"),
            ("làm sao để {task} hiệu quả hơn", 
             "Hướng dẫn cách {task} hiệu quả nhất. Tips, best practices, tools recommended, pitfalls to avoid.", "easy"),
        ],
        "placeholders": {
            "topic": ["git workflow", "virtual environments", "dependency management",
                     "error handling best practices", "design patterns", "code organization"],
            "task": ["quản lý multiple projects", "tổ chức codebase", "debug production issues"],
        }
    },

    "translation": {
        "intent": "Translation",
        "model_hint": "writing-model",
        "templates": [
            ("dịch đoạn văn bản từ {lang_a} sang {lang_b}", 
             "Dịch văn bản từ {lang_a} sang {lang_b}. Giữ nguyên: technical terms, format, tone. Natural translation phù hợp ngữ cảnh.", "easy"),
            ("dịch document kỹ thuật về {topic}", 
             "Dịch document kỹ thuật {topic} sang tiếng Việt. Giữ technical terms gốc tiếng Anh khi không có dịch chuẩn. Format consistent.", "medium"),
        ],
        "placeholders": {
            "lang_a": ["tiếng Anh", "tiếng Pháp", "tiếng Trung", "tiếng Nhật"],
            "lang_b": ["tiếng Việt", "tiếng Anh"],
            "topic": ["API documentation", "technical specs", "user manual", "release notes"],
        }
    },
}

# ============================================================
# ROUTING OUTPUT TEMPLATES
# ============================================================

# Các output template này chỉ làm reference, actual generation dùng json.dumps trong generate_example
# Không dùng runtime, chỉ để document structure

SYSTEM_PROMPTS = [
    "",
    "Bạn là router AI agent, phân loại và làm giàu prompt cho các task.",
    "Input là tiếng Việt, output là JSON routing decision + enriched prompt.",
    "Bạn là assistant hiểu tiếng Việt, chuyên phân tích yêu cầu của Sếp.",
]

def fill_template(template: str, placeholders: Dict[str, List[str]]) -> str:
    """Fill placeholders in template with random values."""
    import re
    result = template
    for key in re.findall(r'\{(\w+)\}', template):
        if key in placeholders:
            result = result.replace(f"{{{key}}}", random.choice(placeholders[key]), 1)
    return result

def generate_example(task_type: str, config: Dict) -> Dict[str, str]:
    """Generate a single training example."""
    template, enriched_template, complexity = random.choice(config["templates"])
    
    input_text = fill_template(template, config["placeholders"])
    
    # Build enriched prompt
    enriched = fill_template(enriched_template, config["placeholders"])
    
    # Build output JSON (manually to avoid format conflicts with { braces)
    output_config = {k: random.choice(v) for k, v in config["placeholders"].items()}
    
    # Map task_type to output structure
    task_type_mapping = {
        "general_inquiry": "general"
    }
    
    output_json = json.dumps({
        "task_type": task_type,
        "domain": output_config.get("domain", "general"),
        "framework": output_config.get("framework", "N/A"),
        "complexity": complexity,
        "enriched_prompt": enriched,
        "needs_search": task_type in ("research",),
        "recommended_model": {
            "code_generation": "coding-model",
            "code_review": "coding-model",
            "debugging": "debugging-model",
            "research": "research-model",
            "data_analysis": "data-model",
            "system_design": "architecture-model",
            "writing": "writing-model",
            "devops": "devops-model",
            "general": "assistant-model",
            "translation": "writing-model",
        }.get(task_type, "assistant-model"),
    }, ensure_ascii=False)
    
    # Add system prompt sometimes
    if random.random() < 0.3:
        system = random.choice(SYSTEM_PROMPTS)
        full_input = f"<|system|>\n{system}\n<|user|>\n{input_text}\n<|assistant|>"
    else:
        full_input = f"<|user|>\n{input_text}\n<|assistant|>"
    
    return {
        "text": f"{full_input}\n{output_json}<|end|>"
    }

def generate_dataset(count: int = 2000) -> List[Dict[str, str]]:
    """Generate the complete dataset."""
    dataset = []
    
    # Distribute evenly across task types
    task_types = list(TASK_TYPES.keys())
    per_type = count // len(task_types)
    remainder = count % len(task_types)
    
    for i, task_type in enumerate(task_types):
        n = per_type + (1 if i < remainder else 0)
        config = TASK_TYPES[task_type]
        
        for _ in range(n):
            dataset.append(generate_example(task_type, config))
    
    # Shuffle
    random.shuffle(dataset)
    return dataset

def add_edge_cases(dataset: List[Dict]) -> List[Dict]:
    """Add edge cases for robustness."""
    edge_cases = [
        # Short/vague queries
        "Sếp ơi fix bug giúp",
        "code chạy không được",
        "làm thế nào",
        "giải thích giúp",
        "có cách nào nhanh hơn không",
        
        # Very detailed queries
        "Em cần viết một API RESTful bằng FastAPI cho user management với đầy đủ JWT auth, pagination, filtering, unit tests, Docker, và CI/CD pipeline. Code theo clean architecture pattern.",
        
        # Vietnamese with tech terms mixed
        "Implement cái authentication flow cho cái SPA dùng React, backend là Node.js Express, database PostgreSQL",
        
        # Follow-up context
        "Như đã nói ở trên, giờ em cần thêm rate limiting vào API đó",
        
        # Code-like input
        "def process_data(data: list) -> dict: return {k: v for k, v in data if v is not None}\ncái này có issue gì không?",
        
        # Error messages
        "ERROR: could not resolve dependency: torch>=2.0 but only found 1.13.1",
        
        # Asking for opinion
        "Theo anh, nên dùng Go hay Python cho microservice mới? Project cần low latency",
    ]
    
    for case in edge_cases:
        dataset.append({
            "text": f"<|user|>\n{case}\n<|assistant|>\n{{\"task_type\":\"general\",\"complexity\":\"medium\",\"enriched_prompt\":\"Phân tích và làm giàu prompt cho: {case}\",\"needs_search\":false,\"recommended_model\":\"assistant-model\"}}<|end|>"
        })
    
    return dataset

def main():
    output_path = os.path.join(os.path.dirname(__file__), "dataset", "training_data.jsonl")
    count = 2000
    seed = random.randint(1000, 9999)
    random.seed(seed)
    
    # Parse args
    for i, arg in enumerate(sys.argv):
        if arg == "--output" and i + 1 < len(sys.argv):
            output_path = sys.argv[i + 1]
        elif arg == "--count" and i + 1 < len(sys.argv):
            count = int(sys.argv[i + 1])
    
    print(f"Generating {count} training examples...")
    
    dataset = generate_dataset(count)
    dataset = add_edge_cases(dataset)
    random.shuffle(dataset)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for example in dataset:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    
    print(f"Written {len(dataset)} examples to {output_path}")
    
    # Print stats
    stats = {"total": len(dataset)}
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            text = data["text"]
            if "general" in text:
                stats.setdefault("general", 0)
                stats["general"] += 1
    
    print(f"Dataset size: {os.path.getsize(output_path) / 1024:.1f} KB")
    print("Done!")

if __name__ == "__main__":
    main()
