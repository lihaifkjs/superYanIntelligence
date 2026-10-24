"""jsonl IO 小工具。独立成模块是为了让纯 CPU 脚本（build_sft/check_contamination）
不拖 torch。"""

import json
from pathlib import Path


def read_jsonl(path) -> list[dict]:
    items = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def append_jsonl(path, item: dict):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
