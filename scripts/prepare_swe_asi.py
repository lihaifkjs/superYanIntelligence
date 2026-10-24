"""SWE / ASI 方向：开源轨迹数据集 → 统一 SFT messages 格式。

本机只验证「格式适配 + 可训性」，真实规模的轨迹生成需要 Docker 执行环境，留云上。

数据源（ModelScope）：
- swe: AI-ModelScope/SWE-agent-trajectories — SWE-agent 在 SWE-bench 类任务上的
  真实修复轨迹，trajectory 字段为 role/text 步序列（system/ai/user）
- asi: deepmath/agent-sci-general — 科研场景 agent 多轮对话（conversations 字段）

输出 data/swe_asi_sample/{expert}.jsonl，每行 {"messages": [...]}，与 build_sft.py 产物同构。

用法：
  python prepare_swe_asi.py --expert both --n 20
"""

import argparse
import json
from pathlib import Path

from modelscope.hub.snapshot_download import snapshot_download

SWE_REPO = "AI-ModelScope/SWE-agent-trajectories"
SWE_PATTERN = ["data/train-00000-of-00012.parquet"]
ASI_REPO = "deepmath/agent-sci-general"
ASI_PATTERN = ["data/train-00000-of-00001.parquet"]

ROLE_MAP = {"system": "system", "ai": "assistant", "assistant": "assistant",
            "user": "user", "human": "user", "tool": "tool"}


def load_first_parquet(repo: str, pattern: list[str], cache_dir: str):
    import glob

    from datasets import load_dataset

    repo_dir = snapshot_download(repo, repo_type="dataset",
                                 allow_file_pattern=pattern, cache_dir=cache_dir)
    parquet = sorted(glob.glob(str(repo_dir) + "/**/*.parquet", recursive=True))
    if not parquet:
        raise RuntimeError(f"No parquet under {repo_dir}")
    return load_dataset("parquet", data_files=parquet[0], split="train")


def convert_swe(row: dict) -> dict | None:
    messages = []
    for step in row.get("trajectory") or []:
        role = ROLE_MAP.get(str(step.get("role", "")).lower())
        text = str(step.get("text", "")).strip()
        if role and text:
            messages.append({"role": role, "content": text})
    # 至少 system + user + 2 轮 assistant 才有训练价值
    n_asst = sum(1 for m in messages if m["role"] == "assistant")
    if len(messages) < 4 or n_asst < 2:
        return None
    return {"messages": messages}


def convert_asi(row: dict) -> dict | None:
    messages = []
    for turn in row.get("conversations") or []:
        role = ROLE_MAP.get(str(turn.get("role", "")).lower())
        content = str(turn.get("content", "")).strip()
        if role and role != "tool" and content:
            messages.append({"role": role, "content": content})
    n_asst = sum(1 for m in messages if m["role"] == "assistant")
    if len(messages) < 4 or n_asst < 2:
        return None
    return {"messages": messages}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert", choices=["swe", "asi", "both"], default="both")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--cache_dir", default="models/models")
    parser.add_argument("--output_dir", default="data/swe_asi_sample")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    if args.expert in ("swe", "both"):
        jobs.append(("swe", SWE_REPO, SWE_PATTERN, convert_swe))
    if args.expert in ("asi", "both"):
        jobs.append(("asi", ASI_REPO, ASI_PATTERN, convert_asi))

    for expert, repo, pattern, convert in jobs:
        print(f"Loading {repo} ...")
        ds = load_first_parquet(repo, pattern, args.cache_dir)
        kept, scanned = [], 0
        for row in ds:
            if len(kept) >= args.n:
                break
            scanned += 1
            item = convert(row)
            if item:
                kept.append(item)
        out_path = out_dir / f"{expert}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for item in kept:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"[{expert}] scanned={scanned} kept={len(kept)} -> {out_path}")


if __name__ == "__main__":
    main()
