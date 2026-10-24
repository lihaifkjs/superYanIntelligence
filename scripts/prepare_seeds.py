"""生成各专家方向的种子任务集。

统一 schema（每行一个 json）：
{
  "id": "T01",
  "expert": "terminal" | "hle" | "swe" | "asi",
  "task": "任务的自然语言描述（作为蒸馏时的 user prompt）",
  "ground_truth": "可选，标准答案（HLE 用于正确性过滤）",
  "meta": {
    "setup": "可选，Terminal 沙箱执行前的初始化命令（bash）",
    "check": "可选，Terminal 完成性检查命令，exit 0 即通过"
  }
}

用法：
  python prepare_seeds.py --expert terminal          # 内置 30 条终端任务
  python prepare_seeds.py --expert hle --n 300       # ModelScope 数学竞赛题抽样
"""

import argparse
import json
import random
from pathlib import Path

from seed_bank import build_generated_terminal, build_terminal_seeds

def build_hle_seeds(n: int, seed: int, dataset_repo: str, cache_dir: str) -> list[dict]:
    """从 ModelScope 数学竞赛数据集抽题。只下载 parquet 分片，不拉全量。"""
    from modelscope.hub.snapshot_download import snapshot_download

    print(f"Downloading dataset parquet from ModelScope: {dataset_repo}")
    repo_dir = snapshot_download(
        dataset_repo,
        repo_type="dataset",
        allow_file_pattern=["*.parquet", "data/*.parquet"],
        cache_dir=cache_dir,
    )
    parquet_files = sorted(Path(repo_dir).rglob("*.parquet"))
    if not parquet_files:
        raise RuntimeError(f"No parquet found under {repo_dir}")
    print(f"Found {len(parquet_files)} parquet files, loading ...")

    from datasets import load_dataset

    ds = load_dataset("parquet", data_files=[str(p) for p in parquet_files], split="train")
    print(f"Dataset columns: {ds.column_names}, rows: {len(ds)}")

    # 字段名防御式适配：不同数据集题目/答案字段不同
    problem_key = next((k for k in ("problem", "question", "query") if k in ds.column_names), None)
    answer_key = next((k for k in ("answer", "final_answer", "solution") if k in ds.column_names), None)
    if problem_key is None:
        raise RuntimeError(f"Cannot find problem field in columns: {ds.column_names}")

    rng = random.Random(seed)
    idx = list(range(len(ds)))
    rng.shuffle(idx)

    seeds = []
    for i in idx:
        if len(seeds) >= n:
            break
        row = ds[i]
        problem = str(row[problem_key]).strip()
        if not (30 <= len(problem) <= 1500):
            continue
        gt = str(row[answer_key]).strip() if answer_key else ""
        seeds.append({
            "id": f"H{len(seeds)+1:04d}",
            "expert": "hle",
            "task": problem,
            "ground_truth": gt or None,
            "meta": {"source": dataset_repo},
        })
    return seeds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert", choices=["terminal", "hle", "all"], default="all")
    parser.add_argument("--n", type=int, default=300, help="HLE 抽样题数")
    parser.add_argument("--n_generated", type=int, default=0,
                        help="Terminal 模板化生成任务数（0 = 只要内置 30 条）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", default="AI-ModelScope/Big-Math-RL-Verified",
                        help="HLE 种子来源（ModelScope 数据集 repo）")
    parser.add_argument("--cache_dir", default="models/models")
    parser.add_argument("--output_dir", default="data/seeds")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.expert in ("terminal", "all"):
        seeds = build_terminal_seeds()
        if args.n_generated > 0:
            seeds += build_generated_terminal(args.n_generated, args.seed)
        path = out_dir / "terminal.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for s in seeds:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"Wrote {len(seeds)} terminal seeds "
              f"(builtin 30 + generated {max(args.n_generated, 0)}) to {path}")

    if args.expert in ("hle", "all"):
        seeds = build_hle_seeds(args.n, args.seed, args.dataset, args.cache_dir)
        path = out_dir / "hle.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for s in seeds:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"Wrote {len(seeds)} hle seeds to {path}")


if __name__ == "__main__":
    main()
