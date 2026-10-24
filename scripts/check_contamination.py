"""污染检测：训练数据 × 评测集的 n-gram 重叠检查（W2 出口要求）。

对训练集每条样本的文本做 n-gram 切分建集合，再逐条扫评测集，
报告：评测集中与训练集共享任一 n-gram 的题目占比（越低越好）。

评测集文件支持：
- jsonl：取每行 task / question / problem 字段
- 其他文本文件：整行作为题目文本

用法：
  python check_contamination.py --train data/sft/hle.jsonl \
      --eval data/hle/datasets/caiss--hle/test.jsonl --n 8
"""

import argparse
import json
import re
from pathlib import Path

NGRAM_RE = re.compile(r"[a-z0-9]+")


def text_of(record: dict) -> str:
    for k in ("task", "question", "problem"):
        if record.get(k):
            return str(record[k])
    return ""


def ngrams(text: str, n: int) -> set[tuple]:
    tokens = NGRAM_RE.findall(text.lower())
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def load_eval_texts(path: str) -> list[str]:
    p = Path(path)
    texts = []
    if p.suffix == ".parquet":
        from datasets import load_dataset

        ds = load_dataset("parquet", data_files=str(p), split="train")
        key = next((k for k in ("question", "problem", "prompt", "task")
                    if k in ds.column_names), None)
        if key is None:
            raise RuntimeError(f"No question-like column in {ds.column_names}")
        texts = [str(r) for r in ds[key]]
    elif p.suffix == ".jsonl":
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                texts.append(text_of(json.loads(line)))
    else:
        texts = [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    return [t for t in texts if t]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="SFT 数据 jsonl")
    parser.add_argument("--eval", dest="eval_files", nargs="+", required=True)
    parser.add_argument("--n", type=int, default=8)
    args = parser.parse_args()

    train_ngrams = set()
    n_train = 0
    with Path(args.train).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            text = text_of(rec)
            if not text:  # tool-call 数据：拼全部 user content
                text = " ".join(
                    m.get("content", "") for m in rec.get("messages", [])
                    if m.get("role") == "user"
                )
            grams = ngrams(text, args.n)
            if grams:
                n_train += 1
                train_ngrams |= grams
    print(f"train samples with >={args.n} tokens: {n_train}, "
          f"unique {args.n}-grams: {len(train_ngrams)}")

    for eval_path in args.eval_files:
        texts = load_eval_texts(eval_path)
        hits = sum(1 for t in texts if ngrams(t, args.n) & train_ngrams)
        rate = hits / max(len(texts), 1)
        flag = "PASS" if rate < 0.01 else "REVIEW"
        print(f"[{flag}] {eval_path}: {hits}/{len(texts)} eval items overlap "
              f"({rate:.2%})")


if __name__ == "__main__":
    main()
