"""HLE baseline runner：从本地 parquet 出题，打云机 vLLM 服务，规则判分出分。

用法（云机）:
    /root/miniconda3/bin/python /root/scripts/hle_baseline.py \
        --parquet /root/autodl-tmp/hle/data/test-00000-of-00001.parquet \
        --n 10 --out /root/autodl-tmp/hle_pred_10.json

全量:
    --n 2500（先确认 judge 方案再跑全量）
"""

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from judge import answers_equal  # noqa: E402  与数据过滤同一把尺子

# 与官方 hle_eval/run_model_predictions.py 一致
SYSTEM_PROMPT = (
    "Your response should be in the following format:\n"
    "Explanation: {your explanation for your answer choice}\n"
    "Answer: {your chosen answer}\n"
    "Confidence: {your confidence score between 0% and 100% for your answer}"
)

PRICE_RE = re.compile(r"￥?([\d.]+)/时")
ANSWER_RE = re.compile(r"Answer:\s*(.+)")


def gen_one(url, model, question, max_tokens, timeout=600):
    r = requests.post(
        f"{url}/chat/completions",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            "temperature": 0.0,
            "max_tokens": max_tokens,
        },
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def extract_answer(response, answer_type):
    m = ANSWER_RE.search(response)
    if not m:
        return None
    raw = m.group(1).strip()
    if answer_type == "multipleChoice":
        letter = re.search(r"\b([A-E])\b", raw)
        return letter.group(1) if letter else None
    return raw.strip().rstrip(".")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="qwen35b")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--out", default="/root/autodl-tmp/hle_pred.json")
    args = parser.parse_args()

    df = pd.read_parquet(args.parquet)
    df = df[(df["image"].isna()) | (df["image"] == "")]
    df = df.head(args.n)
    print(f"questions: {len(df)} (text-only from {args.parquet})", flush=True)

    results = []
    correct = 0
    for i, row in enumerate(df.itertuples(), 1):
        try:
            resp = gen_one(args.url, args.model, row.question, args.max_tokens)
        except Exception as e:
            print(f"[{i}] ERROR: {e}", flush=True)
            continue
        ext = extract_answer(resp, row.answer_type)
        if ext is None:
            ok = False
        elif row.answer_type == "multipleChoice":
            ok = ext.upper() == str(row.answer).strip().upper()
        else:
            # exactMatch：与数据过滤同口径（judge.answers_equal）
            ok = answers_equal(ext, str(row.answer))
        correct += ok
        results.append({
            "id": row.id, "answer_type": row.answer_type,
            "gold": row.answer, "extracted": ext, "correct": bool(ok),
        })
        print(f"[{i}/{len(df)}] {row.answer_type} gold={row.answer!r} ext={ext!r} {'✓' if ok else '✗'}", flush=True)

    Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2))
    acc = round(100 * correct / max(len(results), 1), 2)
    print(f"*** Metrics ***", flush=True)
    print(f"Accuracy(rule-judge): {acc}% | n = {len(results)} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
