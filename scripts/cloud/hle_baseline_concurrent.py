"""HLE baseline runner（并发版）：多个请求并行打 vLLM，为 200 题子集提速。

与 hle_baseline.py 同口径（同 system prompt / 判分 / 输出格式），仅增加 --workers 并发。
结果按题序排列，指标与串行版一致。

用法（云机）:
    /root/miniconda3/bin/python /root/scripts/hle_baseline_concurrent.py \
        --parquet /root/autodl-tmp/hle/datasets/cais--hle/snapshots/master/data/test-00000-of-00001.parquet \
        --n 200 --workers 4 --out /root/autodl-tmp/hle_pred_200.json
"""

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
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

ANSWER_RE = re.compile(r"Answer:\s*(.+)")


def gen_one(url, model, question, max_tokens, timeout=900):
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


def run_one(idx, total, url, model, row, max_tokens, save_raw):
    rec = {"id": row.id, "answer_type": row.answer_type,
           "gold": row.answer, "extracted": None, "correct": False}
    try:
        resp = gen_one(url, model, row.question, max_tokens)
    except Exception as e:
        print(f"[{idx}/{total}] ERROR: {e}", flush=True)
        return rec
    if save_raw:
        rec["raw"] = resp
    ext = extract_answer(resp, row.answer_type)
    rec["extracted"] = ext
    if ext is None:
        ok = False
    elif row.answer_type == "multipleChoice":
        ok = ext.upper() == str(row.answer).strip().upper()
    else:
        ok = answers_equal(ext, str(row.answer))
    rec["correct"] = bool(ok)
    print(f"[{idx}/{total}] {row.answer_type} gold={row.answer!r} ext={ext!r} "
          f"{'✓' if ok else '✗'}", flush=True)
    return rec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="qwen35b")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--save-raw", action="store_true",
                        help="在结果 JSON 中保留每题完整模型回答（含 raw 字段）")
    parser.add_argument("--out", default="/root/autodl-tmp/hle_pred_200.json")
    args = parser.parse_args()

    df = pd.read_parquet(args.parquet)
    df = df[(df["image"].isna()) | (df["image"] == "")]
    df = df.head(args.n)
    total = len(df)
    print(f"questions: {total} (text-only, workers={args.workers}) "
          f"from {args.parquet}", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(
            lambda t: run_one(t[0], total, args.url, args.model, t[1],
                              args.max_tokens, args.save_raw),
            enumerate(df.itertuples(), 1),
        ))

    Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2))
    correct = sum(r["correct"] for r in results)
    acc = round(100 * correct / max(len(results), 1), 2)
    print("*** Metrics ***", flush=True)
    print(f"Accuracy(rule-judge): {acc}% | n = {len(results)} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
