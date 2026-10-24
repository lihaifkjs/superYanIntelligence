"""W2 第 0 步：35B 蒸馏输出格式验证（云上执行，5 分钟，不可跳过）。

放量蒸馏前确认：
1. 35B 的 thinking 输出能被解析器接住（reasoning 非空）
2. ANSWER 行约定被遵守（answer 非空）
3. 人工目检 RAW 全文，确认 thinking 标记形态（兼容裸文本/<think>/特殊 token）

用法（云机）：
  /root/miniconda3/bin/python scripts/cloud/w2_step0_format_check.py \
      --model /root/autodl-tmp/q36 [--n 5]

exit 0 = 通过；exit 1 = 不通过（看 RAW 输出改 judge.py 后重跑）。
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from judge import parse_qa_output  # noqa: E402
from llm_io import generate, load_model_and_tokenizer  # noqa: E402
from prompts import SYSTEM_PROMPT_QA  # noqa: E402

PROBES = [
    "What is 17 + 25?",
    "If a rectangle has length 8 and width 5, what is its area?",
    "What is the greatest common divisor of 48 and 180?",
    "A bag has 3 red and 4 blue balls. What is the probability of drawing a red ball?",
    "Solve for x: 2x + 6 = 20.",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    args = parser.parse_args()

    model, tok = load_model_and_tokenizer(args.model)
    ok = 0
    for i, q in enumerate(PROBES[:args.n], 1):
        msgs = [{"role": "system", "content": SYSTEM_PROMPT_QA},
                {"role": "user", "content": q}]
        raw = generate(msgs, model, tok, max_new_tokens=args.max_new_tokens,
                       temperature=0.7)
        parsed = parse_qa_output(raw)
        print(f"{'=' * 20} PROBE {i}: {q}")
        print("----- RAW -----")
        print(raw)
        print("----- PARSED -----")
        print(f"reasoning_len={len(parsed['reasoning'])}  answer={parsed['answer']!r}")
        if parsed["reasoning"] and parsed["answer"]:
            ok += 1
        else:
            print("!!!!! PARSE EMPTY —— 此条未通过")

    print(f"\nPARSED OK {ok}/{min(args.n, len(PROBES))}")
    if ok == min(args.n, len(PROBES)):
        print("PARSE OK, reasoning_len>0 —— 第 0 步通过（仍需人工目检 RAW 标记形态）")
        return
    print("PARSE FAILED —— 看上方 RAW，改 scripts/judge.py 的 THINK_RE/_DECODE_STRIP 后重跑")
    sys.exit(1)


if __name__ == "__main__":
    main()
