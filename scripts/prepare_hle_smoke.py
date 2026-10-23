import json
import random
import argparse
from pathlib import Path

SYSTEM_PROMPT = (
    "You are a helpful assistant. Solve the given problem step by step. "
    "Wrap your detailed reasoning inside <think>...</think> tags, "
    "and provide the final answer after the reasoning."
)


def make_math_pairs(n: int) -> list[dict]:
    """Generate simple math problems with thinking traces."""
    samples = []
    for _ in range(n):
        a = random.randint(2, 20)
        b = random.randint(2, 20)
        s = a + b
        p = a * b
        q = f"If a + b = {s} and a × b = {p}, what is a² + b²?"
        reasoning = (
            f"We know (a+b)² = a² + 2ab + b², "
            f"so a² + b² = (a+b)² - 2ab = {s}² - 2×{p} = {s*s} - {2*p} = {s*s - 2*p}."
        )
        answer = str(s * s - 2 * p)
        samples.append({"question": q, "thinking": reasoning, "answer": answer})
    return samples


def make_logic_pairs(n: int) -> list[dict]:
    """Generate simple logic puzzles with thinking traces."""
    templates = [
        {
            "question": "Tom is taller than Jerry, and Jerry is taller than Mike. Who is the shortest?",
            "thinking": "Tom > Jerry > Mike in height, so Mike is the shortest.",
            "answer": "Mike",
        },
        {
            "question": "A train travels 60 km in 1 hour. How far does it travel in 3 hours at the same speed?",
            "thinking": "Distance = speed × time = 60 km/h × 3 h = 180 km.",
            "answer": "180 km",
        },
        {
            "question": "What is the next number in the sequence: 2, 4, 8, 16, ...?",
            "thinking": "Each term is multiplied by 2. The next term is 16 × 2 = 32.",
            "answer": "32",
        },
    ]
    samples = []
    for _ in range(n):
        samples.append(random.choice(templates).copy())
    return samples


def build_chat(sample: dict) -> dict:
    content = f"<think>\n{sample['thinking']}\n</think>\n\n{sample['answer']}"
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": sample["question"]},
            {"role": "assistant", "content": content},
        ]
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/hle_smoke.jsonl")
    parser.add_argument("--n_math", type=int, default=25)
    parser.add_argument("--n_logic", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    raw = make_math_pairs(args.n_math) + make_logic_pairs(args.n_logic)
    random.shuffle(raw)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for s in raw:
            f.write(json.dumps(build_chat(s), ensure_ascii=False) + "\n")

    print(f"Wrote {len(raw)} samples to {out_path}")


if __name__ == "__main__":
    main()
