"""HLE harness smoke test.

Loads the local HLE parquet, runs N text-only questions through a local model
with the official HLE system prompt, then scores with a rule-based judge
(exact match / multiple choice extraction).

Official LLM judge (run_judge_results.py) is used later on cloud with a
strong judge model; this script validates the full scoring link locally.
"""

import argparse
import re

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Identical to external/hle/hle_eval/run_model_predictions.py
SYSTEM_PROMPT = (
    "Your response should be in the following format:\n"
    "Explanation: {your explanation for your answer choice}\n"
    "Answer: {your chosen answer}\n"
    "Confidence: {your confidence score between 0% and 100% for your answer}"
)


def load_questions(parquet_path, n):
    df = pd.read_parquet(parquet_path)
    text_only = df[(df["image"].isna()) | (df["image"] == "")]
    return text_only.head(n).to_dict("records")


def generate(model, tokenizer, question, max_new_tokens):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )


def extract_answer(response, answer_type):
    m = re.search(r"Answer:\s*(.+)", response)
    if not m:
        return None
    raw = m.group(1).strip()
    if answer_type == "multipleChoice":
        letter = re.search(r"\b([A-E])\b", raw)
        return letter.group(1) if letter else None
    return raw.strip().rstrip(".")


def is_correct(extracted, gold, answer_type):
    if extracted is None:
        return False
    if answer_type == "multipleChoice":
        return extracted.upper() == gold.strip().upper()
    return extracted.strip().lower() == gold.strip().lower()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="outputs/merged-hle-smoke")
    parser.add_argument(
        "--parquet",
        default="data/hle/datasets/cais--hle/snapshots/master/data/test-00000-of-00001.parquet",
    )
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    args = parser.parse_args()

    questions = load_questions(args.parquet, args.n)

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype="auto", device_map="auto", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    correct_count = 0
    for q in questions:
        response = generate(model, tokenizer, q["question"], args.max_new_tokens)
        extracted = extract_answer(response, q["answer_type"])
        correct = is_correct(extracted, q["answer"], q["answer_type"])
        correct_count += correct
        print("=" * 60)
        print(f"Q [{q['answer_type']}]: {q['question'][:120]}...")
        print(f"Gold: {q['answer']} | Extracted: {extracted} | {'✓' if correct else '✗'}")
        print(f"Response tail: ...{response[-200:]}")

    accuracy = round(100 * correct_count / len(questions), 2)
    print("=" * 60)
    print(f"*** Metrics ***")
    print(f"Accuracy: {accuracy}% +/- n/a | n = {len(questions)}")


if __name__ == "__main__":
    main()
