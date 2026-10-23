import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer


QUESTIONS = [
    "If a + b = 15 and a × b = 56, what is a² + b²?",
    "A car travels 80 km in 1 hour. How far does it travel in 4 hours at the same speed?",
    "What is the next number in the sequence: 3, 6, 12, 24, ...?",
    "Tom is taller than Jerry, and Jerry is taller than Mike. Who is the tallest?",
    "If x - 5 = 12, what is x?",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="outputs/merged-hle-smoke")
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-3B-Instruct")
    args = parser.parse_args()

    model_path = args.model
    print(f"Loading model: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    for q in QUESTIONS:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": q},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\nQ: {q}")
        print(f"A: {response}")


if __name__ == "__main__":
    main()
