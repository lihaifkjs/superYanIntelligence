import argparse
from pathlib import Path

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--adapter", default="outputs/lora-hle-smoke/final")
    parser.add_argument("--output", default="outputs/merged-hle-smoke")
    args = parser.parse_args()

    print(f"Loading base model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print(f"Loading adapter: {args.adapter}")
    model = PeftModel.from_pretrained(model, args.adapter)

    print("Merging and unloading...")
    model = model.merge_and_unload()

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"Merged model saved to {output_path}")


if __name__ == "__main__":
    main()
