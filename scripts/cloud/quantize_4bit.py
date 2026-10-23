"""fp16/bf16 模型 → bnb 4-bit 量化并保存（50GB 数据盘方案的核心工具）。

本地冒烟（3B）:
    python scripts/cloud/quantize_4bit.py \
        --src models/models/qwen--Qwen2.5-3B-Instruct/snapshots/master \
        --dst outputs/quant-3b-smoke --verify-gen

云上 35B（源放网盘，不占数据盘）:
    python scripts/cloud/quantize_4bit.py \
        --src /root/autodl-fs/Qwen3.6-35B-A3B \
        --dst /root/autodl-tmp/Qwen3.6-35B-A3B-4bit --verify-gen
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_model(src: str):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="bfloat16",
    )
    kwargs = dict(quantization_config=bnb_config, device_map="auto", trust_remote_code=True)
    try:
        return AutoModelForCausalLM.from_pretrained(src, **kwargs)
    except (ValueError, KeyError, AttributeError):
        # 多模态条件生成类（如 Qwen3_5MoeForConditionalGeneration）不在 CausalLM 映射里
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText.from_pretrained(src, **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="源模型路径（本地目录或网盘挂载点）")
    parser.add_argument("--dst", required=True, help="4-bit 输出目录")
    parser.add_argument("--verify-gen", action="store_true", help="保存后重载并做短生成校验")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    print(f"loading {args.src} with 4-bit quantization ...")
    model = load_model(args.src)
    tokenizer = AutoTokenizer.from_pretrained(args.src, trust_remote_code=True)

    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    print(f"saving 4-bit model to {dst} ...")
    model.save_pretrained(dst)
    tokenizer.save_pretrained(dst)

    size_gb = sum(f.stat().st_size for f in dst.rglob("*") if f.is_file()) / 1e9
    print(f"saved size: {size_gb:.2f} GB")

    if args.verify_gen:
        del model
        torch.cuda.empty_cache()
        print("reloading from saved 4-bit model ...")
        m4 = load_model(str(dst))
        msgs = [{"role": "user", "content": "用一句话解释什么是量子纠缠。"}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(m4.device)
        with torch.no_grad():
            out = m4.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        print("=== 4-bit gen output ===")
        print(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))


if __name__ == "__main__":
    main()
