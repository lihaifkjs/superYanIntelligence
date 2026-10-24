import argparse
from pathlib import Path

from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

from llm_io import DEFAULT_MODEL, resolve_model_path
from prompts import TOOLS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="本地路径或 ModelScope repo id（自动下载）")
    parser.add_argument("--data", default="data/sft/hle.jsonl")
    parser.add_argument("--output", default="outputs/lora-hle")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--r", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--target_modules", default=None,
                        help="逗号分隔，默认 q/k/v/o/gate/up/down_proj；35B 混合架构按需扩")
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="QLoRA 模式；默认 bf16 全精度（35B MoE experts 不被 bnb 量化，4bit 无收益）")
    parser.add_argument("--tools", action="store_true",
                        help="数据含 tool_calls 时开启，chat template 注入工具 schema")
    return parser.parse_args()


def main():
    args = parse_args()

    kwargs = dict(device_map="auto", trust_remote_code=True)
    if args.load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="bfloat16",
        )
    else:
        kwargs["torch_dtype"] = "bfloat16"

    print(f"Loading model: {args.model} (4bit={args.load_in_4bit})", flush=True)
    model_path = resolve_model_path(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        **kwargs,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    else:
        # bf16 全精度：kbit 预处理会把全部参数 cast fp32（35B 直接 OOM），改为手动冻结
        for p in model.parameters():
            p.requires_grad = False
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    target_modules = (
        args.target_modules.split(",") if args.target_modules
        else ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    lora_config = LoraConfig(
        r=args.r,
        lora_alpha=args.alpha,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_dataset("json", data_files=args.data, split="train")

    def formatting_func(example):
        kwargs = {"tools": TOOLS} if args.tools else {}
        return tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False, **kwargs
        )

    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
        optim="paged_adamw_8bit",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
        formatting_func=formatting_func,
    )

    print("Starting training...")
    trainer.train()

    output_path = Path(args.output) / "final"
    trainer.save_model(output_path)
    print(f"LoRA adapter saved to {output_path}")


if __name__ == "__main__":
    main()
