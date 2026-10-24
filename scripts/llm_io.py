"""模型加载、生成解码与 jsonl IO。唯一允许 import torch 的公共模块。

--model 参数语义（所有脚本一致）：
- 传本地路径 → 直接用
- 传 ModelScope repo id（如 Qwen/Qwen3.6-35B-A3B）→ 自动 snapshot_download（云上）
"""

import json
from pathlib import Path

import torch

from io_utils import append_jsonl, read_jsonl  # noqa: F401  (re-export 兼容)

DEFAULT_MODEL = "models/models/qwen--Qwen2.5-3B-Instruct/snapshots/master"

# 解码时保留特殊 token 文本（Qwen3 系的 <think> 可能是特殊 token，
# skip_special_tokens=True 会把它们抹掉导致 reasoning 丢失），
# 只手动清除聊天结构相关的标记。
_DECODE_STRIP = ("<|im_start|>", "<|im_end|>", "<|endoftext|>")


def clean_decoded(text: str) -> str:
    for tok in _DECODE_STRIP:
        text = text.replace(tok, "")
    return text.strip()


# ---------------------------------------------------------------- model


def resolve_model_path(model: str) -> str:
    if Path(model).exists():
        return model
    print(f"Local path not found, downloading from ModelScope: {model}")
    from modelscope import snapshot_download

    return snapshot_download(model)


def load_model_and_tokenizer(model: str, device_map: str = "auto"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = resolve_model_path(model)
    print(f"Loading model: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype="auto", device_map=device_map, trust_remote_code=True
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return model, tokenizer


@torch.no_grad()
def generate(messages, model, tokenizer, max_new_tokens=768, temperature=0.7, seed=42):
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    do_sample = temperature > 0
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=0.95 if do_sample else None,
        pad_token_id=tokenizer.eos_token_id,
    )
    raw = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                           skip_special_tokens=False)
    return clean_decoded(raw)
