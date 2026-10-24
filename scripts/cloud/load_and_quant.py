"""云机执行：bnb 4bit 加载验证 + 量化保存（一条龙，日志 flush 便于轮询）。"""

import time

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

SRC = "/root/autodl-tmp/q36"
DST = "/root/autodl-tmp/q36-4bit"

t0 = time.time()
print("step1: loading 4bit ...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    SRC,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="bfloat16",
    ),
    device_map="auto",
    trust_remote_code=True,
)
print(f"step1 OK: loaded in {time.time()-t0:.0f}s", flush=True)
print(f"mem allocated: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)
print(f"mem reserved:  {torch.cuda.memory_reserved()/1e9:.1f} GB", flush=True)

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(SRC, trust_remote_code=True)
msgs = [{"role": "user", "content": "用一句话解释什么是量子纠缠。"}]
text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
inputs = tok(text, return_tensors="pt").to(model.device)
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
print("=== gen output ===", flush=True)
print(tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True), flush=True)

print("step2: saving 4bit ...", flush=True)
t1 = time.time()
model.save_pretrained(DST)
tok.save_pretrained(DST)
print(f"step2 OK: saved in {time.time()-t1:.0f}s -> {DST}", flush=True)

import pathlib

size = sum(f.stat().st_size for f in pathlib.Path(DST).rglob("*") if f.is_file()) / 1e9
print(f"saved size: {size:.1f} GB", flush=True)
print("ALL DONE", flush=True)
