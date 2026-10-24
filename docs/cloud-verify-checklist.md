# 上云首验清单（租机后 30–60 分钟）

> 目标：租机烧钱时间内完成全部关键验证，每步有成功标准和备选。
> 量化脚本已本地 3B 冒烟通过（`scripts/cloud/quantize_4bit.py`）。

## 0. 环境确认（2 分钟）

```bash
nvidia-smi                    # 确认 80G 卡
df -h /root/autodl-tmp        # 确认数据盘 ≥ 50GB
ls /root/autodl-fs            # 确认网盘挂载（模型放这不占配额）
```

**失败备选**：无网盘 → 用 `python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen3.6-35B-A3B', cache_dir='/root/autodl-fs')"` 云内直下（云带宽快）。

## 0.5 蒸馏输出格式首验（第 0 步，5 分钟，不可跳过）

Qwen3.6 是原生 thinking 模型，其输出格式/特殊 token 与本机 3B 假设可能不一致——
解析器接不住会整批丢 reasoning。**放量蒸馏前必须先跑这 5 条并人工看输出**：

```bash
python - <<'EOF'
import sys; sys.path.insert(0, 'scripts')
from pipeline_common import (
    SYSTEM_PROMPT_QA, generate, load_model_and_tokenizer, parse_qa_output,
)
model, tok = load_model_and_tokenizer('Qwen/Qwen3.6-35B-A3B')
msgs = [{"role": "system", "content": SYSTEM_PROMPT_QA},
        {"role": "user", "content": "What is 17 + 25?"}]
raw = generate(msgs, model, tok, max_new_tokens=512)
print("===== RAW ====="); print(raw)
parsed = parse_qa_output(raw)
print("===== PARSED =====")
print("reasoning_len:", len(parsed["reasoning"]), "| answer:", parsed["answer"])
EOF
```

- **成功**：RAW 全文打印出来，PARSED 的 reasoning_len > 0 且 answer 非空
- **看什么**：RAW 里 thinking 用什么标记（裸文本 / `<think>` / 特殊 token）、ANSWER 行是否照做
- **失败处理**：reasoning_len = 0 → 看 RAW 实际格式，改 `parse_qa_output`/`_DECODE_STRIP` 后重验；确认前不得启动全量蒸馏

## 1. bnb 4-bit 加载 35B（核心风险，5 分钟）

```bash
python -c "
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
m = AutoModelForCausalLM.from_pretrained('/root/autodl-fs/Qwen3.6-35B-A3B',
     quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype='bfloat16'),
     device_map='auto', trust_remote_code=True)
print('loaded OK')
"
```

- **成功**：无报错，显存 ~20GB
- **失败备选 A**：报 GatedDeltaNet 相关错 → 去掉 linear_attn 层量化（自定义 quantize 跳层逻辑）
- **失败备选 B**：完全不兼容 → 放弃 4bit，等可扩容盘机器跑 fp16（70GB）；或改试 GPTQ 量化路径

## 2. 量化保存（10–30 分钟，网盘读取速度决定）

```bash
python scripts/cloud/quantize_4bit.py \
  --src /root/autodl-fs/Qwen3.6-35B-A3B \
  --dst /root/autodl-tmp/Qwen3.6-35B-A3B-4bit --verify-gen
```

- **成功**：磁盘 ~20–23GB；生成测试输出正常中文且带 thinking
- **失败**：基本无新风险（3B 已验证管道）

## 3. vLLM 服务验证（baseline 推理路径，10 分钟）

```bash
vllm serve /root/autodl-tmp/Qwen3.6-35B-A3B-4bit \
  --served-model-name qwen35b --quantization bitsandbytes \
  --max-model-len 32768 --gpu-memory-utilization 0.92
```

- **成功**：服务起，OpenAI 接口 `/v1/chat/completions` 返回正常
- **失败备选 A**：vllm 不认识 qwen3_5_moe 架构 → 升级 vllm 到最新；仍不行 →  transformers generate 直跑（只冒烟 10 题级，全量评测再想办法）
- **失败备选 B**：bnb 在 vllm 下不支持该架构 → fp16 直跑（80G 装 67G 权重，压 max-model-len/KV cache）或转 AWQ

## 4. 训练冒烟（10 分钟）

用 `scripts/train_lora.py` 改指向 4bit 模型 + 50 条 mock 数据跑 10 step：

- **成功**：loss 下降、显存 < 60GB、速度可估（推 W2 全量时长）
- **失败**：OOM → 降 batch/开 gradient checkpointing；loss 爆炸 → 降 lr

## 全部通过后的动作顺序

1. 挂监控继续盯第二张卡（双卡并行：一张跑 baseline 一张跑训练）
2. 先跑 **Base 四榜 baseline**（W1 出口硬指标）
3. baseline 落档后再进 W2 数据构建
