# 本地 Smoke Test

用 `Qwen/Qwen2.5-3B-Instruct` 在本地 4070Ti 12G 上跑通「数据 → QLoRA 训练 → 合并 → 评测」的完整链路。

## 环境准备

```bash
cd scripts
pip install -r requirements.txt
```

## 运行步骤

### 1. 生成 mock 数据

```bash
python prepare_hle_smoke.py --output ../data/hle_smoke.jsonl
```

### 2. QLoRA 训练

```bash
python train_lora.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --data ../data/hle_smoke.jsonl \
  --output ../outputs/lora-hle-smoke \
  --epochs 1 \
  --batch_size 1 \
  --gradient_accumulation_steps 4
```

### 3. 合并 LoRA

```bash
python merge_lora.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --adapter ../outputs/lora-hle-smoke/final \
  --output ../outputs/merged-hle-smoke
```

### 4. 简单评测

```bash
python eval_simple.py --model ../outputs/merged-hle-smoke
```

## 预期结果

- 训练能在 4070Ti 12G 上跑完，显存占用约 8–10G
- 合并后的模型能加载并回答 5 道测试题
- 输出中应包含 `<think>...</think>` 形式的推理过程
