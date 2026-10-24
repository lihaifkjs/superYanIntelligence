# 流水线脚本

全部脚本 `--model` 参数化：本机用 3B 本地路径，云上换 35B（repo id 或路径），逻辑不变。

本机 3B 路径：`models/models/qwen--Qwen2.5-3B-Instruct/snapshots/master`

## 环境准备

```bash
pip install -r requirements.txt   # 含 modelscope / safetensors
```

## 主流程（W2 流水线）

```bash
# 1. 种子数据：Terminal 内置 30 任务 + HLE 数学抽 300 题
python prepare_seeds.py --expert all

# 2. 自蒸馏（teacher=--model）
python distill.py --mode qa    --seeds data/seeds/hle.jsonl      # HLE 批量问答
python distill.py --mode agent --seeds data/seeds/terminal.jsonl # Terminal 沙箱 agent

# 3. 过滤 + SFT 数据
python build_sft.py --traj data/raw_traj/hle.jsonl
python build_sft.py --traj data/raw_traj/terminal.jsonl
python prepare_swe_asi.py --expert both --n 20   # SWE/ASI 开源轨迹格式适配

# 4. 训练（Terminal 轨迹加 --tools）
python train_lora.py --model <M> --data data/sft/hle.jsonl      --output outputs/lora-hle
python train_lora.py --model <M> --data data/sft/terminal.jsonl --output outputs/lora-terminal --tools

# 5. 多专家合并（adapter 直接参与，TA / TIES）
python merge_experts.py --base <M> --experts outputs/lora-{hle,terminal,swe,asi}/final \
    --method task_arithmetic --output outputs/merged-ta
python merge_experts.py --base <M> --experts outputs/lora-{hle,terminal,swe,asi}/final \
    --method ties --density 0.2 --output outputs/merged-ties
python merge_experts.py --self-test   # 合并数学逻辑单测

# 6. 污染检测
python check_contamination.py --train data/sft/hle.jsonl --eval <评测集 parquet/jsonl>

# 7. 冒烟评测
python eval_simple.py --model outputs/merged-ta
```

## 云上

见 `../docs/cloud-w2-runbook.md`（W2 操作 Runbook）与 `cloud/w2_pipeline.sh`（一键流水线，
`SCALE=smoke/full`，每步 `--skip-xxx`）；`cloud/setup_env.sh` 为环境准备。
进展与验证结论见 `../docs/w2-progress.md`。

## 目录约定

- `data/seeds/` 种子任务，`data/raw_traj/` 原始轨迹（断点续跑），`data/sft/` 训练数据
- `data/swe_asi_sample/` SWE/ASI 格式样本
- `outputs/sandbox/` Terminal 沙箱工作目录
- `outputs/lora-*` adapter，`outputs/merged-*` 合并权重（含 merge_config.json 证据链）

## W1 遗留脚本

`prepare_hle_smoke.py` / `merge_lora.py`（单专家合并）为 W1 mock 链路产物，已被上方流程覆盖，保留作参考。
