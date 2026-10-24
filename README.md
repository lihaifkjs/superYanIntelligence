# superYanIntelligence

基于 Qwen3.6-35B-A3B 的多专家训练与 Model Merging 合版——针对 SWE-bench Pro v2-hard / HLE / ASI-Bench / Terminal-Bench 四榜各训一个 QLoRA 专家，用 Task Arithmetic / TIES 合版为单一模型，目标四榜宏平均较 Base 提升 3 个百分点。

## 仓库地图

| 路径 | 内容 |
|---|---|
| [`task.md`](task.md) | 赛题原文与验收硬要求 |
| [`PRD.md`](PRD.md) | 实施方案：关键决策、数据/训练/合版/评测方案、里程碑、风险 |
| [`AGENTS.md`](AGENTS.md) | Agent 工作规约与文档地图 |
| [`scripts/`](scripts/README.md) | 全流程脚本：种子→蒸馏→过滤→训练→合并→评测 |
| [`scripts/cloud/`](scripts/cloud/run_all.sh) | 云上 runbook：租机后 `setup_env.sh` → `run_all.sh` |
| [`docs/w1-progress.md`](docs/w1-progress.md) | W1：3B pipeline 打通记录 |
| [`docs/w2-progress.md`](docs/w2-progress.md) | W2：本机 3B 全链路预演 + 质量阀加固 + 评审修复记录 |
| [`docs/cloud-verify-checklist.md`](docs/cloud-verify-checklist.md) | 上云首验清单（含蒸馏格式第 0 步） |
| [`docs/spike-35b-arch.md`](docs/spike-35b-arch.md) | 35B 混合架构 LoRA 挂载 spike 结论 |
| [`docs/gpu-rental-notes.md`](docs/gpu-rental-notes.md) | 云 GPU 租用踩坑 |
| [`tests/run_tests.py`](tests/run_tests.py) | 核心逻辑回归测试（`python tests/run_tests.py`） |

## 快速开始（本机 smoke，4070Ti 12G + 3B）

```bash
pip install -r scripts/requirements.txt
python scripts/prepare_seeds.py --expert all --n_generated 50
python scripts/distill.py --mode qa    --seeds data/seeds/hle.jsonl --limit 8
python scripts/distill.py --mode agent --seeds data/seeds/terminal.jsonl --limit 3
python scripts/build_sft.py --traj data/raw_traj/hle.jsonl
python scripts/build_sft.py --traj data/raw_traj/terminal.jsonl
python scripts/train_lora.py --model models/models/qwen--Qwen2.5-3B-Instruct/snapshots/master \
    --data data/sft/terminal.jsonl --output outputs/lora-terminal --tools
python tests/run_tests.py
```

## 上云（A100/H100 80G + 35B）

1. 按 [`docs/cloud-verify-checklist.md`](docs/cloud-verify-checklist.md) 完成首验（**第 0.5 步：蒸馏格式验证不可跳过**）
2. `bash scripts/cloud/setup_env.sh`
3. `bash scripts/cloud/run_all.sh`（CONFIG 区已默认 35B；每步 `--skip-xxx` 可控）

## 当前状态

- ✅ W1/W2 机制链路：本机 3B 全链路验证通过（含 TA/TIES 合版加载冒烟）
- ✅ 质量阀：sympy 判分 / think 解析加固 / Terminal 模板化 / Docker 沙箱 / 契约修复
- ⏳ 35B 真实训练与四榜评测：待上云（runbook 就绪）
- 详细结论与遗留项见 [`docs/w2-progress.md`](docs/w2-progress.md)
