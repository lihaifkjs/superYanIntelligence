# W1 进展记录

> 日期：2026-10-23  
> 状态：本地 pipeline smoke test 完成，35B Base baseline 受限于算力待推进

## 已完成

### 1. 本地端到端 pipeline 跑通

用 `Qwen/Qwen2.5-3B-Instruct` 在本地 4070Ti 12G 上验证了完整链路：

```
data → QLoRA train → LoRA merge → eval
```

| 步骤 | 结果 |
|---|---|
| HLE mock 数据 | 生成 50 条（数学 + 逻辑），带 `<think>` 推理过程 |
| QLoRA 训练 | loss 1.546 → 0.327，mean token accuracy 0.9624 |
| LoRA 合并 | 成功生成 `outputs/merged-hle-smoke` |
| 推理评测 | 5 道测试题全部正确 |

### 2. 新增脚本

- `scripts/prepare_hle_smoke.py` — HLE mock 数据生成
- `scripts/train_lora.py` — QLoRA 训练
- `scripts/merge_lora.py` — LoRA 合并到 base
- `scripts/eval_simple.py` — 简单推理评测
- `scripts/requirements.txt` / `scripts/README.md`

## 关键决策

- **本地先跑 3B smoke test**，不上 35B，验证 pipeline 后再迁移
- **HLE 方向先跑通**，因为数据格式最简单（问答 + thinking）
- **使用 ModelScope 下载模型**：HuggingFace 直接访问 SSL 握手失败，`hf-mirror.com` 同样失败；ModelScope 稳定

## 踩坑与修复

| 问题 | 原因 | 修复 |
|---|---|---|
| HuggingFace 下载 SSL 失败 | 网络环境对 `huggingface.co` / `hf-mirror.com` 握手异常 | 改用 ModelScope `snapshot_download` |
| `DataCollatorForCompletionOnlyLM` 导入失败 | `trl==1.16.0` 已移除该 collator | 改用 `formatting_func` + SFTTrainer 默认处理 |
| `SFTTrainer` 不接受 `tokenizer` / `max_seq_length` | `trl==1.16.0` API 变更 | `tokenizer` → `processing_class`；移除 `max_seq_length` |
| 评测输出 UnicodeEncodeError | Windows 控制台默认 GBK | 运行 eval 时加 `PYTHONIOENCODING=utf-8` |

## 待完成（W1 出口标准缺口）

| 里程碑项 | 状态 | 阻塞 |
|---|---|---|
| LoRA merge smoke test 通过 | ⚠️ 单专家 merge 已通，多专家 merge 未验证 | 本地可继续 |
| 四榜 harness 链路打通 | ⚠️ **HLE 已通**（见下），Terminal/ASI/SWE 未开始 | 需 Docker，移到云上无卡实例跑 |
| Base 四榜分数落档 | ❌ 未开始 | **必须 35B 真机**，AutoDL/优云智算无 80G 库存，阿里云 ¥38/时太贵 |

## HLE harness 打通（2026-10-23）

- 官方仓库：`centerforaisafety/hle`（GitHub 可直连），已 clone 到 `external/hle`
- 数据集：`cais/hle` 在 HF 是 gated 且 hf-mirror 403；改从 **ModelScope 镜像**下载（`data/hle/datasets/caiss--hle/`），2500 题中 2158 题纯文本
- 本地 smoke：`scripts/hle/run_hle_smoke.py`
  - 用官方 SYSTEM_PROMPT（Explanation/Answer/Confidence 格式）
  - 规则判分（multipleChoice 提取字母 / exactMatch 归一化比对）
  - 3B merged 模型跑 3 题，输出 Accuracy 0.0%（符合预期，HLE 难度下 3B 近乎不可能得分）
- 官方 LLM judge（`run_judge_results.py`，默认 o3-mini）留到云上，配合强 judge 模型跑全量

## 下一步选项

1. **补齐 W1 剩余项（推荐）**
   - 本地搭四榜 harness 的 smoke test，用 3B 模型各跑 1–3 题验证能出分
   - 继续蹲便宜云 GPU 跑 35B Base baseline

2. **提前进入 W2**
   - 训第二个专家（如 Terminal），验证多专家 Task Arithmetic / TIES 合并

3. **直接上阿里云 DSW**
   - A100 80G ¥38.21/时，先开 1 小时把 35B Base baseline 跑了
