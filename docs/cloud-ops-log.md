# 云机操作台账（Ops Log）

> 用途：记录 003 机（A800专区，98fc43be4e）上执行过的所有操作，**避免重复劳动**。
> 规则：每次在云机执行操作后，在「操作记录」末尾追加一条（时间/操作/命令/结果/是否需重做）。
> 关联文档：本文即状态档案（机器/决策/失效清单），往前做看 `cloud-w1-runbook.md`，W2 看 `cloud-w2-runbook.md`。

## 机器档案

| 项 | 值 |
|---|---|
| 实例 | AutoDL A800专区 003机（容器 98fc43be4e-7d7ab2fb） |
| 规格 | 1×A800-80GB / 14核 / 120GB 内存 / 系统盘30G + 数据盘50G（固定不可扩） |
| SSH | 见 credentials.md；直连 IP `36.137.238.251` 端口 32375（重启后可能变，以控制台为准） |
| 计费 | 按量 ¥5.59/时（无卡模式 ~¥0.1/时，shm 仅 2GB） |

## 关键决策（2026-10-24，源自已归档的 session-state，长期有效）

- ~~4bit 量化省磁盘~~ → **废弃**：routed experts（32B/35B 参数）是批量参数非 Linear，bnb 量化不到，省 1.3GB 无意义
- **改为 bf16 全精度训练**：显存 67GB 模型 + 余量 OK；只存 LoRA adapter（42MB/个），50GB 盘够用
- fp16 模型源全程保留在磁盘，训练/评测直接从 `/root/autodl-tmp/q36` 加载
- `prepare_model_for_kbit_training` 在 bf16 模式禁用（会把全参数 cast fp32 导致 OOM），train_lora.py 已改手动冻结+梯度检查点，**勿回退**
- 待验证遗留：bnb 4bit 在 GatedDeltaNet 的兼容性已随量化路线一并放弃，无需再验

## 重启失效清单（每次开机必查）

| 资产 | 位置 | 关机后 |
|---|---|---|
| 模型分片（~20个,~50GB） | /dev/shm/q36_shards | **清空，需重下**（跑 w1_refill_model.sh，约30分钟） |
| 模型分片（6个,~17GB） | /root/autodl-tmp/q36_shards | 保留（≤15天） |
| Python 环境（vllm/torch2.13/训练栈） | /root/miniconda3（系统盘） | 保留 |
| HLE parquet、脚本、日志 | /root/autodl-tmp | 保留 |

## 操作记录

### 2026-10-24（上午，GPU 模式）

1. **环境检查** — nvidia-smi/df/python 版本确认。结果：A800-80G、tmp 50G、shm 60G（有卡模式）、Py3.12。✅ 无需重做
2. **模型下载 67GB** — `download_model.py`（ModelScope，3路并行 ~30MB/s），分片拆放 shm 51G + tmp 17G，统一视图 `/root/autodl-tmp/q36`。结果：26/26 DONE。⚠️ 关机丢 shm 部分，重启用 w1_refill_model.sh 自动补
3. **装训练栈** — pip（阿里云源）：transformers 5.20.2 / peft 0.21.2 / trl 1.16.0 / bitsandbytes 0.50.2 / accelerate 1.15.0 / datasets 5.1.0。✅ 无需重做（系统盘持久）
4. **bnb 4bit 加载+量化试验**（`load_and_quant.py`）— 结果：加载 14s ✅、thinking 生成 ✅、**保存失败 No space**。结论：MoE routed experts（32B 参数）不被 bnb 量化，saved 体积 ~66GB > 盘余量 → **废弃量化路线，定案 bf16 全精度训练**。✅ 结论有效，实验勿重做
5. **训练冒烟 #1** — `train_lora.py --model /root/autodl-tmp/q36`（12 种 target modules）。结果：**OOM**（prepare_model_for_kbit_training 把全参数 cast fp32）。→ 已修脚本（bf16 模式改手动冻结+梯度检查点）。❌ 已修复，勿复现
6. **训练冒烟 #2**（修后）— 结果：✅ 91s/13 步，loss 0.72，aux_loss 8.32，adapter 存 `/root/autodl-tmp/lora-smoke/final`。显存峰值 <79G。**35B 全链路首次打通**。✅ 已验证
7. **HLE 数据下载** — ModelScope 镜像 `cais/hle` → `/root/autodl-tmp/hle/datasets/caiss--hle/.../test-00000-of-00001.parquet`（274MB）。✅ 无需重做
8. **vLLM 安装 #1**（vllm==0.32.0）— 中断于 torch 2.13 下载阶段（用户关机）。❌ 已被 #11 取代

### 2026-10-24（下午，无卡模式）

9. **开机体检** — 确认无卡模式：无 GPU、shm 仅 2GB、tmp 分片幸存 6 个、训练环境完好。✅ 结论：补模型必须 GPU 模式
10. **（尝试装 causal-conv1d，弃）** — 源码编译过慢，放弃。✅ 结论：勿再试，如需提速试 flash-linear-attention
11. **vLLM 安装 #2** — `pip install vllm==0.32.0`（阿里云源，依赖链 torch2.13+cu13 ~3GB，总耗时约 2.5h 含降速）。结果：✅ vllm 0.32.0 + torch 2.13.0+cu130，import 正常；训练栈 import 验证 OK（torch 2.13 下兼容）。**注意**：`vllm --version` 报 device 错误是无卡模式假象。
12. **HLE baseline 脚本上传** — `/root/hle_baseline.py`（打 vLLM OpenAI 端点 + 规则判分）。✅ 就位待跑

### 当前状态（2026-10-24 晚，已关机待命）

- ✅ 已完成：下载→加载→生成→训练冒烟→vLLM 环境→HLE 数据+脚本
- ⬜ 未做（下个 agent 按 `cloud-w1-runbook.md` 执行）：补模型 → vLLM 服务 → HLE 冒烟/200题 → 三榜 harness → 落档
- 本日花费：GPU 约 1.5h ≈ ¥8.4 + 无卡约 0.3

---

## 追加模板（后续 agent 照此追加）

```
### YYYY-MM-DD（时段，模式）

N. **操作名** — 命令/方式。结果：✅/❌ + 关键数据。是否需重做：✅/⚠️/❌ + 说明
```
