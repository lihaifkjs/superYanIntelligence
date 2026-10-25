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

### 2026-10-25（上午，无卡模式）

13. **833机 Docker 可行性测试**（`ssh -p 56630 root@connect.westc.seetacloud.com`，凭据已录入 credentials.md）— 目的：验证 AutoDL 容器内能否跑 Docker（Phase 5 CPU 机候选）。结果：❌ **不可行，勿再试**。
    - 容器 capabilities 缺 `CAP_SYS_ADMIN` / `CAP_NET_ADMIN` / `CAP_SYS_RESOURCE`，`unshare -Urn` 与 `-Ur` 均 Operation not permitted（user namespace 被禁）
    - dockerd `--bridge=none --iptables=false --storage-driver=vfs` 可起 daemon，但 **vfs 注册层报 `unshare: operation not permitted`**，拉取镜像无法落盘
    - overlay2 在 xfs 数据盘上 `failed to mount overlay: operation not permitted`
    - 结论：**AutoDL（至少西北B区该机）容器内 Docker 全路线被封死**，rootless/podman 同样无 userns 不可用。Phase 5 Docker 机需改道：本机 Docker Desktop 或外部 VPS
    - 附带资产：docker.io 29.1.3 已装在该机（无害残留）；docker.io 官方仓库被墙，docker.m.daocloud.io 镜像站可通
    - **后续：该机已于 2026-10-25 由用户删除**（Docker 走不通 + 改用本机 Docker Desktop），凭据已从 credentials.md 移除

### 2026-10-25（下午，GPU 模式，W1 推进）

14. **vLLM 反复假死根因定位** — 现象：READY 后处理某些请求时 EngineCore 致命崩溃（`FileNotFoundError: 'ninja'`），API 层变僵尸（/v1/models 通、chat/completions 空响应）。原因：**机器没装 ninja**，torch 内核编译在新 batch shape 触发时找不到它。处置：`pip install ninja`（1.13.2），重启后验证。⚠️ 镜像里默认无 ninja，**重装环境/换机必装**，写进环境检查。
15. **W1 下午进展** — HLE 200 题 baseline 出分：**0.5%（n=200，95% 死于不收束不交出 Answer）**，结果本地 `outputs/base-eval/hle_pred_{10,200}.json`；Terminal-Bench 打通（hello-world PASS，10 题子集跑分中）；ASI-Bench 打通（opencode→vLLM→本地打分，0 分=base 工具调用弱）；SWE-bench Pro v2 攻坚中。
16. **工具调用格式结论** — Qwen3.6-35B base 原生工具格式是 **Qwen XML 风格**（`<function=名><parameter=参数>`），vLLM 需 `--tool-call-parser qwen3_xml`（hermes 解析不出）；探针验证可结构化。W2 训练数据格式须对齐。
17. **隧道运维教训** — Windows 上 SO_REUSEADDR 允许双绑：隧道自动重连会产生「僵尸监听+新监听」并存，一半连接空响应；排障先 `netstat | grep 8000` 确认单实例。隧道脚本已加 SSH keepalive。

### 2026-10-25（晚，GPU 模式，W2 smoke）

18. **W2 云上 smoke 首轮（结论：机制全通，格式协议是瓶颈，放量前必须做吞吐优化）**
    - Phase 0 格式验证 ✅：35B 蒸馏输出 think 标签 + `ANSWER:` 行照做（人工目检 RAW 通过）
    - **坑**：git 仓库 clone 带下 3B 预演时代的 raw_traj（commit `ecb9ef2` 误提交数据）→ 断点续跑直接跳过 = 差点混入 3B 轨迹。**云机 clone 后必先清 `data/raw_traj/`**（已备份至 `/root/autodl-tmp/raw_traj_3b_backup/`）
    - HLE qa 冒烟 30 题：通过率 10% → 46.7% → **50.0%**（三轮修复：①`parse_qa_output` 改全文扫描+跳过 `<final answer>` 占位符回声——模型把答案写进 think 计划区，旧逻辑只扫 think 外；②`--max_new_tokens` 768→3072——35B 规划冗长，768 在 ANSWER 行前截断（3B 默认值不适用）；③`normalize_answer` 补纯数字括号对折叠 `-(1)/(2)`≡`-1/2` 和一元正号 `+infty`≡`infty`）。三 commit：`51dcaac`/`7b4d58e`
    - Terminal agent 冒烟 3/10（主动停）：沙箱 **Docker 不可用降级 local**（PRD v0.5 已接受 Git Bash 级沙箱）；机制健康（T02 两轮完美走通规划→执行→验证→交卷），失败主因 = **模型对「只输出 JSON」协议服从不稳**，JSON 解析失败空转烧光 12 轮。通过率 1/3，n 太小不下结论
    - **吞吐实测（放量优化的依据）**：qa batch 4 直跑 ~40s/题（3000 题≈33h≈¥185）；Terminal 串行 ~22 分/任务（330 任务≈120h≈¥670）——裸跑超 PRD 预算 10 倍。优化方向：qa batch→16、Terminal 多沙箱并行×8、max_turns 12→8、（可选）qa 切 vLLM 生成。优化后预估 ¥100-150
    - 产物已拉回本地 `outputs/w2-smoke/`；云机 raw_traj/sft 数据在 `/root/syi/data/` 完好

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
