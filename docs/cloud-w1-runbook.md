# W1 云机操作 Runbook（交接文档）

> 面向：独立云操作 agent。目标：完成 W1 剩余云侧工作，产出 Base 四榜分数落档。
> 前置阅读：`docs/cloud-ops-log.md`（当前机器状态/关键决策）、`docs/spike-35b-arch.md`（架构结论）。
> 所有脚本在 git 仓库 `scripts/cloud/`，同名文件也在云机 `/root/scripts/` 或 `/root/`（若丢失按本文「输入」节补传）。

## 0. 输入清单（开始前必须齐备）

| 输入 | 说明 |
|---|---|
| GPU 模式开机的 A800-80GB 实例 | 按量计费，镜像 PyTorch2.8+CUDA12.8（已被升级，不影响） |
| SSH 凭据 | `credentials.md` 的「AutoDL 实例 SSH」节；**注意关机重开 SSH 可能变化，以控制台为准** |
| 网络可达 | modelscope.cn（国内直连）、mirrors.aliyun.com（pip）；huggingface.co 不保证可达，不要依赖 |
| 云机已有资产 | Python3.12(/root/miniconda3) + vllm 0.32.0 + torch 2.13 + 训练栈；模型 6/26 分片在 tmp；HLE parquet；本文脚本 |
| git 仓库 | github.com/lihaifkjs/superYanIntelligence（脚本最新版在 `scripts/cloud/`） |

## 1. 范围

### 做什么（W1 云侧剩余）

1. 环境体检 → 2. 补模型到 shm → 3. vLLM 起服务 → 4. HLE baseline 出分（冒烟 10 题 → 200 题子集）→ 5. 训练冒烟复验（torch 2.13 后可选）→ 6. Terminal/ASI/SWE 三榜 harness 打通（各 1-3 题）→ 7. 分数落档 `docs/` 并 git push

### 不做什么（超出 W1）

- ❌ 训练任何专家（W2/W3 的活），不跑自蒸馏数据生成
- ❌ 合并实验（Task Arithmetic / TIES）
- ❌ 全量评测（2500 题×4 榜是 W4；W1 只需每榜 10 题级子集 + HLE 200 题子集，PRD 成本控制条款）
- ❌ 上传 HuggingFace（token 未就绪，等主 agent）
- ❌ 删改 `/root/autodl-tmp/q36` 软链视图、不动 `/root/autodl-tmp/q36_shards`
- ❌ 给 router 加 LoRA、不冻结以外的 router 操作
- ❌ 装 `causal-conv1d`（源码编译又慢又易失败）
- ❌ 在**无卡模式**下补模型（shm 仅 2GB）

## 2. 执行步骤

### Phase 0：开机体检（2 分钟）

```bash
bash /root/scripts/w1_env_check.sh
```

**验收**：输出 `ENV CHECK DONE`；GPU 是 A800-80GB；vllm 0.32.0。
**坑**：`vllm --version` 报 `Failed to infer device type` 是**无卡模式的正常假象**，以 `import vllm` 为准。

### Phase 1：补模型（约 30 分钟，自动）

```bash
bash /root/scripts/w1_refill_model.sh
# 监控（另开窗口或轮询）：
tail -f /root/autodl-tmp/refill.log
```

**验收**：日志出现 `DONE unified: /root/autodl-tmp/q36`；`ls /dev/shm/q36_shards/*.safetensors | wc -l` 约 20 个，tmp 里 6 个。
**坑**：
- shm 是内存盘，每次关机清空 → 每次开机都要跑这一步，属正常成本
- 脚本自动跳过 tmp 已有分片，只补缺；不要手动删任何分片
- 下载中断直接重跑同一命令，续传

### Phase 2：vLLM 服务（5-15 分钟）

```bash
bash /root/scripts/w1_start_vllm.sh
```

**验收**：输出 `VLLM READY` + 模型列表含 `qwen35b`。
**坑（按概率排序）**：

| 现象 | 原因 | 对策 |
|---|---|---|
| `KeyError/ValueError ... qwen3_5_moe` | vllm 0.32 不认此新架构 | 记录版本报错 → 升级 vllm 到最新；仍不行 → 转备选：transformers 直跑（见附录 A） |
| `CUDA out of memory` | 67GB 权重挤 80G 卡 | 降 `--max-model-len 8192`、`--gpu-memory-utilization 0.99`；仍不行 → 备选 A |
| 启动超时但无报错 | 权重加载慢 | 等满 15 分钟再判超时 |

### Phase 3：HLE baseline（冒烟 5 分钟 → 子集 1-2 小时）

```bash
bash /root/scripts/w1_hle_baseline.sh 10      # 冒烟
bash /root/scripts/w1_hle_baseline.sh 200     # W1 出分用 200 题
```

**验收**：输出 `Accuracy(rule-judge): X% | n = N`，结果 JSON 落盘。
**口径声明（必须写进结果文档）**：规则判分只覆盖 exactMatch/multipleChoice；thinking 输出上限 8192 token；n=200 的 95% 置信半宽约 ±3.5pp，**与官方 judge 的分数不可直接对比**（官方用 o3-mini 做 LLM judge）。这是 W1 的口径，W4 全量再统一。
**坑**：
- HLE 难度极高，35B 预期分数个位数百分比，0% 也可能是正确结果——先看判分样例再怀疑链路
- 单题最长 8k token，客户端 timeout ≥600s（脚本已设）
- 若 vLLM 挂了：现象是请求超时/连接拒绝，先看 `tail -20 /root/autodl-tmp/vllm.log`

### Phase 4：训练冒烟复验（10 分钟，可选但建议）

```bash
cd /root && /root/miniconda3/bin/python scripts/train_lora.py \
  --model /root/autodl-tmp/q36 --data data/hle_smoke.jsonl \
  --output /root/autodl-tmp/lora-smoke2 \
  --target_modules q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,in_proj_a,in_proj_b,out_proj,gate_proj,up_proj,down_proj \
  --epochs 1 --batch_size 1 --gradient_accumulation_steps 4
```

**验收**：`train_loss` 有限（<2）、显存峰值 <79GB、最后输出 `LoRA adapter saved`。
**坑**：vLLM 服务会占显存，**先停 vLLM** 再跑训练（`ps aux | grep 'vllm serve' | grep -v grep | awk '{print $2}' | xargs -r kill`）。

### Phase 5：三榜 harness 打通（风险最高，预留时间）

HLE 已通；**Terminal-Bench / ASI-Bench / SWE-bench 需要 Docker**，本 GPU 机无 Docker。两条路选一条：

- **路 A（推荐）**：另开 AutoDL **CPU 无卡实例**（¥0.1/时）装 Docker，GPU 机只留 vLLM 服务（开 `--host 0.0.0.0` + 安全组放行 8000），harness 从 CPU 机打 GPU 机的 OpenAI 端点
- 路 B：GPU 机内装 Docker（AutoDL 容器内通常不允许，试试 `docker info`）

各榜仓库（clone 后锁 commit hash 进 README，PRD 硬性要求）：

| 榜 | 仓库 | 打通标准 |
|---|---|---|
| Terminal-Bench | github.com/laude-institute/terminal-bench（Harbor：`harbor run`） | 1 题跑完出 pass/fail |
| ASI-Bench | 仓库官方框架 + adapter（PRD 4.5，需现查） | 1 题出分 |
| SWE-bench Pro v2 | agent harness + 断网沙箱；备选 mini-swe-agent | 1 题出 patch |

**这部分细节超出本 Runbook 已验证范围**，agent 需自行 spike；每榜记录：版本号、1-3 题原始输出、分数。卡住超 1 小时 → 记录阻塞原因 + 已试路径，停下来上报，**不要无限重试**。

### Phase 6：落档与交付

1. 结果写入 `docs/base-baseline-scores.md`：每榜分数 + n + 口径 + 原始 JSON 路径 + GPU 时长
2. `git add && commit && push`（push 若失败：直连不通换手机热点，几 MB 而已）
3. 微信通知主 agent（`tools/notify.py`，标题带时间戳）

## 3. 坑点速查（血泪汇总）

1. **shm 内存盘**：关机清空、无卡模式仅 2GB → 补模型只在 GPU 模式
2. **SSH 域名解析失败**：`ssh_exec.py` 用 `--host 36.137.238.251` IP 直连（重启后 IP 可能变）
3. **AutoDL SSH 限流**：命令间隔 ≥1 分钟，带重试；密集连接会 Authentication timeout
4. **pip 镜像**：阿里云可用；清华源 403；vllm 0.32 全依赖链 ~3GB（torch 2.13+cu13 系），耐心
5. **bnb 4bit 对 MoE routed experts 无效**（批量参数非 Linear）→ 全精度 bf16，别走量化弯路
6. **`prepare_model_for_kbit_training` 在 bf16 模式会 OOM**（全参数 cast fp32）→ 已修，勿回退
7. **HLE 数据集 HF gated 403** → 用 ModelScope 镜像 `cais/hle`（已在云机）
8. **67GB 权重挤 80G 卡**：KV cache 紧张，上下文别超 16K，并发 ≤4
9. **GPU 闲时切无卡模式**（¥0.59→¥0.1/时 级差），但切换前确认无进程在用 GPU

## 4. W1 出口对照

| PRD 出口标准 | 达成方式 |
|---|---|
| LoRA merge smoke test | ✅ 已验证（本地 3B 单专家 + 云机 35B 训练冒烟），见 cloud-ops-log.md 操作记录 |
| 四榜 harness 链路打通（各 1-3 题出分） | HLE=本 Runbook Phase 3；其余=Phase 5 |
| Base 四榜分数落档 | 每榜 ≥10 题级子集分数（带口径声明），HLE 可到 200 题 |

## 附录 A：vLLM 失败备选（transformers 直跑冒烟）

```bash
/root/miniconda3/bin/python /root/scripts/hle_baseline_transformers.py \
  --model /root/autodl-tmp/q36 --n 10
```

（此脚本需从 `scripts/hle/run_hle_smoke.py` 改造：把本地模型加载换成 OpenAI 端点调用以外的 transformers 版本——若需要，先写再跑；200 题子集约 2-4 小时，仅用于 vLLM 全挂时保 W1 出口。）
