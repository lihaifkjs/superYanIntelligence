# W2 云机操作 Runbook（交接文档）

> 面向：独立云操作 agent。目标：35B 自蒸馏数据构建 ×4 + 首个专家（HLE）端到端训练与小集评测出分。
> 前置阅读：`docs/cloud-session-state.md`（机器状态）、`docs/cloud-w1-runbook.md`（W1 未完成项是本文前置）、`docs/cloud-verify-checklist.md`（含第 0.5 步）。
> 机制全部在本机 3B 验证过（`docs/w2-progress.md`）；本文只解决"35B + 云环境"的差异。

## 0. 输入清单（开始前必须齐备）

| 输入 | 说明 |
|---|---|
| W1 完成 | vLLM 可选；**补模型必须完成**（`/root/autodl-tmp/q36` 统一视图可用，`bash /root/scripts/w1_refill_model.sh`） |
| git 仓库最新 main | **必须含 10-24 评审修复**（commit `8123ad8` 之后）：`git clone https://github.com/lihaifkjs/superYanIntelligence /root/syi`；云上旧副本 `git pull` |
| 云机 Python | `/root/miniconda3/bin/python`（3.12，训练栈已装）；**全部命令加 `PYTHONIOENCODING=utf-8`** |
| HLE 评测 parquet | `/root/autodl-tmp/hle/datasets/caiss--hle/snapshots/master/data/test-00000-of-00001.parquet`（W1 已备） |
| Docker | Terminal 沙箱用（`distill.py --sandbox auto` 自动探测）；`docker pull alpine/git`（git 类任务需要） |
| SSH 凭据 | `credentials.md`；关机重开 SSH/IP 可能变，以控制台为准；命令间隔 ≥1 分钟（AutoDL 限流） |

## 1. 范围

### 做什么（W2 云侧）

1. **第 0 步**：35B 蒸馏输出格式验证（5 条 qa 人工看原始输出，**不通过不得放量**）
2. 种子数据：HLE 数学 3000 题 + Terminal 模板生成 300 任务
3. 自蒸馏：HLE qa 批量 + Terminal agent（Docker 沙箱）→ 过滤 → SFT 数据 ×4（SWE/ASI 用开源轨迹 `prepare_swe_asi.py`）
4. 污染检测（HLE 侧 n-gram）
5. HLE 专家训练（bf16 全精度）→ HLE 50 题小集评测，与 Base 同口径对比落档
6. 结果落档 `docs/` + push + 微信通知

### 不做什么（W2 之外）

- ❌ 合版矩阵实验（TA/TIES 全量对比是 W3；本机已验证机制，别抢跑）
- ❌ SWE/ASI **真实轨迹自蒸馏**（需要仓库执行环境 + 断网沙箱，随 harness 打通后再做，W1 Phase 5 路 A 的 CPU 机就位后启动）
- ❌ 全量评测（2500 题 × 4 榜是 W4）
- ❌ 上传 HuggingFace（token 未就绪）
- ❌ 动 `/root/autodl-tmp/q36*` 权重布局；给 router 加 LoRA
- ❌ 装 `causal-conv1d`（编译坑，线性注意力慢 2-5 倍先忍着）

## 2. 执行步骤

### Phase 0：第 0 步格式验证（5 分钟，不可跳过）

```bash
cd /root/syi && git log --oneline -1   # 确认 ≥ 8123ad8
/root/miniconda3/bin/python scripts/cloud/w2_step0_format_check.py \
  --model /root/autodl-tmp/q36
```

**验收**：脚本报 `PARSE OK, reasoning_len>0`；**人工看打印的 RAW 全文**——确认 thinking 用什么标记、ANSWER 行是否照做。若 `reasoning_len=0`：看 RAW 实际格式，改 `scripts/judge.py` 的 `THINK_RE`/`_DECODE_STRIP` 后重验。
**坑**：Qwen3.6 的 `<think>` 是特殊 token，旧解码方式会静默抹掉（本机已踩过，代码已修，此步是防回归）。

### Phase 1：数据（蒸馏是大头，先 smoke 后放量）

```bash
cd /root/syi
export PYTHONIOENCODING=utf-8
PY=/root/miniconda3/bin/python

# 1. 种子
$PY scripts/prepare_seeds.py --expert hle --n 3000 --cache_dir /root/autodl-tmp/ms_cache
$PY scripts/prepare_seeds.py --expert terminal --n_generated 300

# 2. 冒烟：HLE 30 题 + Terminal 10 任务（约 15 分钟，看通过率再放量）
$PY scripts/distill.py --mode qa --model /root/autodl-tmp/q36 \
    --seeds data/seeds/hle.jsonl --limit 30 --batch_size 4
$PY scripts/distill.py --mode agent --model /root/autodl-tmp/q36 \
    --seeds data/seeds/terminal.jsonl --limit 10 --max_turns 12

# 3. 放量（断点续跑：中断直接重跑同命令；估算 2-6 小时，看 smoke 速度）
$PY scripts/distill.py --mode qa --model /root/autodl-tmp/q36 \
    --seeds data/seeds/hle.jsonl --batch_size 4
$PY scripts/distill.py --mode agent --model /root/autodl-tmp/q36 \
    --seeds data/seeds/terminal.jsonl --max_turns 12 --sandbox auto

# 4. 过滤 + SWE/ASI 格式数据
$PY scripts/build_sft.py --traj data/raw_traj/hle.jsonl
$PY scripts/build_sft.py --traj data/raw_traj/terminal.jsonl
$PY scripts/prepare_swe_asi.py --expert both --n 2000 --cache_dir /root/autodl-tmp/ms_cache

# 5. 污染检测（exit 非零即停）
$PY scripts/check_contamination.py --train data/sft/hle.jsonl \
    --eval /root/autodl-tmp/hle/datasets/caiss--hle/snapshots/master/data/test-00000-of-00001.parquet
```

**验收**：`build_sft` 两行 `filter_pass_rate` 打印（HLE 期望 ≥50%，3B 时 72%；Terminal 完成检查通过率 ≥40% 算健康）；`ls data/sft/` 四个文件非空；污染检测 `[PASS]`。
**坑**：

| 现象 | 原因 | 对策 |
|---|---|---|
| `build_sft.py` 非零退出报 unchecked | `outputs/sandbox/` 被清理 | 该目录是 Terminal check 的依赖，**别删**；重跑 distill 或补回 |
| `filter_pass_rate` 极低（<10%） | 35B 不照 ANSWER 格式输出 | 回 Phase 0 看 RAW；可微调 `SYSTEM_PROMPT_QA` 后重蒸馏（断点续跑只补新 id，改 prompt 前先删 `data/raw_traj/hle.jsonl`） |
| agent 轨迹全 max_turns | done JSON 解析失败 | 抽 3 条看 `raw_output` 字段；`parse_agent_action` 已兼容裸 JSON/tool_call 两种形状，第三种形状才需要改代码 |
| Terminal 沙箱走成本地 bash | Docker 探测失败 | `docker info` 看守护进程；`--sandbox docker` 强制报错诊断 |
| OOM | batch_size 8 + 长生成 | qa 模式 `--batch_size 4`（已默认建议）；agent 模式逐轮天然省显存 |

### Phase 2：HLE 专家训练（bf16，约 1-2 小时）

```bash
cd /root/syi
ps aux | grep 'vllm serve' | grep -v grep | awk '{print $2}' | xargs -r kill  # 停 vLLM 腾显存
/root/miniconda3/bin/python scripts/train_lora.py \
  --model /root/autodl-tmp/q36 \
  --data data/sft/hle.jsonl \
  --output /root/autodl-tmp/lora-hle \
  --target_modules q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,in_proj_a,in_proj_b,out_proj,gate_proj,up_proj,down_proj \
  --epochs 2 --batch_size 1 --gradient_accumulation_steps 4
```

**验收**：`train_loss` 有限下降（<2 起步、尾值 <0.8 期望）；显存峰值 <79GB；`LoRA adapter saved`。
**坑**：

- `--load-in-4bit` **不要加**：bnb 量化不到 MoE routed experts，且 kbit 预处理会把全参数 cast fp32 直接 OOM（决策已废弃，见 cloud-session-state）
- target_modules 用上面全量列表（spike 结论，含线性注意力 in_proj_*）；漏挂 `in_proj_*` 等于线性注意力层白训
- 中断直接重跑，checkpoint 按 epoch 存

### Phase 3：小集评测与 Base 对比（1-2 小时）

Base 与专家**同口径各跑一遍**（thinking 模式、同一批 50 题、同一 judge）：

```bash
cd /root/syi
# 起 vLLM（评测走服务；训练已完成不占显存）
nohup /root/miniconda3/bin/vllm serve /root/autodl-tmp/q36 --served-model-name qwen35b \
  --max-model-len 16384 --gpu-memory-utilization 0.97 --enforce-eager --port 8000 \
  > /root/autodl-tmp/vllm.log 2>&1 &
# 等 VLLM READY 后：
/root/miniconda3/bin/python scripts/cloud/hle_baseline.py \
  --parquet /root/autodl-tmp/hle/datasets/caiss--hle/snapshots/master/data/test-00000-of-00001.parquet \
  --url http://127.0.0.1:8000/v1 --model qwen35b --n 50 --out /root/autodl-tmp/hle_base_50.json
# 合并 LoRA 出专家独立权重（merge_lora 单专家合并，本机已验证）再同样跑 50 题
/root/miniconda3/bin/python scripts/merge_lora.py --model /root/autodl-tmp/q36 \
  --adapter /root/autodl-tmp/lora-hle/final --output /root/autodl-tmp/q36-hle
# 重启 vLLM 指向 /root/autodl-tmp/q36-hle，重复上面 baseline 命令 --out .../hle_expert_50.json
```

**验收**：两个 JSON 落盘；`Accuracy(rule-judge)` 写进结果文档。n=50 时 95% 置信半宽 ±14pp——**只能证明"有动静、无劣化"，不能证明 +3pp**；专家分数 ≥ Base 即过 W2 出口（宏平均是 W4 的事）。
**坑**：HLE 极难，35B 预期个位数百分比；0% vs 0% 也是有效结果（写明即可）。判分口径 = `judge.answers_equal`（与数据过滤同尺），但与官方 o3-mini judge 不可直接对比（W4 统一）。

### Phase 4：落档与交付

1. 写 `docs/w2-cloud-results.md`：蒸馏通过率、数据规模、训练 loss、Base/专家 50 题对比（带口径）、GPU 时长（`AutoDL 控制台`）、原始 JSON 路径
2. `git add && commit && push`
3. 微信通知主 agent（`tools/notify.py`，标题带时间戳）

## 3. 坑点速查（W2 特有风险）

1. **第 0 步不可跳过**：格式不符放量 = 整批数据 reasoning 为空，build_sft 会全灭报错（这是最后的防线，别真靠它）
2. **显存互斥**：vLLM 服务 / 蒸馏 / 训练三者同时只能活一个（67GB 权重 + KV cache vs 80G 卡）
3. **改 prompt 后必须删旧 raw_traj**：断点续跑按 id 跳过，不改文件重跑等于白跑
4. **`outputs/sandbox/` 是 check 的依赖**：Terminal 过滤要重放 check 命令，清理沙箱 = 全部 unchecked 报错
5. **成本节奏**：每阶段先 smoke 再放量；等待期间切无卡模式（¥5.59→¥0.1/时）；数据盘只有 50GB，产物及时 commit/push 后可清（raw_traj 保留，sandbox 可重建）
6. **SSH 限流**：命令间隔 ≥1 分钟；长任务一律 `nohup ... &` + 日志文件，别挂前台 SSH

## 4. W2 出口对照（PRD + v0.6 调整）

| 出口标准 | 达成方式 |
|---|---|
| 数据通过污染检测 | Phase 1 第 5 步（HLE 侧；SWE/ASI 侧随 harness 后补，已在遗留项登记） |
| 首个专家端到端 | Phase 2+3：HLE 专家训练 + 50 题小集 vs Base 同口径落档（PRD 原文写 SWE 专家；SWE 真实轨迹依赖 Docker 执行环境，W2 内以开源轨迹格式数据 + HLE 专家端到端替代，差异已声明） |

## 5. 一键脚本

`bash scripts/cloud/w2_pipeline.sh`（同文件 rsync 到云机）= Phase 0-3 串行，每步 `--skip-xxx` 可跳，失败即停。冒烟/放量分两档：`SCALE=smoke|full`。
