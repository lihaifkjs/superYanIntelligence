# Base 模型四榜 Baseline 分数（W1 落档）

> 日期：2026-10-25 ｜ 模型：Qwen3.6-35B-A3B bf16 全精度（未训练 Base）
> 服务：vLLM 0.32.0（eager 模式，32k 上下文，`--tool-call-parser qwen3_xml`）
> ⚠️ **口径声明**：以下分数均为**自研规则判分 / harness 原生判分**，与官方 leaderboard 的 judge 口径**不可直接对比**（官方多用 LLM judge）。W1 只用于「Base vs 专家/合版」的同口径纵向对比，W4 全量评测时再统一口径。

## 总览

| 榜 | Base 分数 | n | 原始数据 |
|---|---|---|---|
| HLE | **0.5%** | 200 | `outputs/base-eval/hle_pred_200.json`（云机 `/root/autodl-tmp/`） |
| Terminal-Bench | **0/10 resolved** | 10 | WSL `~/w1/outputs/tb-baseline10/2026-10-25__14-07-10/results.json` |
| ASI-Bench | **0 / 100** | 1 | WSL `~/w1/out/local_score_seed31415.json` + trajectory |
| SWE-bench Pro v2 | 见下方专节 | 1（攻坚中） | WSL `~/w1/outputs/swe-mini3/` |

## HLE（200 题子集）

- **Accuracy(rule-judge): 0.5%（1/200）**；冒烟 10 题 0/10（链路验证，不含在 200 内）
- 判分：规则判分（exactMatch 走 `judge.py` answers_equal；multipleChoice 提取字母），thinking 上限 8192 token，temperature=0
- 失败画像：**95%（190/200）未提取到答案**（模型不收束、不交 `Answer:` 行），4.5% 提取到但答错——行为问题远大于知识问题
- 注：约 10 题因 vLLM 重启被记为错题（下偏影响 ≤0.5pp）；n=200 的 95% 置信半宽约 ±1pp
- 评测脚本：`scripts/cloud/hle_baseline_concurrent.py`（--workers 8 --save-raw）

## Terminal-Bench（10 题子集）

- **0/10 resolved**（另：hello-world 单题冒烟 PASS，验证 harness 链路）
- 子集：original-tasks 按字母序前 10（3d-model-format-legacy … mlflow-register），terminus agent，n-concurrent=2
- 口径备注：前 3 题（puzzle-solver / reshard-c4-data / dna-assembly）遭遇 vLLM ninja 故障期，agent 中途断连记为未解决——**该 0/10 偏悲观**，W4 全量前建议故障-free 复测
- harness：terminal-bench @ `d28711d0da2675d0bb1d56de45ae5df6082438a3`（锁定）

## ASI-Bench（单题打通验证）

- **0 / 100**（astronomy.nbody_close_encounters，B1 档，seed31415 本地打分 non-official）
- 链路已全通：asibench → opencode_cli agent（linux_ns 沙箱）→ vLLM → 本地 scoring；**0 分原因是 base 模型不发规范工具调用**（step_finish reason=stop，无 tool_call）——W2 训练靶点
- harness：ASI-Bench @ `b100adef1acef9e38cc2cea298f2d07a8cc1fd84`（锁定）；CLI：asibench 0.2.0

## SWE-bench Pro v2（攻坚专节）

- 官方仓库：github.com/scaleapi/SWE-bench_Pro-os（Scale AI，V2 已公开，642 题 + HARD-51；V2 用 Harbor 协议）
- 已验证：官方 ghcr 任务镜像可拉（3.6GB）、mini-swe-agent 2.4.6 容器沙箱跑通、**模型↔agent 工具调用已双向验证**（`qwen3_xml` 探针：`<function=bash><parameter=command>` 正确解析为结构化 tool_calls）
- 状态：单题 model.patch 产出进行中（NodeBB post-cache 任务）；详见 cloud-ops-log.md 操作记录
- harness：mini-swe-agent 2.4.6（SWE-bench 官方轻量 agent，备选方案，未用 Modal/locked 协议——本地 Docker 沙箱）

## 关键工程结论（影响 W2-W4，详见 cloud-ops-log.md）

1. **Qwen3.6-35B base 的原生工具调用格式是 Qwen XML 风格**（`<function=名><parameter=参数>`），vLLM 须 `--tool-call-parser qwen3_xml`；hermes/JSON 解析器全是错配
2. **vLLM 假死根因 = 缺 ninja**（flashinfer 采样 JIT 编译）：已软链 /usr/bin/ninja + 启动脚本 PATH 修复 + env_check 门禁
3. HLE 的 0.5% 主要死于「不收束」（95% 无 Answer），**格式/行为训练是第一优先级**，知识灌输其次
4. ASI/SWE 的 0 分均死于「不发工具调用」，agentic 任务的工具调用格式训练是 W2 必修课

## 成本

- GPU 时长：2026-10-25 约 11:00-17:30 ≈ 6.5h × ¥5.59 ≈ **¥36**（003 机）
