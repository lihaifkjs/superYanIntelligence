# PRD：多专家训练与 Model Merging 合版

> 状态：v0.6（2026-10-24，实现对齐：adapter 直算合并 / bf16 训练默认 / 模块拆分与测试入仓）
> 题源：[`task.md`](task.md)（出处未标注）
> 文档原则：只记录决策、契约、坑点与验证结论，不堆砌过程。

## 1. 背景与目标

基于 Qwen3.6-35B-A3B，针对 SWE-bench Pro v2-hard / HLE / ASI-Bench / Terminal-Bench 四榜，训练多个专家模型并合版为单一模型。

- **挑战目标**：四榜宏平均较 Base 提升 3 个百分点
- **验收硬要求**：真实训练、真实合版、真实评测（不接受 Dry Run）；交付 GitHub 仓库 + HuggingFace 模型权重 + 可复现证据链
- **性质**：面试作业，过程质量 ≥ 最终分数

## 2. 关键决策（本轮讨论结论）

| 决策点 | 结论 | 理由 |
|---|---|---|
| 专家划分 | **4 个**：SWE / HLE / ASI / Terminal，与 benchmark 一一对应 | 与题目字面"围绕四个 Benchmark"零偏差；各专家目标聚焦。专家数量本为开放设计项，此为最直白方案 |
| 训练方式 | QLoRA SFT，挂在 instruct 模型上 | Base 为 post-trained 模型（见 4.1），续训是最自然的设定，也是 merging 文献标准玩法 |
| thinking 策略 | 分密度：HLE / SWE / ASI 带完整 thinking，Terminal 按命令复杂度分档（短 thinking 或不带）；轨迹以自蒸馏为主 | 与评测形态（thinking on）一致防退化；自蒸馏每方向约十几元，成本可控；W1 用"全带 vs 分密度"对照验证收益 |
| 合版方法 | 对比 ≥2 种：候选 TIES / DARE / Task Arithmetic；router 层处理作为配置变量 | 满足"比较 2 种方法"要求；MoE router 是否参与合并本身是现成对比素材 |
| 评测策略 | 固定 harness + 推理配置，**先跑 Base baseline**，全量评测只跑一轮 | 无 baseline 则提升无据可依；小集验证先行，控制评测成本 |
| 算力路线 | AutoDL 按小时租：A100 训练 / H100 评测；评测时 harness 与模型服务分离 | 训练便宜（MoE 仅 3B 激活）；评测是推理密集，H100 省时间反而省钱 |
| 兜底条款 | 题目允许降 7B–8B 并说明原因（当前不启用） | 预算可控则不降，保留 35B 完整性 |

## 3. 范围与不做的事

- 不做全参数训练、不做 RL/DPO（题目未要求，SFT 足够支撑 +3pp 目标）
- 不增设第五专家、不细分 HLE 子方向（维持与 benchmark 1:1 对应，控制变量）
- 不追求评测边界用例覆盖，聚焦主流程冒烟 + 记录遗留项

## 4. 技术方案

### 4.1 Base 模型事实（已验证，来源：官方模型卡）

- **是 post-trained instruct 模型**（非预训练裸模）：默认 thinking 模式，支持 tool call，官方 chat 采样参数
- **多模态**（带视觉编码器）：四榜纯文本，训练与评测走 text-only
- **混合架构**：Gated DeltaNet 线性注意力 + Gated 全注意力，40 层
- **MoE**：256 专家，每 token 激活 8 routed + 1 shared；总 35B / 激活约 3B
- 官方评测采样参数：thinking 模式 `temp=1.0, top_p=0.95, top_k=20`；输出上限建议 32K～82K token；上下文原生 262K（建议评测≥128K）

### 4.2 数据方案

| 专家 | 方向 | 数据形态 | 规模目标 | 来源思路 |
|---|---|---|---|---|
| SWE | 真实仓库修复 | 多轮 agent 轨迹（含工具调用） | 3k～8k 条 | 公开 SWE 任务集轨迹、开源 agent 轨迹数据集（避开评测 11 仓库） |
| HLE | 极难知识推理 | 难题问答 + 推理过程 | 3k～8k 条 | 高难度 QA/竞赛/推理数据集（剔除与 HLE 题目重叠） |
| ASI | 科研项目级任务 | 多轮 agent 轨迹（科研 workflow） | 3k～8k 条 | 科研计算/实验操作类轨迹数据 |
| Terminal | 终端操作任务 | 多轮 agent 轨迹（终端命令） | 3k～8k 条 | 终端操作、shell 任务类轨迹数据 |

- **格式**：全部 chat 格式，工具调用 schema 对齐选定 harness
- **thinking 策略**：分密度（见决策表）——HLE / SWE / ASI 轨迹带完整 thinking，Terminal 按命令复杂度分档；轨迹来源以自蒸馏为主（Base 模型在 vLLM 上自生成，每方向约十几元），辅以现成开源 thinking 数据集；W1 用"全带 vs 分密度"对照实验验证收益，防合版后 thinking 退化难排查
- **污染控制三动作（写进报告）**：① n-gram 重叠检测训练集×评测集；② SWE-bench 11 个源仓库黑名单；③ HLE 题目文本去重。报告给出检测方法与通过率

### 4.3 训练方案

- QLoRA 挂 instruct 底座。加载方式按机型二选一：80G 云卡默认 **bf16 全精度 + 手动冻结**（bnb 不量化 MoE experts，4bit 无收益且 kbit 预处理会把参数 cast fp32 导致 OOM）；低显存机器 `--load-in-4bit` 回退 QLoRA。r / target modules / lr 以 W1 spike 结果为准（混合架构注意力层命名需适配）
- 序列 packing 消除 padding 浪费；agent 轨迹做长度截断/筛轮
- 节奏：每专家先 300～500 条冒烟（约 1 小时），确认 loss 与评测有动静再放量
- 预算：单专家 2 epoch × 30M token ≈ 2 小时 ≈ 4 元；四专家合计 10 小时内

### 4.4 合版方案

- 流程：各专家 LoRA adapter **直接**参与合并（delta = α/r·B@A 逐 tensor 流式计算，无需先合入底座，35B 省 70G+ 内存；`scripts/merge_experts.py`）→ 相对 Base 的 task vector → 方法对比合并（4 个 task vector 同时合并，干扰源多于文献常见的 2～3 任务场景，去噪剪枝方法的预期收益更大，这本身是重点验证假设）
- 对比矩阵（≥2 种方法 × router 参与/冻结）：
  | 方法 | 说明 |
  |---|---|
  | Task Arithmetic | 基线：直接加权求和 |
  | TIES 或 DARE | 去噪剪枝，预期多任务合并更稳 |
- **MoE 专项**：router 权重被扰动可能导致路由崩溃；shared expert 与 routed expert 分开处理。此为报告分析重点（能力互补与退化的机制解释）
- 合版 smoke test：两专家小模型先验证合并→加载→推理链路

### 4.5 评测方案

| 项 | 决定 |
|---|---|
| HLE | HF 数据集直接评测，唯一无 Docker 的榜 |
| SWE harness | 官方 harness 优先（分数可对照 leaderboard）；备选 mini-swe-agent（已实证可跑本模型 + 本地 vLLM）。注意其行为塑造效应（30s 命令超时）需与数据方案联动 |
| Terminal-Bench harness | Harbor 官方（`harbor run`）+ 支持 OpenAI 端点的 agent；评模型前先跑 oracle 验证环境不 flake |
| ASI-Bench harness | 仓库官方框架 + adapter；接不上时自写薄 adapter |
| HLE 评测脚本 | 官方 eval 脚本（cais/hle 仓库）；逻辑简单可自写兜底 |
| SWE-bench Pro v2-hard | agent harness + 断网沙箱；diff 导出干净镜像重判 |
| ASI-Bench | 仓库 harness，agent + Docker 执行 |
| Terminal-Bench | Harbor harness（`harbor run`），沙箱跑 agent |
| 推理配置 | 对齐官方：`temp=1.0, top_p=0.95, top_k=20`，thinking on |
| 版本锁定 | 全部依赖锁版本号 + commit hash 入 README（Terminal-Bench 为持续更新榜，必须锁 tag；SWE-bench Pro v2 发布于 2026-10-16，注意跟踪修正） |
| 评测矩阵 | Base×4 + 合版×4 + 各专家×对应榜（4） |
| 成本控制 | 每榜先 10 题级子集验证，全量评测只跑一轮；评测方差（Hard 仅 51 题）在报告中声明 |

### 4.6 基础设施

- 训练：AutoDL A100 80G；开发/传数据用无卡模式
- 评测架构：GPU 机跑 vLLM 服务（OpenAI 接口），CPU 机跑 harness + Docker（规避容器内套 Docker 的限制）
- 代码与实验记录：git 管理，每轮实验记录 config / loss / 分数（W&B 或等价工具）

## 5. 里程碑

| 周 | 目标 | 出口标准 |
|---|---|---|
| W1 | 环境 + 架构适配 spike + Base baseline | LoRA merge smoke test 通过；四榜 harness 链路打通（各 1～3 题出分）；Base 四榜分数落档 |
| W2 | 数据构建 ×4 + 首个专家端到端 | 数据通过污染检测；SWE 专家小集评测 > Base |
| W3 | 全部专家训练 + 合版矩阵 | 合版小集筛选出 1～2 个候选配置 |
| W4 | 全量评测 + 分析 + 仓库整理 | 交付物齐套，可复现 |

铁律：**先打通端到端细线，再扩展**。

## 6. 成本预算

- 训练：约 10～20 元
- 评测（成本大头）：AutoDL H100 按小时，总预算控制在 100～200 元
- 先充 100 元起步；若获出题方算力补贴再升配置

## 7. 风险与对策

| 风险 | 等级 | 对策 |
|---|---|---|
| 混合架构工具链不适配（PEFT/mergekit 对线性注意力+MoE 支持未知） | 高 | W1 第一个 spike 验证，预留 2～3 天改代码 |
| thinking 形态不一致导致分数莫名退化 | 中 | W1 用小型对照实验定数据格式 |
| Terminal-Bench 版本滚动 / SWE-bench Pro v2 刚发布有变动 | 中 | 锁 tag + 记录 hash；关注官方 issue |
| router 合并导致路由崩溃 | 中 | router 冻结作为默认配置，参与合并作为对比项 |
| 评测方差吞掉 +3pp | 中 | 固定配置 + 报告中声明方差；必要时多次取均值 |
| 算力/时间超支 | 低 | 小集先行；兜底降 7B 并说明 |

## 8. 交付物清单（对照 task.md 第三节）

- [ ] GitHub 仓库：数据处理/训练/合版/评测代码、专家配置、Loss 曲线、实验日志、结果分析、README（含复现步骤）
- [ ] HuggingFace：4 个专家权重 + 合版模型 + README（加载方式、合版配置）
- [ ] 报告：Base/专家/合版四榜对比、退化归因、成本账（GPU 时长×单价）


## 9. 变更记录

- v0.1（2026-10-23）：初版，汇总讨论结论。待定项：harness 选型、thinking 数据策略、LoRA 超参（W1 spike 定）。
- v0.2（2026-10-23）：专家划分由 3 改为 4（与 benchmark 一一对应），同步更新数据方案、评测矩阵、里程碑与交付物清单。
- v0.3（2026-10-23）：harness 选型落定（4.5 节），W1 出口标准补充"四榜 harness 链路打通"。待定项剩 2：thinking 数据策略、LoRA 超参。
- v0.4（2026-10-23）：thinking 策略定为"分密度 + 自蒸馏为主"（决策表 + 4.2 节），W1 增加"全带 vs 分密度"对照实验。待定项剩 1：LoRA 超参。
- v0.5（2026-10-23）：W2 增加本机 3B 预演阶段——自蒸馏先 3B 后 35B（脚本 `--model` 参数化，逻辑相同仅换路径）；Terminal 轨迹自蒸馏用 Git Bash 沙箱替代 Docker（smoke 级）；SWE/ASI 本机只做开源轨迹格式适配，真实生成留云；新增多专家合并脚本（TA/TIES，流式，router 冻结开关）与云上 runbook（`scripts/cloud/`）。详见 `docs/w2-progress.md`。
- v0.6（2026-10-24）：实现对齐 + 评审修复。①训练默认改 bf16 全精度（`--load-in-4bit` 回退），合版改为 adapter 直算 task vector；②`pipeline_common` 拆为 prompts/judge/llm_io/io_utils（build_sft 不再拖 torch），Terminal 种子库拆为 `seed_bank.py`，HLE 判分统一走 `judge.answers_equal`（数据过滤与评测同尺）；③修复评审发现的 5 个真 bug：freeze_router 丢权重、合并产物 fp32 混存、qa 批解码抹 thinking、LoRA key 前缀硬编码无断言（**该断言 retroactively 发现 10-23 的合并产物实为 base 拷贝**，已修复重跑）、train_lora 默认参数云上必失败；④`tests/run_tests.py` 入仓（判分/解析/合并数学 26 项）。
