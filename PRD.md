# PRD：多专家训练与 Model Merging 合版

> 状态：草案 v0.1（2026-10-23，待确认后定稿）
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
| 专家划分 | **3 个**：SWE 专家 / HLE 推理专家 / Agentic 专家（ASI+Terminal 合并） | 两榜同属 agentic 操作，数据形态重叠；减少合并干扰；省一路数据。题目允许自主设计专家数量 |
| 训练方式 | QLoRA SFT，挂在 instruct 模型上 | Base 为 post-trained 模型（见 4.1），续训是最自然的设定，也是 merging 文献标准玩法 |
| 合版方法 | 对比 ≥2 种：候选 TIES / DARE / Task Arithmetic；router 层处理作为配置变量 | 满足"比较 2 种方法"要求；MoE router 是否参与合并本身是现成对比素材 |
| 评测策略 | 固定 harness + 推理配置，**先跑 Base baseline**，全量评测只跑一轮 | 无 baseline 则提升无据可依；小集验证先行，控制评测成本 |
| 算力路线 | AutoDL 按小时租：A100 训练 / H100 评测；评测时 harness 与模型服务分离 | 训练便宜（MoE 仅 3B 激活）；评测是推理密集，H100 省时间反而省钱 |
| 兜底条款 | 题目允许降 7B–8B 并说明原因（当前不启用） | 预算可控则不降，保留 35B 完整性 |

## 3. 范围与不做的事

- 不做全参数训练、不做 RL/DPO（题目未要求，SFT 足够支撑 +3pp 目标）
- 不做四专家方案（除非三专家实验显示 Agentic 域内冲突显著，再考虑拆四）
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
| Agentic | 终端/科研操作 | 多轮 agent 轨迹 | 3k～8k 条 | 终端操作、科研 workflow 类轨迹数据 |

- **格式**：全部 chat 格式，工具调用 schema 对齐选定 harness
- **thinking 处理**：待定（W1 spike 决定）——训练数据与评测形态（thinking on）需一致，否则合版后 thinking 退化难排查
- **污染控制三动作（写进报告）**：① n-gram 重叠检测训练集×评测集；② SWE-bench 11 个源仓库黑名单；③ HLE 题目文本去重。报告给出检测方法与通过率

### 4.3 训练方案

- QLoRA：4-bit 底座约 20GB 显存，A100 80G 单卡可训；r / target modules / lr 以 W1 spike 结果为准（混合架构的注意力层命名需适配）
- 序列 packing 消除 padding 浪费；agent 轨迹做长度截断/筛轮
- 节奏：每专家先 300～500 条冒烟（约 1 小时），确认 loss 与评测有动静再放量
- 预算：单专家 2 epoch × 30M token ≈ 2 小时 ≈ 4 元；四专家合计 10 小时内

### 4.4 合版方案

- 流程：各专家 LoRA 合入底座得到独立权重 → 计算相对 Base 的 task vector → 方法对比合并
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
| SWE-bench Pro v2-hard | agent harness + 断网沙箱；diff 导出干净镜像重判 |
| ASI-Bench | 仓库 harness，agent + Docker 执行 |
| Terminal-Bench | Harbor harness（`harbor run`），沙箱跑 agent |
| 推理配置 | 对齐官方：`temp=1.0, top_p=0.95, top_k=20`，thinking on |
| 版本锁定 | 全部依赖锁版本号 + commit hash 入 README（Terminal-Bench 为持续更新榜，必须锁 tag；SWE-bench Pro v2 发布于 2026-10-16，注意跟踪修正） |
| 评测矩阵 | Base×4 + 合版×4 + 各专家×对应榜（≥3） |
| 成本控制 | 每榜先 10 题级子集验证，全量评测只跑一轮；评测方差（Hard 仅 51 题）在报告中声明 |

### 4.6 基础设施

- 训练：AutoDL A100 80G；开发/传数据用无卡模式
- 评测架构：GPU 机跑 vLLM 服务（OpenAI 接口），CPU 机跑 harness + Docker（规避容器内套 Docker 的限制）
- 代码与实验记录：git 管理，每轮实验记录 config / loss / 分数（W&B 或等价工具）

## 5. 里程碑

| 周 | 目标 | 出口标准 |
|---|---|---|
| W1 | 环境 + 架构适配 spike + Base baseline | LoRA merge smoke test 通过；Base 四榜分数落档 |
| W2 | 数据构建 ×3 + 首个专家端到端 | 数据通过污染检测；SWE 专家小集评测 > Base |
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
- [ ] HuggingFace：3 个专家权重 + 合版模型 + README（加载方式、合版配置）
- [ ] 报告：Base/专家/合版四榜对比、退化归因、成本账（GPU 时长×单价）

## 9. 待确认问题（接题时向 HR/出题方提出）

1. 截止时间？（在职申请延长至 4 周）
2. 是否提供算力补贴或云资源？
3. 验收细节：7B 降配接受度、提交格式、评分权重（过程 vs 结果）

## 10. 变更记录

- v0.1（2026-10-23）：初版，汇总讨论结论。待定项：harness 选型、thinking 数据策略、LoRA 超参（W1 spike 定）。
