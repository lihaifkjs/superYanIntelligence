# 一、题目背景
随着开源大模型能力不断提升，针对不同任务训练专精模型，并通过 Model Merging 将多个专家的能力整合到单一模型中，有机会在不显著增加推理成本的情况下提升模型的综合表现。

本题要求你基于 [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)，针对 SWE-bench Pro v2-hard、HLE、ASI-Bench、Terminal-Bench 四个 Benchmark，设计并训练多个专家模型，探索有效的模型合版方法，并验证合版模型能否同时提升不同任务上的能力。重点考察你的数据构建、模型训练、合版方法研究、实验设计、评测分析及成本控制能力。

# 二、核心任务
- 基于 Qwen3.6-35B-A3B，自主设计专家数量、任务划分及训练方案，围绕四个 Benchmark 构建高质量训练数据，完成数据清洗、去重、污染控制及各专家模型的实际训练。
- 调研并实现 Model Merging 方法，将多个专家合版为单一模型。至少比较 2 种合版方法或不同合版配置，通过对照实验分析不同方案的有效性、能力互补及性能退化。
- 在以下四个 Benchmark 上比较 Base、各专家模型与最终合版模型的表现：
  - [SWE-bench Pro v2-hard](https://labs.scale.com/blog/swe-bench-pro-v2)
  - [HLE](https://huggingface.co/datasets/cais/hle)
  - [ASI-Bench](https://github.com/apexin-ai/ASI-Bench)
  - [Terminal-Bench](https://github.com/harbor-framework/terminal-bench)
- 固定评测版本、数据及推理配置，Base 与合版模型须完成全部四项评测，各专家至少完成对应目标 Benchmark 的评测。分析模型能力的提升与退化、失败原因及不同训练和合版策略的影响，以四项 Benchmark 宏平均分数相对 Base 提升 3 个百分点为挑战目标。
- 记录各专家训练、模型合版及评测的配置、Loss 曲线、实验结果、GPU 使用时长和成本，分析不同方案的训练效率与投入产出比。

# 三、提交要求
请将完整方案及相关代码上传至个人 GitHub 仓库，最终以 GitHub 仓库链接提交。

仓库应包含数据处理与训练代码、专家模型配置、模型合版脚本、评测脚本、Loss 曲线、实验日志、结果分析及 README，确保实验可复现。

各专家模型及最终合版模型需上传至个人 Hugging Face 仓库，并在 GitHub README 中提供模型链接、加载方式及合版配置。

原则上以 35B 级模型开展实验，算力不足时允许降至 7B–8B 级模型，但需说明原因。不要求全参数训练，允许使用 LoRA/QLoRA 等高效训练方式，但必须实际完成专家训练、模型合版和指定评测，不接受仅提交实验设计或 Dry Run。