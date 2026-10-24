# W2 进展记录

> 日期：2026-10-23（10-24 两次加固评审后更新）
> 状态：本机 3B 全路径预演完成（10/10 验证通过）+ 两轮代码评审修复；云上 35B 开机即跑

## 评审修复第二轮（2026-10-24，外部 agent 全仓评审）

评审四组问题，处置如下（详见 PRD v0.6 变更记录）：

**会真出错的 5 条——全部属实，已修：**

1. `merge_experts --freeze_router` 直接 continue → router 权重不进产物（"冻结"变"丢权重"，35B 必炸）→ 改为原样拷入 base 权重
2. `merge_tensor` 返回 fp32 → 产物 dtype 混存体积翻倍 → 返回 cast 回 base dtype
3. `distill.py` qa 批解码 `skip_special_tokens=True` → 35B thinking 被抹（w2 记过的坑换处回归）→ 与 `llm_io.generate` 同策略
4. LoRA key 前缀硬编码无断言 → **新加的整体匹配率断言 retroactively 发现 10-23 的 merged-ta/merged-ties 实为 base 拷贝**（key 拼接多了一段 `.weight`，全部匹配失败但静默继续；当时的"合并冒烟通过"是 base 3B 自己会做简单数学题，验证方法不够刁钻）。已修（adapter key = 模块路径去 `.weight` 后缀 + lora_A/B）并重跑：每 adapter 252/434 key 命中，merged 与 base 实测 max|Δ|=1.2e-4
5. `train_lora.py` 默认 HF repo id 不走 ModelScope → 云上默认参数必失败 → 接入 `resolve_model_path`，默认改当前流水线路径

**内聚/重复：** pipeline_common 拆为 `prompts` / `judge` / `llm_io` / `io_utils`（build_sft 不再拖 torch）；Terminal 种子库拆 `seed_bank.py`；HLE 判分统一 `judge.answers_equal`（run_hle_smoke / hle_baseline 与数据过滤同尺）。

**隐式契约：** build_sft 加 `--expert` 显式参数（文件名推断失败即报错）+ 轻量轨迹 schema 校验；agent system prompt 改为描述工具调用协议（与 tool_calls 训练格式一致），`parse_agent_action` 兼容裸 JSON 与 tool_call 两种形状。

**测试与文档：** `tests/run_tests.py` 入仓（26 项：判分/解析/合并数学/router 正则/模板自检）；根 `README.md` 补建；PRD 4.3/4.4 与实现对齐。`load_and_quant.py`（10-24 建的云脚本）与 quantize_4bit.py 重复一事已知悉，因是当天手工操作产物暂保留，上云 runbook 以 train_lora 的 bf16 路径为准。

**遗留（评审提及未做）：** SWE 11 仓库黑名单、HLE 训练集×评测集去重（check_contamination 目前只覆盖 HLE 评测集侧）——等 SWE/ASI harness 上云后一并落地。

## 目标与定位

W2 正式交付（数据×4 + 首个专家端到端）依赖 35B 自蒸馏与 Docker harness，本机做不了。
本轮把 W2/W3 的**整条机制链路**用 3B 在本机全部跑通，所有脚本 `--model` 参数化，
上云后按 `docs/cloud-w2-runbook.md` 执行（`scripts/cloud/w2_pipeline.sh`），不浪费计费时长。

## 本机验证结果（3B，Qwen2.5-3B-Instruct，4070Ti 12G）

| 环节 | 结果 |
|---|---|
| 种子数据 | Terminal 内置 30 任务（带 setup/check 检查命令）；HLE 从 ModelScope `AI-ModelScope/Big-Math-RL-Verified`（25 万题）抽 300 |
| HLE 蒸馏（qa） | 50 题生成成功，答案过滤通过率 40%（20 条入 SFT） |
| Terminal 蒸馏（agent） | 30 任务跑完（27 done / 3 触顶），Git Bash 沙箱安全黑名单验证通过（rm -rf /、fork bomb 等拦截，正常命令放行），任务完成检查通过率 53%（16 条入 SFT） |
| SWE/ASI 格式 | `SWE-agent-trajectories`（role/text 步序列）与 `agent-sci-general`（conversations）各转 20 条统一 messages，四专家训练全部成功（loss 正常） |
| 多轮 tool-call 训练 | `train_lora.py --tools` 经 `apply_chat_template(tools=...)` 序列化，Qwen2.5 template 原生支持，无需额外处理 |
| 多专家合并 | 4 个 LoRA adapter 直接参与合并（无需先 merge 进底座）：Task Arithmetic 与 TIES 各出一版，434 tensors，流式按 tensor 处理 |
| 合并产物冒烟 | 自写 safetensors 分片 + index 能被 transformers 正常加载，两版 merged 均生成正常输出 |
| 污染检测 | HLE SFT 数据 × 评测集 8-gram 重叠 0%（2500 题） |
| runbook | `run_all.sh` 全 skip dry-run 通过；`bash -n` 语法检查通过 |

## 新增脚本

- `pipeline_common.py` — prompt / tool schema / 模型加载（路径不存在自动走 ModelScope 下载）/ 答案比对 / IO
- `prepare_seeds.py` — Terminal 内置任务 + HLE 数学抽题（parquet 分片下载，不拉全量）
- `distill.py` — 自蒸馏：qa 批量生成（左 padding batch）/ agent 逐轮 Git Bash 沙箱执行；断点续跑
- `build_sft.py` — 过滤（答案比对 / 沙箱 check 重跑）+ 统一 messages
- `prepare_swe_asi.py` — 开源轨迹数据集格式适配
- `merge_experts.py` — TA / TIES 流式合并，`--freeze_router` 预留 MoE router 处理；`--self-test` 合成小矩阵手算验证
- `check_contamination.py` — n-gram 重叠检测（jsonl/parquet 评测集）
- `cloud/setup_env.sh` / `cloud/w2_pipeline.sh` — 云上环境准备与 W2 主流水线（SCALE=smoke/full；10-23 的 run_all.sh 已被本脚本取代删除）

## 踩坑与修复

| 问题 | 修复 |
|---|---|
| `safe_open` 无 `close()`，合并收尾崩溃导致产物缺 index/config | 改 `contextlib.ExitStack` 管理句柄 |
| 评测集是 parquet 非 jsonl | `check_contamination.py` 加 parquet 分支（question/problem 字段自适应） |
| 答案宽松匹配 `'1' in gt` 恒真 | contains 匹配要求两侧归一化后长度 ≥2 |
| ModelScope 数据集搜索返回裸名 | `RepoInfo.repo_id` 取全名（如 `deepmath/agent-sci-general`） |

## 上云待办（按序）

1. GPU 模式开机 → W1 补模型（cloud-w1-runbook）→ `docs/cloud-w2-runbook.md` 全流程（`w2_pipeline.sh`，云机路径已固化）
2. `distill.py` 换 35B 路径：3B 非 thinking 模型靠 prompt 包裹 `<think>`，35B 默认 thinking 模式会自然生成，prompt 兼容
3. SWE/ASI 真实规模轨迹生成需 Docker 执行环境（本机只验格式）
4. Terminal-Bench / SWE-bench Pro / ASI-Bench harness 锁 tag 拉取，补进 `setup_env.sh` 注释位
5. 35B 混合架构 `target_modules` 以 `docs/spike-35b-arch.md` 结论为准传入 `train_lora.py`

## 遗留项

- 3B 的 Terminal 轨迹质量不做要求（机制验证）；35B 版 system prompt 已加「done 前必须自检」行为约束，轨迹质量以 check 通过率为准报告
- HLE 种子全是数学题，与 HLE 跨领域分布有差距，35B 阶段需补其他领域题源

## 质量阀加固（2026-10-24，针对代码评审四问）

| 问题 | 修复 | 验证 |
|---|---|---|
| ① 判分误杀（LaTeX/空格） | `answers_equal` 改两层：快路归一化 + sympy 慢路（`convert_xor` + 隐式乘法 + simplify 判等价）；新增依赖 sympy | 11 组单测全过；50 条 raw_traj 回放通过率 40% → 72.4%（`$4`/`4`、`18,000`/`18000` 等误杀回收） |
| ② think 标签强耦合 | `generate` 解码不再 skip 特殊 token（Qwen3 的 `<think>` 是特殊 token 会被抹掉），只手动清聊天标记；`parse_qa_output` 降级：无标签时 ANSWER 行之前全文作 reasoning | 4 种输出场景（标准标签/异构标签/无标签/裸文本）解析均保住 reasoning；**上云第 0.5 步**写入 cloud-verify-checklist（35B 先跑 5 条人工看格式再放量） |
| ③ done 假阳性 + check 不可规模化 | 澄清：假 done 本就被 meta.check 硬门拦掉（27 done → 16 入训），危害是产量损失非数据污染；Terminal 任务模板化（10 模板，check 同构生成，参数来自安全字符池）+ done 前自检的 prompt 约束 | 生成 100 条：bash 语法全过、setup 全成功、check 无误通过（去重键修过一轮：常文模板按 check 区分参数） |
| ④ 沙箱能力边界 | `distill.py --sandbox auto/docker/local`：有云 Docker 时 agent 在 `--network none --memory 2g` 容器执行（真实防线），本机 Git Bash 仅 smoke；黑名单补 curl/wget/nc/ssh 等 | 本机 auto 探测 local 后端单任务端到端通过；runbook 接 `N_TERMINAL_GENERATED=300` 与镜像拉取 |

### 踩坑记录（本轮新增）

- sympy 默认变换不支持隐式乘法（`2ab` 解析失败）与 `^`（需 `convert_xor`），`_TRANSFORMS` 必须显式加
- `"x = 5"` 类答案需先取等号右侧再判等
- 模板去重按 task 文本会杀死常文模板（extract_column 只剩 1 条），去重键须含 check 命令

## 契约修复（2026-10-24，第二轮评审三坑）

| 坑 | 修复 |
|---|---|
| ① 训练/推理终止方式不一致：done 轮 JSON 被跳过，最后一轮训练成纯文本总结，与 system prompt「每轮只输出 JSON」矛盾，评测时模型可能永远走不到 done | `build_agent` 最后一轮改为训练模型输出 `{"explanation","done":true,"summary"}` JSON（`distill.py` done 轮加显式标记，`build_sft.py` 兼容旧轨迹回退识别）；已重建数据、重训 lora-terminal、重跑两法合并并冒烟通过 |
| ② 覆盖写无失败保护：沙箱目录被清理后 check 全 False，静默产出空训练文件照常退出 | `check_terminal` 改三态（通过/未通过/**无法检查**），无法检查与零样本双双非零退出（`--allow-unchecked` 可放行无法检查，但零样本仍拒绝写文件）；空输入同样报错 |
| ③ 输出路径靠 `records[0]` 推断，空文件会得到 `unknown.jsonl` | 改为从轨迹文件名推断（`Path(traj).stem`），空文件在推断前即报错退出 |

