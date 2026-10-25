# Agent 工作规约
> 适用：本md文档用于指引你帮我完成既定目的，是核心指导，这里包含项目大纲和注意事项，你应该严格遵守，且如果需求有改变或者调整或任何项目应该注意的事你都应该考虑在本文档记录相关信息，本文档要求精简有效。


# 文档地图
（有新的相关文档，补充在这里，并用一句话介绍文档内容）

- [`task.md`](task.md)：赛题原文——基于 Qwen3.6-35B-A3B 训练四个 Benchmark（SWE-bench Pro v2-hard / HLE / ASI-Bench / Terminal-Bench）方向专家模型，探索 Model Merging 合版方法，目标宏平均较 Base 提升 3 个百分点。
- [`README.md`](README.md)：仓库门面——地图、本机/云上 quickstart、当前状态；交付物要求的根 README。
- [`PRD.md`](PRD.md)：实施方案——4 专家（每 benchmark 一个）QLoRA 训练 + 合版对比（TIES/DARE 等）+ 四榜评测的完整决策、里程碑、成本与风险；待定项在 W1 spike 定。
- [`docs/w1-progress.md`](docs/w1-progress.md)：W1 进展——本地 3B QLoRA pipeline 跑通记录、踩坑修复（trl API / ModelScope 镜像 / Windows 编码）、剩余缺口与下一步选项。
- [`docs/w2-progress.md`](docs/w2-progress.md)：W2 本机 3B 预演——全链路（种子→蒸馏→过滤→训练→TA/TIES 合并）10/10 验证通过、新脚本清单、上云待办。
- [`docs/gpu-rental-notes.md`](docs/gpu-rental-notes.md)：云 GPU 租用踩坑——2026-10-23 AutoDL/优云智算/阿里云 DSW 实况、80G 显存红线、租赁时段规律与价格锚点。
- [`docs/spike-35b-arch.md`](docs/spike-35b-arch.md)：W1 架构 spike 结论——Qwen3.6-35B-A3B 混合架构 LoRA 挂载方案、MoE 专家/Router 处理、待上云验证项。
- [`docs/cloud-w1-runbook.md`](docs/cloud-w1-runbook.md)：W1 云机操作 Runbook（交接独立云 agent）——输入清单、做什么/不做什么、分 Phase 命令与验收、坑点速查、W1 出口对照，配 scripts/cloud/w1_*.sh。
- [`docs/cloud-w2-runbook.md`](docs/cloud-w2-runbook.md)：W2 云机操作 Runbook——第 0 步蒸馏格式验证、35B 自蒸馏×4 + HLE 专家端到端的分 Phase 命令与验收、W2 特有风险坑点，配 scripts/cloud/w2_pipeline.sh（SCALE=smoke/full）。
- [`docs/cloud-ops-log.md`](docs/cloud-ops-log.md)：云机操作台账——003 机已执行操作的时间线记录（含是否需重做标记）、重启失效清单、追加模板，防止跨 agent 重复劳动。
- [`docs/hf-token-blocked.md`](docs/hf-token-blocked.md)：HF token 受阻记录（2026-10-25 决策）——真站被 GFW+VPN 客户端镜像劫持双重封锁的全路径排查、真伪判别方法、走 hf-mirror/ModelScope 的决策与 token 恢复选项。

## 任务通知（必须照做）

用户经常不在电脑前。**任务完成时、卡住时，必须推微信通知**，否则他不知道进展或你正在等他。

### 何时推

- **任务完成**：目标达成 / 本轮交付完毕 → 推完再写最终回复
- **卡住**：要用户点审批、缺只有他能给的信息、同一问题失败 2 次以上、要在方案间做选择
  → **在弹审批之前推**，然后才等
- 不推中间步骤、单步进度、已自行修好的错误；同一事件只推一次

### 怎么推

在工作区根目录执行（脚本只用 Python 标准库，3.8+ 任意解释器可跑）：

```bash
# macOS / Linux
python3 tools/notify.py -t "[DSH] 完成 · 任务简称" -c "结果: 一句话
时间: 2026-10-17 23:05"
```

```powershell
# Windows PowerShell（PATH 里通常没有 python，用 DSH 运行时）
$py = "$env:DSH_HOME\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe"
& $py tools\notify.py -t "[DSH] 完成 · 任务简称" -c "结果: 一句话
时间: 2026-10-17 23:05"
```

### 三条红线

1. 正文**必须带时间戳**（否则平台按"相同内容 1 小时最多 3 条"把消息吞掉）
2. 推送失败**不能阻塞任务**，也不要为发通知申请提权
3. `tools/notify.config.json` 里的 token 不打印、不粘贴到会被分享的文件

**其余细节（失败处理、额度、环境坑、消息模板、排查顺序）全在 [`tools/NOTIFY.md`](tools/NOTIFY.md)，需要时再读。改动只改那一份。**

# 需求变更记录
- 2026-10-08：砍掉「重新生成 / 编辑重发」功能（原 PRD 3.1），相关接口（/api/chat/regenerate、消息编辑截断）不再实现。
- 2026-10-11：mock 策略降级——自下一里程碑（M7）起 mock 只留最小桩（返回写死的假数据保证页面可渲染，不复刻后端行为逻辑），行为流程统一在 real 联调时验证；既有高保真 mock 代码不返工。
- 2026-10-11：砍掉 M6 参数实验场（代码与 /lab 页面已拆除；experiments 两表 schema 保留以维持 user_version 迁移链）。理由：全局参数单例 + 多窗口手工对比已覆盖主要诉求，独立页面教学价值不足。
- 2026-10-14：验证强度下调——开发过程不做多轮验证，尤其前端仅要求 `npm run build` 通过，页面效果不专门验证；问题出现后再修。M9 起生效。
- 2026-10-22：修 M9 工具结果注入——单条结果超 `tool_inject_max_tokens` 时由「整条丢弃」改为按剩余预算二分截断正文后注入（原先天气类长正文会导致 tool response 落成「搜索无结果」，模型照实拒答；见 docs/后端m9实现.md 坑点修正）。

# 角色定位
你是一个专业的智能助手，通常你在计算机领域有很强的能力，同时你也精通其他领域。你需要协助开发一些项目。在这些项目中你一般需要和我讨论思路构建prd，技术方案，架构设计。以及完成代码的编写。

# 代码规范
1. Git 规范
- 所有代码变更必须给出 git commit 建议，commit message 使用约定式提交（type(scope): message）
- 新增/修改代码时，提示需要新增哪些文件、哪些文件需要diff
- 重要节点建议创建tag，例如 v0.1-hardware-implement

2. 文档驱动开发
- 任何功能设计、代码实现前，必须对齐文档；不能自行新增文档中没有定义的功能；如果需求冲突 / 模糊，优先提问确认，不要自行假设。你可以查看相关文档，他们的目录记录在“文档地图”中。
- 文档篇幅要求在能理解的基础上做精简：保留决策、契约、坑点与验证结论等关键信息，不堆砌过程性描述。

3. 代码质量
- 代码需要保证高内聚，低耦合。
- 需要有良好的可读性，特别注意一些涉及到数学或者整体架构的地方，你应该用注释标注。对于较简单的部分则可以不用注释而是通过代码本身告知意图，例如变量名通俗易懂。
- 提供测试验证方法
4. 测试与联调强度
- 测试聚焦主功能冒烟：验证核心路径可用即可，不追求边界用例全覆盖。
- 控制测试与联调的时间投入，不在这上面花过多精力；边缘情况以记录遗留项的方式处理。

## 禁止行为
- 禁止输出大段文本（超过1000字）除非征得同意
- 不要在没有确认前提条件的情况下直接给出最终方案；遇到模糊点优先提问澄清。特别是代码，如果没有得到我的同意直接给出代码是一定要禁止的。
- 禁止不经过我的同意翻看文档。编辑代码的时候如果有需要查看相关代码也需要经过我的同意才能翻看，禁止直接查看。
