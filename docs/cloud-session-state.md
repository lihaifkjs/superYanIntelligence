# 云机会话状态快照（2026-10-24 暂停点）

> 用户手动关机暂停。恢复时按本文「恢复步骤」执行，SSH 信息若变化以 credentials.md 为准。

## 机器

- 实例：AutoDL A800专区 003机（98fc43be4e），1×A800-80GB，¥5.59/时，数据盘 50GB 固定
- 状态：**已关机（用户操作）**，数据保留约 15 天
- 模型源：67GB 已下载，跨 `/dev/shm/q36_shards`(51G) + `/root/autodl-tmp/q36_shards`(17G)，统一视图 `/root/autodl-tmp/q36`

## 已验证通过（勿重做）

| 项 | 结果 | 证据 |
|---|---|---|
| bnb 4bit / bf16 加载 35B | ✅ 14s | quant.log |
| thinking 生成 | ✅ 格式正常 | quant.log gen output |
| LoRA 训练冒烟（bf16 全精度） | ✅ 91s/13步 loss 0.72 | train_smoke.log，adapter 在 `/root/autodl-tmp/lora-smoke/final` |
| MoE aux_loss | ✅ 正常记录 | train_smoke.log |

## 关键决策修正（与旧文档不同）

- ~~4bit 量化省磁盘~~ → **废弃**：routed experts（32B/35B 参数）是批量参数非 Linear，bnb 量化不到，省 1.3GB 无意义
- **改为 bf16 全精度训练**：显存 67GB 模型 + 余量 OK；只存 LoRA adapter（42MB/个），50GB 盘够用
- fp16 模型源全程保留在磁盘，训练/评测直接从 `/root/autodl-tmp/q36` 加载
- `prepare_model_for_kbit_training` 在 bf16 模式禁用（会把全参数 cast fp32 导致 OOM），改手动冻结+梯度检查点

## 待办（恢复后顺序）

1. **重装 vLLM**：关机时 pip 停在 torch 2.13 下载阶段（环境未损坏，torch 2.8.0 完好）
   `pip install vllm==0.32.0`（会升级 torch 2.13，训练环境需快速复验）
2. **vLLM 起服务**：`vllm serve /root/autodl-tmp/q36 --served-model-name qwen35b --max-model-len 16384 --gpu-memory-utilization 0.97 --enforce-eager --port 8000`
   - 悬念：vllm 0.32 是否支持 qwen3_5_moe 架构；67GB 权重挤 80G 卡
   - 备选：transformers 直跑（慢但稳）
3. **HLE baseline**：`/root/hle_baseline.py --parquet /root/autodl-tmp/hle/datasets/caiss--hle/snapshots/master/data/test-00000-of-00001.parquet --n 10` 先冒烟再全量
4. 训练冒烟复验（torch 2.13 后）：重跑 13 步确认
5. 自蒸馏数据生成（四方向）→ W2 正式训练

## 环境备忘

- 云机 Python：`/root/miniconda3/bin/python`（3.12），pip 镜像已配阿里云
- 已装：transformers 5.20.2 / peft 0.21.2 / trl 1.16.0 / bitsandbytes 0.50.2 / accelerate 1.15.0 / modelscope 1.40.2 / datasets 5.1.0
- 线性注意力在跑 PyTorch 参考内核（慢 2-5 倍），`causal-conv1d` 源码编译慢已弃，可再试 `flash-linear-attention` 提速
- SSH 走 IP 直连（域名 getaddrinfo 失败）：`--host 36.137.238.251`；AutoDL 有连接限流，命令间隔 ≥1 分钟

## 未解：HF 真 token

- 现状：4 个 token 均被真站 401；浏览器走 VPN（fake-ip）路径异常
- 方案：hosts 写 `18.65.14.87 huggingface.co` + 关 VPN + flushdns → 真站注册/登录 → Write token
- 验证：`curl -H "Authorization: Bearer <token>" https://huggingface.co/api/whoami`
