# W1 Spike：Qwen3.6-35B-A3B 架构适配验证（E1）

> 日期：2026-10-23　结论：**通过**——transformers + PEFT 原生支持，无需改代码

## 模型事实（来源：`models/Qwen3.6-35B-A3B/config.json`）

- 架构 `Qwen3_5MoeForConditionalGeneration`（model_type `qwen3_5_moe`），transformers 5.20.2 原生识别，无 remote code
- 40 层：linear_attention × 30（Gated DeltaNet）+ full_attention × 10（每 4 层 1 个全注意力）
- MoE：256 routed experts（批量参数存储 `gate_up_proj (256,1024,2048)` / `down_proj (256,2048,512)`）+ shared expert + TopKRouter
- MTP 层 ×1（推测解码用）；多模态 ViT（depth 27），训练评测走 text-only
- hidden 2048，vocab 248320，bf16；权重 67GB（26 shards）
- 采样参数与 PRD 4.1 一致：`temp=1.0, top_p=0.95, top_k=20`；EOS `[248046, 248044]`

## LoRA 挂载验证（meta device 实测）

- **可挂**：全注意力 `q/k/v/o_proj`（10 层）、线性注意力 `in_proj_qkv/z/a/b, out_proj`（30 层）、shared expert `gate/up/down_proj`（40 层）
- r=16 / alpha=32 下可训练参数 **21.2M**（620 个 LoRA tensor）
- **挂不上**：routed experts（批量参数，非 Linear，PEFT 按名挂载失效）、router（裸 Parameter）→ 要动 routed expert 需自定义 LoRA 实现
- 影响：PRD「router 默认冻结」与此天然一致；shared expert + attention 的容量是否足够是 W2 spike 点

## Chat Template

- `enable_thinking` 开关：默认开，generation prompt 以 `<think>\n` 开头 → 训练数据带 `<think>...</think>` 与之一致
- 多模态 token 处理在模板内，text-only 数据无影响

## 遗留验证（上云第一件事）

- bnb 4-bit 在该混合架构的真实加载与显存占用（本机 12G 不够，A800 上验）：`load_in_4bit` + GatedDeltaNet 内核是否兼容 bitsandbytes
- 4-bit 量化脚本：网盘流式读 fp16 → 量化 → 本地 20GB（对应 50GB 数据盘方案）
