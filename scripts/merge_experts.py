"""多专家合并：Task Arithmetic / TIES，流式按 tensor 处理（不整模型驻留内存）。

输入专家可以是两种形态（自动识别）：
- 已 merge 的独立权重目录（含 model.safetensors*）
- LoRA adapter 目录（含 adapter_config.json），此时 delta 直接由 B@A 计算，
  无需先把 adapter 合入底座 —— 云上 35B 省 70G+ 内存

合并公式（n = 专家数，Δ_i = expert_i - base）：
- task_arithmetic: merged = base + scaling * mean_i(Δ_i)
- ties: 每个 Δ_i 只保留 magnitude top-`density` 比例的元素；
  对每个元素做符号选举 s = sign(Σ_i sign(Δ̃_i))；
  merged = base + scaling * mean{Δ̃_i : sign(Δ̃_i) == s}

--freeze_router 跳过 MoE router 权重（key 含 .mlp.gate.，Qwen3.6 MoE 的 router 命名；
注意不匹配 dense 层的 gate_proj），为 35B 预留，3B dense 模型上空转。

输出：merged 权重目录（safetensors 分片 + 从 base 拷贝 config/tokenizer）+
merge_config.json（方法/超参/专家清单，进证据链）。

用法：
  python merge_experts.py --base <base> --experts e1 e2 e3 --method ties --density 0.2 --output out/
  python merge_experts.py --self-test     # 合成小矩阵单测，不加载模型
"""

import argparse
import json
import re
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROUTER_KEY_RE = re.compile(r"\.mlp\.gate\.")

# ---------------------------------------------------------------- 专家 delta


class ExpertDelta:
    """按 tensor key 提供 expert - base 的 delta。支持 merged 目录与 LoRA adapter。"""

    def __init__(self, path: str):
        self.path = Path(path)
        adapter_cfg = self.path / "adapter_config.json"
        self.is_lora = adapter_cfg.exists()
        if self.is_lora:
            cfg = json.loads(adapter_cfg.read_text(encoding="utf-8"))
            self.r = cfg.get("r", 16)
            self.alpha = cfg.get("lora_alpha", 32)
            self.lora_files = sorted(self.path.glob("*.safetensors"))
            if not self.lora_files:
                raise RuntimeError(f"No safetensors in adapter {self.path}")
        else:
            self.files = sorted(self.path.glob("model*.safetensors"))
            if not self.files:
                raise RuntimeError(f"No model safetensors in {self.path}")

    def _lora_delta(self, key: str, base_w: torch.Tensor) -> torch.Tensor | None:
        """从 adapter 文件计算某 key 的 LoRA delta：scale * B @ A。

        PEFT 的 adapter key = {模块路径}.lora_A/B.weight，其中模块路径是 base key
        去掉 .weight 后缀（base: ...q_proj.weight → adapter: ...q_proj.lora_A.weight）。
        前缀兼容 base_model.model.{模块} 与裸 {模块} 两种存储格式。
        """
        scale = self.alpha / self.r
        module = key[: -len(".weight")] if key.endswith(".weight") else key
        candidates = (f"base_model.model.{module}", module)
        for f in self.lora_files:
            with safe_open(str(f), framework="pt", device="cpu") as sf:
                keys = sf.keys()
                for cand in candidates:
                    b_key = f"{cand}.lora_B.weight"
                    a_key = f"{cand}.lora_A.weight"
                    if b_key in keys and a_key in keys:
                        b = sf.get_tensor(b_key).to(torch.float32)
                        a = sf.get_tensor(a_key).to(torch.float32)
                        return scale * (b @ a)
        return None

    def get(self, key: str, base_w: torch.Tensor) -> torch.Tensor | None:
        if self.is_lora:
            return self._lora_delta(key, base_w)
        for f in self.files:
            with safe_open(str(f), framework="pt", device="cpu") as sf:
                if key in sf.keys():
                    return sf.get_tensor(key) - base_w
        return None


# ---------------------------------------------------------------- 合并方法


def merge_tensor(base_w: torch.Tensor, deltas: list[torch.Tensor],
                 method: str, scaling: float, density: float) -> torch.Tensor:
    """单 tensor 合并。计算在 fp32 进行，返回保持 base_w 的 dtype
    （否则产物 bf16/fp32 混存，体积翻倍）。"""
    if method == "task_arithmetic":
        delta = torch.stack([d.to(torch.float32) for d in deltas]).mean(dim=0)
    elif method == "ties":
        stacked = torch.stack([d.to(torch.float32) for d in deltas])
        # 1) trimming：每个专家保留 magnitude top-density
        k = max(1, int(stacked[0].numel() * density))
        flat = stacked.view(stacked.shape[0], -1)
        thresh = flat.abs().kthvalue(flat.shape[1] - k + 1, dim=1, keepdim=True).values
        mask = flat.abs() >= thresh
        trimmed = flat * mask
        # 2) 符号选举
        signs = torch.sign(trimmed)
        vote = signs.sum(dim=0)
        elected = torch.sign(vote)
        elected[elected == 0] = 1  # 平局取正
        # 3) 只合并符号一致的专家，取均值
        agree = signs == elected.unsqueeze(0)
        denom = agree.sum(dim=0).clamp(min=1)
        delta = (trimmed * agree).sum(dim=0) / denom
        delta = delta.view_as(base_w)
    else:
        raise ValueError(f"unknown method: {method}")
    merged = base_w.to(torch.float32) + scaling * delta
    return merged.to(base_w.dtype)


# ---------------------------------------------------------------- 流式合并


def run_merge(base: str, experts: list[str], method: str, scaling: float,
              density: float, freeze_router: bool, output: str,
              shard_max_gb: float = 4.0):
    base_p = Path(base)
    expert_deltas = [ExpertDelta(e) for e in experts]
    kinds = ["lora" if e.is_lora else "merged" for e in expert_deltas]
    print(f"Merging {len(experts)} experts ({method}, scaling={scaling}, "
          f"density={density}, freeze_router={freeze_router})")
    for e, k in zip(experts, kinds):
        print(f"  expert: {e} [{k}]")

    out_p = Path(output)
    out_p.mkdir(parents=True, exist_ok=True)

    shard, shard_idx, shard_bytes = {}, 0, 0
    max_bytes = int(shard_max_gb * (1 << 30))
    n_router_frozen = 0
    n_matched = [0] * len(expert_deltas)  # 每个专家实际命中 delta 的 key 数

    def flush():
        nonlocal shard, shard_idx, shard_bytes
        if shard:
            # 先按序号写临时名，最后统一改名并生成 index
            save_file(shard, str(out_p / f"shard_tmp_{shard_idx}.safetensors"))
            shard_idx += 1
            shard = {}
            shard_bytes = 0

    files = sorted(base_p.glob("model*.safetensors"))
    from contextlib import ExitStack

    stack = ExitStack()
    opened = [stack.enter_context(safe_open(str(f), framework="pt", device="cpu"))
              for f in files]
    seen = set()
    n_keys = 0
    try:
        for f in opened:
            for key in f.keys():
                if key in seen:
                    continue
                seen.add(key)
                base_w = f.get_tensor(key)
                if freeze_router and ROUTER_KEY_RE.search(key):
                    # 冻结 = 保留 base 权重原样拷入产物，不是丢弃
                    shard[key] = base_w
                    shard_bytes += base_w.numel() * base_w.element_size()
                    n_router_frozen += 1
                    n_keys += 1
                    if shard_bytes >= max_bytes:
                        flush()
                    continue
                deltas = []
                for i, ed in enumerate(expert_deltas):
                    d = ed.get(key, base_w)
                    if d is not None:
                        n_matched[i] += 1
                        deltas.append(d)
                if not deltas:
                    merged = base_w  # 该 key 无专家修改（如视觉塔/未挂载层），原样拷贝
                else:
                    if len(deltas) < len(expert_deltas):
                        print(f"  WARN: key {key} missing from some experts, "
                              f"using {len(deltas)} deltas")
                    merged = merge_tensor(base_w, deltas, method, scaling, density)
                shard[key] = merged
                shard_bytes += merged.numel() * merged.element_size()
                n_keys += 1
                if shard_bytes >= max_bytes:
                    flush()
    finally:
        stack.close()
    flush()

    # 整体匹配率断言：任何一个专家命中 0 个 key，说明 adapter 与 base
    # 完全对不上（前缀/结构不兼容），继续跑只会产出"等于 base"的假产物
    for e, k, m in zip(experts, kinds, n_matched):
        if m == 0:
            raise RuntimeError(
                f"FATAL: expert {e} matched 0 keys against base — adapter 与 "
                f"base 结构不兼容，拒绝产出假合并结果")
        print(f"  matched: {e} [{k}] {m}/{n_keys} keys")

    # 统一改名为 model-XXXXX-of-YYYYY.safetensors 并写 index
    shards = sorted(out_p.glob("shard_tmp_*.safetensors"))
    total = len(shards)
    index = {"metadata": {"total_size": 0}, "weight_map": {}}
    for i, sp in enumerate(shards):
        name = f"model-{i:05d}-of-{total:05d}.safetensors"
        sp.rename(out_p / name)
        with safe_open(str(out_p / name), framework="pt", device="cpu") as sf:
            for key in sf.keys():
                index["weight_map"][key] = name
    index["metadata"]["total_size"] = sum(
        (out_p / f"model-{i:05d}-of-{total:05d}.safetensors").stat().st_size
        for i in range(total)
    )
    (out_p / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2), encoding="utf-8")

    # config / tokenizer 从 base 拷贝
    for fname in ("config.json", "generation_config.json", "tokenizer.json",
                  "tokenizer_config.json", "vocab.json", "merges.txt",
                  "special_tokens_map.json", "added_tokens.json"):
        src = base_p / fname
        if src.exists():
            shutil.copy2(src, out_p / fname)

    merge_cfg = {
        "method": method,
        "scaling": scaling,
        "density": density,
        "freeze_router": freeze_router,
        "base": base,
        "experts": experts,
        "expert_kinds": kinds,
        "n_tensors": n_keys,
        "router_frozen": n_router_frozen,
        "expert_matched_keys": n_matched,
    }
    (out_p / "merge_config.json").write_text(
        json.dumps(merge_cfg, indent=2), encoding="utf-8")
    print(f"Merged model saved to {out_p} ({n_keys} tensors, "
          f"{n_router_frozen} router tensors frozen)")


# ---------------------------------------------------------------- 自测


def self_test():
    """合成小矩阵验证两种方法 against 手算结果。"""
    torch.manual_seed(0)
    base = torch.randn(4, 8)

    # 专家1：delta 全正；专家2：部分元素反向
    d1 = torch.tensor([[1.0, -2, 0.5, 0, 0, 0, 0, 3]]).repeat(4, 1)
    d2 = torch.tensor([[0.5, 2, 0.1, 0, 0, 0, 0, -3]]).repeat(4, 1)

    # --- task arithmetic, scaling=1: base + mean(d1, d2)
    got = merge_tensor(base, [d1, d2], "task_arithmetic", 1.0, 0.2)
    want = base + (d1 + d2) / 2
    assert torch.allclose(got, want, atol=1e-5), "task_arithmetic mismatch"
    print("task_arithmetic OK")

    # --- ties, density=0.25 (每专家保留 8 个里 top2), scaling=1
    got = merge_tensor(base, [d1, d2], "ties", 1.0, 0.25)
    # 手算：d1 top2 = -2 (col1), 3 (col7)；d2 top2 = 2 (col1), -3 (col7)
    # col1: signs -1, +1 → 平局 vote=0 → elected=+1 → 只有 d2 一致 → delta=2
    # col7: signs +1, -1 → 平局 → +1 → d1 → delta=3
    # 其余元素被 trim 掉 → delta=0
    want_delta = torch.zeros(4, 8)
    want_delta[:, 1] = 2.0
    want_delta[:, 7] = 3.0
    assert torch.allclose(got, base + want_delta, atol=1e-5), \
        f"ties mismatch:\n{got - base}"
    print("ties OK")

    # --- ties 非平局：三专家两正一负
    d3 = torch.tensor([[1.0, 0, 0, 0, 0, 0, 0, 0]]).repeat(4, 1)
    d4 = torch.tensor([[3.0, 0, 0, 0, 0, 0, 0, 0]]).repeat(4, 1)
    d5 = torch.tensor([[-9.0, 0, 0, 0, 0, 0, 0, 0]]).repeat(4, 1)
    got = merge_tensor(base, [d3, d4, d5], "ties", 1.0, 0.25)
    # col0: vote = 1+1-1 = 1 > 0 → elected +1 → mean(1,3)=2
    want_delta = torch.zeros(4, 8)
    want_delta[:, 0] = 2.0
    assert torch.allclose(got, base + want_delta, atol=1e-5), "ties vote mismatch"
    print("ties sign-election OK")

    # --- freeze_router 正则自检
    assert ROUTER_KEY_RE.search("model.layers.3.mlp.gate.weight")
    assert not ROUTER_KEY_RE.search("model.layers.3.mlp.gate_proj.weight")
    print("router regex OK")

    print("ALL SELF-TESTS PASSED")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=None)
    parser.add_argument("--experts", nargs="+", default=None)
    parser.add_argument("--method", choices=["task_arithmetic", "ties"],
                        default="task_arithmetic")
    parser.add_argument("--scaling", type=float, default=1.0)
    parser.add_argument("--density", type=float, default=0.2,
                        help="TIES 保留比例")
    parser.add_argument("--freeze_router", action="store_true")
    parser.add_argument("--output", default=None)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return
    if not args.base or not args.experts or not args.output:
        parser.error("--base/--experts/--output are required (or use --self-test)")
    run_merge(args.base, args.experts, args.method, args.scaling,
              args.density, args.freeze_router, args.output)


if __name__ == "__main__":
    main()
