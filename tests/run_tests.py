"""核心逻辑回归测试（stdlib，无 pytest 依赖）。

用法：
  python tests/run_tests.py

覆盖最易错的纯逻辑：判分（judge.answers_equal）、QA 解析（parse_qa_output）、
agent 动作解析（distill.parse_agent_action 两种形状）、合并数学（merge_tensor）、
router 正则。模型/数据相关链路由各脚本自带 smoke 覆盖。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from judge import answers_equal, parse_qa_output  # noqa: E402

FAILED = 0


def check(name, got, want=True):
    global FAILED
    if got != want:
        FAILED += 1
        print(f"FAIL {name}: got {got!r}, want {want!r}")
    else:
        print(f"ok   {name}")


# ---------------- judge.answers_equal ----------------

JUDGE_CASES = [
    (r"\frac{1}{4}", "0.25", True),
    (r"\dfrac{1}{4}", "1/4", True),
    ("2x + 1", "2x+1", True),
    (r"2 \times 3", "6", True),
    (r"\boxed{77}", "77", True),
    (r"(a+b)^2", "a^2+2ab+b^2", True),
    ("77", "78", False),
    ("Mike", "Mike", True),
    ("Mike", "Jerry", False),
    ("x = 5", "5", True),
    (r"\frac{2}{3}", "0.5", False),
    ("1,000", "1000", True),
    ("", "5", False),
]
for i, (p, g, want) in enumerate(JUDGE_CASES):
    check(f"answers_equal#{i} ({p!r},{g!r})", answers_equal(p, g), want)

# ---------------- parse_qa_output ----------------

r = parse_qa_output("reasoning here\nANSWER: 42")
check("qa.no-tag.reasoning", r["reasoning"], "reasoning here")
check("qa.no-tag.answer", r["answer"], "42")

r = parse_qa_output("<think>deep</think>\n\nANSWER: 7")
check("qa.tag.reasoning", r["reasoning"], "deep")
check("qa.tag.answer", r["answer"], "7")

r = parse_qa_output("一些推理。\n最终答案是 99")
check("qa.bare.answer", r["answer"], "最终答案是 99")
check("qa.bare.reasoning", r["reasoning"], "一些推理。")

# ---------------- distill.parse_agent_action ----------------

from distill import parse_agent_action  # noqa: E402

a = parse_agent_action('{"explanation": "x", "command": "ls"}')
check("agent.bare", a["command"], "ls")

a = parse_agent_action(
    '{"name": "run_command", "arguments": {"command": "pwd"}}')
check("agent.toolcall", a and a.get("command"), "pwd")

a = parse_agent_action('{"done": true, "summary": "ok"}')
check("agent.done", bool(a and a.get("done")), True)

check("agent.garbage", parse_agent_action("no json here"), None)

# ---------------- merge_tensor ----------------

import torch  # noqa: E402

from merge_experts import ROUTER_KEY_RE, merge_tensor  # noqa: E402

base = torch.randn(4, 8, dtype=torch.bfloat16)
d1 = torch.tensor([[1.0, -2, 0.5, 0, 0, 0, 0, 3]]).repeat(4, 1)
d2 = torch.tensor([[0.5, 2, 0.1, 0, 0, 0, 0, -3]]).repeat(4, 1)

got = merge_tensor(base, [d1, d2], "task_arithmetic", 1.0, 0.2)
check("merge.ta.dtype", got.dtype, torch.bfloat16)  # 产物 dtype 必须跟随 base
check("merge.ta.value",
      torch.allclose(got.float(), base.float() + (d1 + d2) / 2, atol=1e-2))

got = merge_tensor(base, [d1, d2], "ties", 1.0, 0.25)
delta = (got - base).float()
want_delta = torch.zeros(4, 8)
want_delta[:, 1] = 2.0
want_delta[:, 7] = 3.0
check("merge.ties.value", torch.allclose(delta, want_delta, atol=1e-2))
check("merge.ties.dtype", got.dtype, torch.bfloat16)

check("router.re.hit", bool(ROUTER_KEY_RE.search("m.layers.3.mlp.gate.weight")))
check("router.re.miss_gateproj",
      not ROUTER_KEY_RE.search("m.layers.3.mlp.gate_proj.weight"))

# ---------------- seed_bank 模板自检 ----------------

from seed_bank import build_generated_terminal  # noqa: E402

seeds = build_generated_terminal(50, 42)
check("seeds.count", len(seeds), 50)
check("seeds.unique", len({s["task"] + s["meta"]["check"] for s in seeds}), 50)
check("seeds.have_check", all("check" in s["meta"] for s in seeds))

print()
if FAILED:
    print(f"{FAILED} FAILURES")
    sys.exit(1)
print("ALL TESTS PASSED")
