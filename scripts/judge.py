"""判分与 QA 输出解析：数据过滤与评测共用的同一把尺子。

分层判等（answers_equal）：
1. 快路：字符串归一化（LaTeX 分数展开、等号取右、去格式符号）
2. 数值近似（77 vs 77.0）
3. 非平凡子串匹配
4. 慢路：sympy 数学等价（1/4 vs 0.25、(a+b)^2 vs 展开式）

HLE 评测脚本（scripts/hle/run_hle_smoke.py、scripts/cloud/hle_baseline.py）
的 exactMatch 判分必须从这里 import，保持口径一致。
"""

import re

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
ANSWER_RE = re.compile(r"ANSWER\s*:\s*(.+)", re.IGNORECASE)


def parse_qa_output(text: str) -> dict:
    """从 qa 模式生成结果中拆出 reasoning 与 final answer。

    格式不依赖 <think> 标签：35B 原生 thinking 可能用别的标记或裸输出，
    此时 ANSWER 行之前的全部文本都算 reasoning（SFT 侧再统一包装格式）。

    扫描范围是全文（模型可能把 ANSWER 行写进 think 块内的计划区）；
    跳过内容含 "final answer" 的占位符回声（模型复述格式指令产生的
    "ANSWER: <final answer>" 不算答案）。
    """
    text = text.strip()
    m = THINK_RE.search(text)
    reasoning = m.group(1).strip() if m else ""

    am = None
    for cand in ANSWER_RE.finditer(text):
        if "final answer" in cand.group(1).lower():
            continue
        am = cand  # 取最后一个有效 ANSWER: 行
    if am:
        answer = am.group(1).strip()
        if not reasoning:
            # 无 think 标签：ANSWER 行之前的内容整体作为推理过程
            reasoning = text[:am.start()].strip()
    else:
        # 退化处理：think 外（无 think 则全文）的最后一行当答案，其余当推理
        rest = THINK_RE.sub("", text).strip() if m else text
        lines = [l for l in rest.splitlines() if l.strip()]
        answer = lines[-1].strip() if lines else ""
        if not reasoning:
            reasoning = "\n".join(lines[:-1]).strip()
    return {"reasoning": reasoning, "answer": answer, "raw": text}


def normalize_answer(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"\\boxed\{([^}]*)\}", r"\1", s)
    s = s.replace("$", "")
    # "x = 5" / "answer: 5" 取等号右侧
    if "=" in s:
        s = s.split("=")[-1]
    # \frac{a}{b} -> a/b（先展开分数，再清理其余 latex 命令）
    for _ in range(3):  # 至多三层嵌套
        s = re.sub(r"\\[dt]?frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", s)
    # frac 展开产生的纯数字括号对折叠：-(1)/(2) 与 -1/2 同义
    s = re.sub(r"\((-?[\d.]+)\)/\((-?[\d.]+)\)", r"\1/\2", s)
    # 一元正号归一：+infty/+x 与 infty/x 同义（[2,+infty) vs [2, infty)）
    s = re.sub(r"\+\s*(?=[a-zA-Z(\\])", "", s)
    s = re.sub(r"\\(text|mathrm|mathbf|operatorname)\{([^{}]*)\}", r"\2", s)
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\times", "*").replace("\\cdot", "*").replace("\\div", "/")
    s = s.replace("\\", "").replace("{", "").replace("}", "")
    s = re.sub(r"\s+", " ", s).strip().strip(".")
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)  # 千分位逗号
    return s


# sympy 慢路：数学等价判断（缺 sympy 时自动退化为只有快路）
try:
    from sympy.parsing.sympy_parser import (
        convert_xor,
        implicit_multiplication_application,
        parse_expr,
        standard_transformations,
    )

    _TRANSFORMS = (standard_transformations
                   + (convert_xor, implicit_multiplication_application))

    def _sympy_equal(p: str, g: str) -> bool:
        """把两侧当数学表达式解析，判断等价。解析失败返回 False。"""
        try:
            ep = parse_expr(p, transformations=_TRANSFORMS)
            eg = parse_expr(g, transformations=_TRANSFORMS)
        except Exception:  # noqa: BLE001 —— 非数学文本（人名/单词答案）走不到等价
            return False
        if ep == eg:
            return True
        try:
            return bool((ep - eg).simplify() == 0)
        except Exception:  # noqa: BLE001
            return False
except ImportError:  # pragma: no cover
    def _sympy_equal(p: str, g: str) -> bool:
        return False


def _looks_mathy(s: str) -> bool:
    """启发式：含数字或运算符才值得送 sympy，避免 'mike' 之类误解析。"""
    return bool(re.search(r"[0-9]|[-+*/^=]", s))


def answers_equal(pred: str, gt: str) -> bool:
    p, g = normalize_answer(pred), normalize_answer(gt)
    if not p or not g:
        return False
    if p == g:
        return True
    # 纯空格/排版差异（数学式去空格再比一次）
    if _looks_mathy(p) and p.replace(" ", "") == g.replace(" ", ""):
        return True
    # 数值近似比较（容忍 77 vs 77.0）
    try:
        pf, gf = float(p.replace(",", "")), float(g.replace(",", ""))
        return abs(pf - gf) < 1e-4
    except ValueError:
        pass
    # gt 包含 pred 的宽松匹配（限两侧都非平凡，避免 '1' 命中一切）
    if min(len(p), len(g)) >= 2 and (p in g or g in p):
        return True
    # sympy 数学等价（1/4 vs 0.25、2x+1 vs 2*x+1、(a+b)^2 展开式等）
    if _looks_mathy(p) and _looks_mathy(g):
        return _sympy_equal(p, g)
    return False
