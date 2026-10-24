"""轨迹过滤 + 统一 SFT 数据构建。

qa（HLE）：答案与 ground_truth 比对，只保留答对且带推理过程的。
agent（Terminal）：只保留 finished 的轨迹，并在原沙箱目录重跑 meta.check 确认任务真的完成。

输出 data/sft/{expert}.jsonl，每行 {"messages": [...]}：
- qa 型：system / user / assistant(<think> + answer)，与 W1 mock 格式一致
- agent 型：system / user / assistant(tool_calls) / tool ... / assistant(done JSON)。
  最后一轮必须是 {"done": true, ...} 的 JSON 文本——与 SYSTEM_PROMPT_AGENT 的
  「每轮只能输出 JSON」一致；如果训练成纯文本总结，评测时模型永远走不到 done 分支。

失败保护：check 依赖沙箱目录，目录缺失/执行异常记为「无法检查」并以非零退出
（--allow-unchecked 可放行）；空输入或过滤后零样本也报错退出，防止静默产出空文件。

用法：
  python build_sft.py --traj data/raw_traj/hle.jsonl
  python build_sft.py --traj data/raw_traj/terminal.jsonl
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from judge import answers_equal
from io_utils import read_jsonl
from prompts import SYSTEM_PROMPT_AGENT, SYSTEM_PROMPT_QA, TOOL_NAME

KNOWN_EXPERTS = ("terminal", "hle", "swe", "asi")


def build_qa(traj: dict) -> dict | None:
    if not traj.get("reasoning"):
        return None
    gt = traj.get("ground_truth")
    if gt and not answers_equal(traj["answer"], gt):
        return None
    content = f"<think>\n{traj['reasoning']}\n</think>\n\n{traj['answer']}"
    return {"messages": [
        {"role": "system", "content": SYSTEM_PROMPT_QA},
        {"role": "user", "content": traj["task"]},
        {"role": "assistant", "content": content},
    ]}


def check_terminal(traj: dict) -> bool | None:
    """True=检查通过，False=未通过，None=无法检查（沙箱缺失/执行异常）。

    None 必须与 False 区分：前者是基础设施问题（重跑会全灭），不是轨迹质量问题。
    """
    check = traj.get("meta", {}).get("check")
    if not check:
        return True
    workdir = Path(traj.get("sandbox_dir", ""))
    if not workdir.is_dir():
        return None
    try:
        proc = subprocess.run(
            ["bash", "-c", check], cwd=workdir,
            capture_output=True, timeout=30, encoding="utf-8", errors="replace",
        )
        return proc.returncode == 0
    except Exception:  # noqa: BLE001
        return None


def _find_done_turn(traj: dict) -> dict | None:
    """定位 done 轮。新轨迹有显式标记；旧轨迹兼容：command 为空且 exit_code=0。"""
    for turn in traj.get("turns", []):
        if turn.get("done"):
            return turn
    for turn in traj.get("turns", []):
        if turn.get("command") is None and turn.get("exit_code") == 0:
            return turn
    return None


def build_agent(traj: dict) -> dict | None:
    if not traj.get("finished"):
        return None
    done_turn = _find_done_turn(traj)
    if done_turn is None or not traj.get("final_summary"):
        return None
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_AGENT},
        {"role": "user", "content": traj["task"]},
    ]
    call_i = 0
    for turn in traj.get("turns", []):
        cmd = turn.get("command")
        if cmd is None:
            continue  # done 轮单独处理；解析失败轮不进训练数据
        call_id = f"call_{call_i}"
        call_i += 1
        messages.append({
            "role": "assistant",
            "content": turn.get("explanation") or None,
            "tool_calls": [{
                "id": call_id,
                "type": "function",
                "function": {
                    "name": TOOL_NAME,
                    "arguments": json.dumps({"command": cmd}),
                },
            }],
        })
        messages.append({
            "role": "tool",
            "tool_call_id": call_id,
            "name": TOOL_NAME,
            "content": f"exit_code={turn['exit_code']}\n{turn['output']}",
        })
    if len(messages) < 4:  # 至少一轮工具调用才有训练价值
        return None
    # 最后一轮：训练模型输出 done JSON（与 system prompt 的格式契约一致），
    # 而不是纯文本总结——否则评测时模型走不到 done 分支
    done_msg = {"explanation": done_turn.get("explanation", ""),
                "done": True,
                "summary": traj["final_summary"]}
    messages.append({"role": "assistant",
                     "content": json.dumps(done_msg, ensure_ascii=False)})
    return {"messages": messages}


def validate_traj(r: dict) -> str | None:
    """轻量契约校验，返回错误描述（None=合法）。防止隐式结构漂移静默产出垃圾。"""
    for k in ("id", "expert", "task"):
        if k not in r:
            return f"missing key {k!r}"
    if r.get("expert") == "terminal":
        if not isinstance(r.get("turns"), list):
            return "terminal traj missing 'turns' list"
        if not r.get("finished"):
            return None  # 未完成轨迹走正常过滤，不算结构错误
        if _find_done_turn(r) is None:
            return "finished traj has no done turn"
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj", required=True, help="raw_traj jsonl")
    parser.add_argument("--expert", choices=KNOWN_EXPERTS, default=None,
                        help="显式指定分支（terminal 走 check+tool_calls，其余走 qa）；"
                             "默认从轨迹文件名推断，不在已知列表则报错")
    parser.add_argument("--output", default=None, help="默认 data/sft/{expert}.jsonl")
    parser.add_argument("--allow-unchecked", action="store_true",
                        help="允许存在「无法检查」的轨迹时不报错退出（默认报错）")
    args = parser.parse_args()

    traj_path = Path(args.traj)
    records = read_jsonl(traj_path)
    if not records:
        print(f"ERROR: {traj_path} is empty", file=sys.stderr)
        sys.exit(1)

    if args.expert:
        expert = args.expert
    else:
        stem = traj_path.stem
        if stem not in KNOWN_EXPERTS:
            print(f"ERROR: cannot infer expert from filename {traj_path.name!r} "
                  f"(known: {KNOWN_EXPERTS}); pass --expert explicitly",
                  file=sys.stderr)
            sys.exit(1)
        expert = stem
    if args.output is None:
        args.output = f"data/sft/{expert}.jsonl"

    kept, n_pass, n_unchecked, n_invalid = [], 0, 0, 0
    for r in records:
        err = validate_traj(r)
        if err:
            n_invalid += 1
            print(f"  WARN: traj {r.get('id', '?')} invalid: {err}", file=sys.stderr)
            continue
        if expert == "terminal":
            chk = check_terminal(r)
            if chk is None:
                n_unchecked += 1
                continue
            if chk:
                n_pass += 1
                item = build_agent(r)
            else:
                item = None
        else:
            item = build_qa(r)
            if item:
                n_pass += 1
        if item:
            kept.append(item)

    print(f"traj={len(records)}  filter_pass={n_pass}  unchecked={n_unchecked}  "
          f"invalid={n_invalid}  sft_kept={len(kept)}")
    print(f"filter_pass_rate={n_pass / max(len(records), 1):.1%}")

    if n_invalid > 0:
        print(f"ERROR: {n_invalid} trajectories failed schema validation",
              file=sys.stderr)
        sys.exit(1)
    if n_unchecked > 0 and not args.allow_unchecked:
        print(f"ERROR: {n_unchecked} trajectories could not be checked "
              f"(sandbox dir missing/exec error). Refill outputs/sandbox/ "
              f"or rerun distill, or pass --allow-unchecked to proceed.",
              file=sys.stderr)
        sys.exit(1)
    if not kept:
        print("ERROR: 0 samples survived filtering; refusing to write empty "
              "SFT file", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for item in kept:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"SFT data saved to {out_path}")


if __name__ == "__main__":
    main()
