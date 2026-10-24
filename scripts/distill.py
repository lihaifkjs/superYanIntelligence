"""自蒸馏：用 teacher 模型（默认本机 3B，云上换 35B 路径即可）生成训练轨迹。

两种模式：
- qa：问答型（HLE）。批量生成 <think> 推理 + ANSWER 最终答案。
- agent：终端操作型（Terminal）。逐轮生成命令 → Git Bash 沙箱执行 → 输出回注，
  直到模型宣布完成或达到轮数上限，记录完整多轮轨迹。

原始轨迹写入 data/raw_traj/{expert}.jsonl，逐条追加，崩溃不丢已完成部分。

本机示例：
  python distill.py --mode qa --seeds data/seeds/hle.jsonl --limit 50
  python distill.py --mode agent --seeds data/seeds/terminal.jsonl --limit 10
云上示例：
  python distill.py --mode qa --model Qwen/Qwen3.6-35B-A3B --seeds data/seeds/hle.jsonl --n 3000
"""

import argparse
import json
import re
import subprocess
from pathlib import Path

from judge import parse_qa_output
from io_utils import append_jsonl, read_jsonl
from llm_io import (
    DEFAULT_MODEL,
    clean_decoded,
    generate,
    load_model_and_tokenizer,
)
from prompts import SYSTEM_PROMPT_AGENT, SYSTEM_PROMPT_QA

# ---------------------------------------------------------------- 沙箱安全

FORBIDDEN_PATTERNS = [
    r"rm\s+-rf?\s+/(?:\s|$)",   # rm -rf /
    r"rm\s+-rf?\s+~",
    r"rm\s+-rf?\s+(\.\./)+",
    r"mkfs",
    r"dd\s+.*of=/dev/",
    r":\(\)\s*\{",              # fork bomb
    r"shutdown|reboot|poweroff",
    r"format\s+[a-z]:",         # windows format
    r"\bdel\s+/[sfq]",          # windows del 危险开关
    r">\s*/dev/sd",
    r"chmod\s+-R\s+777\s+/",
    # 网络与外泄（云上由 docker --network none 兜底，本机靠这层）
    r"\b(curl|wget|nc|ncat|ssh|scp|ping|telnet)\b",
    r"/etc/(passwd|shadow|ssh)",
]

SANDBOX_ROOT = Path("outputs/sandbox")


def is_safe_command(cmd: str) -> bool:
    for pat in FORBIDDEN_PATTERNS:
        if re.search(pat, cmd):
            return False
    return True


def docker_available() -> bool:
    import shutil

    if not shutil.which("docker"):
        return False
    try:
        r = subprocess.run(["docker", "info"], capture_output=True, timeout=15)
        return r.returncode == 0
    except Exception:  # noqa: BLE001
        return False


def run_in_sandbox(cmd: str, workdir: Path, timeout: int,
                   backend: str = "local", docker_image: str = "bash:5") -> dict:
    """沙箱执行命令。backend=local 用宿主机 bash（仅限 smoke）；
    backend=docker 在 --network none 容器里跑（云上真实运行的防线）。"""
    if not is_safe_command(cmd):
        return {"exit_code": -1, "output": "BLOCKED: command matched safety blacklist"}
    try:
        if backend == "docker":
            argv = [
                "docker", "run", "--rm",
                "--network", "none",          # 断网
                "--memory", "2g", "--cpus", "2",
                "-v", f"{Path(workdir).resolve()}:/work", "-w", "/work",
                docker_image, "bash", "-c", cmd,
            ]
            proc = subprocess.run(argv, capture_output=True, text=True,
                                  timeout=timeout, encoding="utf-8", errors="replace")
        else:
            proc = subprocess.run(
                ["bash", "-c", cmd],
                cwd=workdir,
                capture_output=True,
                text=True,
                timeout=timeout,
                encoding="utf-8",
                errors="replace",
            )
        output = (proc.stdout + proc.stderr).strip()
        return {"exit_code": proc.returncode, "output": output[:4000]}
    except subprocess.TimeoutExpired:
        return {"exit_code": -2, "output": f"TIMEOUT after {timeout}s"}
    except Exception as e:  # noqa: BLE001
        return {"exit_code": -3, "output": f"EXEC ERROR: {e}"}


# ---------------------------------------------------------------- agent loop

JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_agent_action(text: str) -> dict | None:
    """从模型输出中解析动作。兼容两种形状：

    - 裸 JSON：{"explanation": ..., "command": "..."}（teacher 常见的直接输出）
    - tool_call JSON：{"name": "run_command", "arguments": {"command": ...}}
      （Qwen 系按工具协议输出时）
    done 标记两种形状都可能出现。
    """
    m = JSON_OBJ_RE.search(text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    if "command" not in obj and isinstance(obj.get("arguments"), dict):
        args = obj["arguments"]
        if "command" in args:
            obj = {"explanation": obj.get("explanation", ""),
                   "command": args["command"]}
    return obj


def run_agent_task(seed: dict, model, tokenizer, args) -> dict:
    workdir = SANDBOX_ROOT / seed["id"]
    workdir.mkdir(parents=True, exist_ok=True)

    setup = seed.get("meta", {}).get("setup")
    if setup:
        run_in_sandbox(setup, workdir, args.timeout, args.sandbox,
                       args.docker_image)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_AGENT},
        {"role": "user", "content": seed["task"]},
    ]
    turns = []
    finished = False
    final_summary = None

    for turn in range(args.max_turns):
        text = generate(
            messages, model, tokenizer,
            max_new_tokens=args.max_new_tokens, temperature=args.temperature,
        )
        action = parse_agent_action(text)
        if action is None:
            turns.append({"command": None, "raw_output": text,
                          "output": "PARSE ERROR: model did not output a JSON object",
                          "exit_code": -4})
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content":
                "Your last message was not a valid JSON object. "
                "Output ONLY the JSON object, nothing else."})
            continue

        if action.get("done"):
            finished = True
            final_summary = str(action.get("summary", ""))
            turns.append({"command": None, "done": True,
                          "explanation": action.get("explanation", ""),
                          "raw_output": text, "output": "", "exit_code": 0})
            break

        cmd = str(action.get("command", "")).strip()
        if not cmd:
            turns.append({"command": None, "raw_output": text,
                          "output": "EMPTY COMMAND", "exit_code": -5})
            continue

        result = run_in_sandbox(cmd, workdir, args.timeout, args.sandbox,
                                args.docker_image)
        turns.append({"command": cmd, "explanation": str(action.get("explanation", "")),
                      "raw_output": text, **result})

        messages.append({"role": "assistant", "content": text})
        messages.append({"role": "user", "content":
            f"exit_code={result['exit_code']}\n{result['output']}"})

    return {
        "id": seed["id"],
        "expert": seed["expert"],
        "task": seed["task"],
        "ground_truth": seed.get("ground_truth"),
        "turns": turns,
        "finished": finished,
        "final_summary": final_summary,
        "meta": seed.get("meta", {}),
        "sandbox_dir": str(workdir),
    }


def run_qa_batch(seeds: list[dict], model, tokenizer, args):
    """批量问答蒸馏。左 padding 组 batch 生成。"""
    import torch

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for start in range(0, len(seeds), args.batch_size):
        batch = seeds[start:start + args.batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "system", "content": SYSTEM_PROMPT_QA},
                 {"role": "user", "content": s["task"]}],
                tokenize=False, add_generation_prompt=True,
            )
            for s in batch
        ]
        enc = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=args.temperature if args.temperature > 0 else None,
                top_p=0.95 if args.temperature > 0 else None,
                pad_token_id=tokenizer.eos_token_id,
            )
        for i, seed in enumerate(batch):
            # 与 llm_io.generate 同策略：保留特殊 token 再手动清理，
            # 否则 35B 的 thinking 标记会被 skip_special_tokens 抹掉
            raw = tokenizer.decode(
                out[i][enc["input_ids"].shape[1]:], skip_special_tokens=False
            )
            parsed = parse_qa_output(clean_decoded(raw))
            append_jsonl(args.output, {
                "id": seed["id"],
                "expert": seed["expert"],
                "task": seed["task"],
                "ground_truth": seed.get("ground_truth"),
                **parsed,
                "meta": seed.get("meta", {}),
            })
        print(f"[qa] {min(start + args.batch_size, len(seeds))}/{len(seeds)} done", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["qa", "agent"], required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--output", default=None,
                        help="默认 data/raw_traj/{expert}.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=8, help="qa 模式批大小")
    parser.add_argument("--max_new_tokens", type=int, default=768)
    parser.add_argument("--max_turns", type=int, default=10, help="agent 模式轮数上限")
    parser.add_argument("--timeout", type=int, default=30, help="单条命令超时秒")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sandbox", choices=["auto", "local", "docker"],
                        default="auto",
                        help="agent 沙箱后端；auto=有 Docker 用 Docker，否则本地 bash")
    parser.add_argument("--docker_image", default="bash:5",
                        help="docker 沙箱镜像（含 git 任务时用带 git 的镜像）")
    args = parser.parse_args()

    if args.mode == "agent":
        if args.sandbox == "auto":
            args.sandbox = "docker" if docker_available() else "local"
        print(f"Sandbox backend: {args.sandbox}"
              + (f" (image={args.docker_image})" if args.sandbox == "docker" else ""))

    seeds = read_jsonl(args.seeds)
    expert = seeds[0]["expert"] if seeds else "unknown"
    if args.output is None:
        args.output = f"data/raw_traj/{expert}.jsonl"
    if args.limit:
        seeds = seeds[:args.limit]

    # 断点续跑：跳过已有 id
    done_ids = set()
    out_path = Path(args.output)
    if out_path.exists():
        done_ids = {r["id"] for r in read_jsonl(out_path)}
        if done_ids:
            print(f"Resuming: {len(done_ids)} records already in {args.output}")
    seeds = [s for s in seeds if s["id"] not in done_ids]
    if not seeds:
        print("Nothing to do.")
        return

    import torch

    torch.manual_seed(args.seed)
    model, tokenizer = load_model_and_tokenizer(args.model)

    if args.mode == "qa":
        run_qa_batch(seeds, model, tokenizer, args)
    else:
        for i, seed in enumerate(seeds, 1):
            traj = run_agent_task(seed, model, tokenizer, args)
            append_jsonl(args.output, traj)
            status = "done" if traj["finished"] else "max_turns"
            print(f"[agent] {i}/{len(seeds)} {seed['id']} turns={len(traj['turns'])} {status}",
                  flush=True)

    print(f"Raw trajectories saved to {args.output}")


if __name__ == "__main__":
    main()
