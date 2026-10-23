#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DSH 通知器 —— 把任务进展推送到 微信 / QQ / 短信。

用法:
  python notify.py -t "标题" -c "正文"
  python notify.py -t "标题" -c "正文" --channel wechat,qq,sms
  python notify.py --watch --stale 20 -t "任务卡住了" -c "20 分钟没有进展,回来看看"
  python notify.py --heartbeat            # 刷新心跳(配合 --watch 用的"死信开关")
  python notify.py --test                 # 发一条测试

配置: 与本文件同目录的 notify.config.json (从 notify.config.example.json 复制)
优先级: PushPlus(一个 token 打通微信/QQ/短信) > Server酱(微信) > Qmsg酱(QQ)

设计约束(实测): 本机 pwsh 的 curl/schannel 出网被限制, python/node 的出网正常,
所以这里一律用 python 的 urllib 发 POST, 不走 curl。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "notify.config.json"
LOG_PATH = HERE / "notify.log"
# 本机沙箱下子进程写工作区会失败(EBADF), 日志兜底写到临时目录, 保证链路有痕迹
FALLBACK_LOG_PATH = Path(tempfile.gettempdir()) / "dsh-notify.log"
HEARTBEAT_PATH = HERE / ".notify.heartbeat"

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

PUSHPLUS_URL = "https://www.pushplus.plus/send"


def log(line: str) -> None:
    """追加写日志。注意: 本机沙箱下子进程无法新建文件, 只能追加/修改已有文件,
    所以 LOG_PATH / HEARTBEAT_PATH 需要预先存在(见 README 的 '' 一键初始化)。"""
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for path in (LOG_PATH, FALLBACK_LOG_PATH):
        try:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(f"{stamp} {line}\n")
            return
        except Exception:
            continue


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        sys.exit(
            f"缺少配置文件: {CONFIG_PATH}\n"
            f"请复制 notify.config.example.json 为 notify.config.json 并填入 token。"
        )
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        sys.exit(f"配置文件不是合法 JSON: {exc}")


def http_post_json(url: str, payload: dict, timeout: int = 12) -> dict:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "User-Agent": "dsh-notify/1.0",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"code": -1, "msg": raw[:300]}


def http_get(url: str, timeout: int = 12) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "dsh-notify/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"code": -1, "msg": raw[:300]}


def send_pushplus(cfg: dict, title: str, content: str, channels: list[str]) -> list[dict]:
    token = (cfg.get("pushplus") or {}).get("token", "").strip()
    if not token:
        return [{"channel": "pushplus", "ok": False, "msg": "未配置 pushplus.token"}]

    results = []
    for channel in channels:
        payload = {
            "token": token,
            "title": title,
            "content": content,
            "template": "txt",
            "channel": channel,
        }
        option = ((cfg.get("pushplus") or {}).get("options") or {}).get(channel)
        if option:
            payload["option"] = option
        ok, detail = False, ""
        for attempt in range(2):
            try:
                data = http_post_json(PUSHPLUS_URL, payload)
                code = data.get("code")
                ok = code == 200
                detail = data.get("msg") or ""
                if ok:
                    detail = f"shortCode={data.get('data')}"
                break
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                detail = f"{type(exc).__name__}: {str(exc)[:160]}"
                if attempt == 0:
                    time.sleep(1.5)
        results.append({"channel": channel, "ok": ok, "msg": detail})
    return results


def send_serverchan(cfg: dict, title: str, content: str) -> dict:
    sc = cfg.get("serverchan") or {}
    sendkey = sc.get("sendkey", "").strip()
    if not sendkey:
        return {"channel": "serverchan", "ok": False, "msg": "未配置 serverchan.sendkey"}
    url = sc.get("api") or f"https://sctapi.ftqq.com/{sendkey}.send"
    query = urllib.parse.urlencode({"title": title, "desp": content})
    try:
        data = http_get(f"{url}?{query}")
        ok = data.get("code") == 0
        return {"channel": "serverchan", "ok": ok, "msg": data.get("message", str(data)[:120])}
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return {"channel": "serverchan", "ok": False, "msg": f"{type(exc).__name__}: {str(exc)[:160]}"}


def send_qmsg(cfg: dict, content: str) -> dict:
    key = (cfg.get("qmsg") or {}).get("key", "").strip()
    if not key:
        return {"channel": "qmsg", "ok": False, "msg": "未配置 qmsg.key"}
    url = f"https://qmsg.zendee.cn/v3/send/{key}"
    payload = {"msg": content[:1800]}
    group = (cfg.get("qmsg") or {}).get("group")
    if group:
        payload["group"] = str(group)
    try:
        data = http_post_json(url, payload)
        return {"channel": "qmsg", "ok": bool(data.get("success")), "msg": data.get("message", "")}
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return {"channel": "qmsg", "ok": False, "msg": f"{type(exc).__name__}: {str(exc)[:160]}"}


def dispatch(cfg: dict, title: str, content: str, channels: list[str]) -> tuple[bool, str]:
    results: list[dict] = []
    pushplus_channels = [c for c in channels if c in {"wechat", "qq", "sms", "mail", "voice", "cp"}]
    if pushplus_channels:
        results.extend(send_pushplus(cfg, title, content, pushplus_channels))
    if "serverchan" in channels:
        results.append(send_serverchan(cfg, title, content))
    if "qmsg" in channels:
        results.append(send_qmsg(cfg, f"{title}\n{content}"))

    ok = any(r["ok"] for r in results)
    summary = "; ".join(f"{r['channel']}={'OK' if r['ok'] else 'FAIL(' + r['msg'] + ')'}" for r in results)
    log(f"title={title!r} channels={channels} -> {summary}")
    return ok, summary


def resolve_channels(cfg: dict, override: str | None) -> list[str]:
    if override:
        return [c.strip() for c in override.split(",") if c.strip()]
    chans = cfg.get("channels") or ["wechat"]
    return list(chans)


def touch_heartbeat() -> None:
    """追加一行时间戳(不能覆盖写: 新建文件在沙箱下会 ENOENT, 而追加会同时刷新 mtime)。"""
    with HEARTBEAT_PATH.open("a", encoding="utf-8") as fh:
        fh.write(datetime.now().isoformat(timespec="seconds") + "\n")


def heartbeat_age_seconds() -> float:
    if not HEARTBEAT_PATH.exists():
        return float("inf")
    return time.time() - HEARTBEAT_PATH.stat().st_mtime


def watch(cfg: dict, stale_minutes: float, title: str, content: str, channels: list[str]) -> int:
    """死信开关: 心跳超过 stale_minutes 没刷新, 就推一次告警(每次卡住只推一次)。"""
    stale = stale_minutes * 60
    alerted = False
    print(f"[watch] 监视心跳 {HEARTBEAT_PATH}, 超过 {stale_minutes:g} 分钟未刷新即告警", flush=True)
    while True:
        age = heartbeat_age_seconds()
        if age == float("inf") or age > stale:
            if not alerted:
                ok, summary = dispatch(cfg, title, f"{content}\n(已 {age / 60:.0f} 分钟无进展)" if age != float("inf") else content, channels)
                print(f"[watch] 触发告警: {summary}", flush=True)
                alerted = True
        else:
            if alerted:
                print("[watch] 心跳恢复, 重置告警状态", flush=True)
            alerted = False
        time.sleep(30)


def main() -> int:
    parser = argparse.ArgumentParser(description="DSH 通知器")
    parser.add_argument("-t", "--title", default="DSH 任务通知")
    parser.add_argument("-c", "--content", default="")
    parser.add_argument("--channel", default=None, help="逗号分隔: wechat,qq,sms,serverchan,qmsg")
    parser.add_argument("--test", action="store_true", help="发一条测试消息")
    parser.add_argument("--watch", action="store_true", help="开启死信开关模式")
    parser.add_argument("--stale", type=float, default=20, help="死信阈值(分钟), 默认 20")
    parser.add_argument("--heartbeat", action="store_true", help="刷新心跳时间戳")
    args = parser.parse_args()

    if args.heartbeat:
        touch_heartbeat()
        print(f"[ok] 心跳已刷新: {HEARTBEAT_PATH}")
        return 0

    cfg = load_config()
    channels = resolve_channels(cfg, args.channel)

    if args.test:
        title = args.title if args.title != "DSH 任务通知" else "DSH 通知测试"
        content = args.content or f"这是一条测试消息, 时间 {datetime.now():%Y-%m-%d %H:%M:%S}"
        ok, summary = dispatch(cfg, title, content, channels)
        print(("[ok] " if ok else "[fail] ") + summary)
        return 0 if ok else 1

    if args.watch:
        touch_heartbeat()
        return watch(cfg, args.stale, args.title, args.content, channels)

    if not args.content:
        parser.error("需要 --content (或用 --test)")

    ok, summary = dispatch(cfg, args.title, args.content, channels)
    print(("[ok] " if ok else "[fail] ") + summary)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
