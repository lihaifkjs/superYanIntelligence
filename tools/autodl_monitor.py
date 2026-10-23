"""AutoDL GPU 库存监控。

用已保存的登录会话（tools/autodl_state.json）打开算力市场，
遍历所有地区，解析 GPU 型号筛选项的 `空闲/总量` 计数；
目标卡有空闲时通过微信（tools/notify.py）通知。

用法：
    python tools/autodl_monitor.py                # 前台，5 分钟一轮
    python tools/autodl_monitor.py --interval 180 --headless

停止：创建 tools/autodl_monitor.stop 文件，或 Ctrl+C。
"""

import argparse
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
STATE_PATH = ROOT / "tools" / "autodl_state.json"
STOP_FILE = ROOT / "tools" / "autodl_monitor.stop"
LOG_FILE = ROOT / "tools" / "autodl_monitor.log"
NOTIFY = ROOT / "tools" / "notify.py"
MARKET_URL = "https://www.autodl.com/market/list"

DEFAULT_TARGET_GPUS = ["A100", "A800", "H100", "H800", "H20"]
# 筛选项行格式：'A800-80GB (3/464)' / 'H800 (0/96)'，名称后可能带 -XXGB 后缀
FILTER_RE_TEMPLATE = r"^{name}(?:-\d+GB)? \((\d+)/(\d+)\)$"
CARD_TITLE_RE = r"^([A-Za-z0-9 \-]+) / (\d+)\s*GB$"
PRICE_RE = re.compile(r"￥([\d.]+)/时")
RENTABLE_RE = re.compile(r"(\d+)卡可租")
SOLDOUT_TEXT = "已租完"


def log(msg: str):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def notify(title: str, content: str):
    try:
        subprocess.run(
            [sys.executable, str(NOTIFY), "-t", title, "-c", content],
            cwd=str(ROOT),
            timeout=30,
            capture_output=True,
        )
    except Exception as e:  # 通知失败不阻塞监控
        log(f"notify failed: {e}")


def parse_filter_counts(body_text: str, targets: list[str]) -> dict[str, tuple[int, int]]:
    """从 body 文本解析目标 GPU 的 (空闲, 总量)。"""
    counts = {}
    lines = [l.strip() for l in body_text.splitlines() if l.strip()]
    for name in targets:
        pat = re.compile(FILTER_RE_TEMPLATE.format(name=re.escape(name)))
        for line in lines:
            m = pat.match(line)
            if m:
                counts[name] = (int(m.group(1)), int(m.group(2)))
                break
    return counts


def parse_min_price(body_text: str, gpu_name: str) -> float | None:
    """在实例卡片中找目标 GPU 可租机器的最低价。"""
    lines = [l.strip() for l in body_text.splitlines() if l.strip()]
    prices = []
    for i, line in enumerate(lines):
        m = re.match(CARD_TITLE_RE, line)
        if not m or m.group(1).strip() != gpu_name:
            continue
        block = lines[i : i + 20]
        price = status = None
        for b in block:
            pm = PRICE_RE.search(b)
            if pm and price is None:
                price = float(pm.group(1))
            if RENTABLE_RE.search(b):
                status = "rentable"
            if SOLDOUT_TEXT in b:
                status = "soldout"
        if price is not None and status == "rentable":
            prices.append(price)
    return min(prices) if prices else None


def get_regions(page) -> list[str]:
    # 地区名都以「区」结尾；排除计费方式（按量计费等）和 GPU 数量（数字）按钮
    return [
        t.strip()
        for t in page.locator("label.el-radio-button .el-radio-button__inner").all_inner_texts()
        if t.strip().endswith("区")
    ]


def click_region(page, region: str):
    page.locator("label.el-radio-button", has_text=region).first.click()
    page.wait_for_timeout(5000)


def is_login_wall(body_text: str) -> bool:
    return "GPU型号" not in body_text and "登录" in body_text[:300]


def check_once(page, targets: list[str]) -> dict[str, dict]:
    """遍历所有地区，返回 {gpu: {region, free, total, price}}（只含有空闲的）。"""
    page.goto(MARKET_URL, wait_until="domcontentloaded", timeout=60000)
    page.wait_for_timeout(8000)

    body = page.inner_text("body")
    if is_login_wall(body):
        raise RuntimeError("login wall detected (session expired or risk control)")

    found = {}
    regions = get_regions(page)
    log(f"regions: {regions}")
    for region in regions:
        try:
            click_region(page, region)
        except Exception as e:
            log(f"click region {region} failed: {e}")
            continue
        body = page.inner_text("body")
        counts = parse_filter_counts(body, targets)
        for name, (free, total) in counts.items():
            if free > 0:
                price = parse_min_price(body, name)
                found[name] = {
                    "region": region,
                    "free": free,
                    "total": total,
                    "price": price,
                }
                log(f"FOUND {name} in {region}: free {free}/{total}, min price {price}")
        log(f"checked {region}: {counts}")
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=300, help="检查间隔秒数")
    parser.add_argument("--headless", action="store_true", help="无头模式（可能被风控，默认有头）")
    parser.add_argument("--gpus", default=",".join(DEFAULT_TARGET_GPUS), help="目标 GPU，逗号分隔")
    parser.add_argument("--max-notify-interval", type=int, default=1800,
                        help="持续有货时每多少秒重复通知一次")
    args = parser.parse_args()
    targets = [g.strip() for g in args.gpus.split(",") if g.strip()]

    if not STATE_PATH.exists():
        print(f"no session state: {STATE_PATH}, run tools/autodl_login.py first")
        sys.exit(1)

    log(f"monitor started, targets={targets}, interval={args.interval}s")
    notify("[DSH] AutoDL 监控启动", f"盯着 {targets}，{args.interval}s 一轮\n时间: {datetime.now():%Y-%m-%d %H:%M}")

    last_found: dict[str, dict] = {}
    last_notify_time = 0.0

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=args.headless)
        context = browser.new_context(storage_state=str(STATE_PATH))
        page = context.new_page()

        try:
            while not STOP_FILE.exists():
                try:
                    found = check_once(page, targets)
                except RuntimeError as e:
                    log(f"check error: {e}")
                    notify("[DSH] AutoDL 监控异常", f"会话可能失效，需重新登录\n时间: {datetime.now():%Y-%m-%d %H:%M}")
                    break
                except Exception as e:
                    log(f"check error: {e}")
                    time.sleep(args.interval)
                    continue

                new_items = {k: v for k, v in found.items() if k not in last_found}
                now = time.time()
                if new_items:
                    lines = [
                        f"{k}: {v['region']} 空闲 {v['free']}/{v['total']}"
                        + (f" 最低 ¥{v['price']}/时" if v["price"] else "")
                        for k, v in new_items.items()
                    ]
                    notify(
                        "[DSH] AutoDL 有货 " + ",".join(new_items),
                        "新发现空闲:\n" + "\n".join(lines) + f"\n时间: {datetime.now():%Y-%m-%d %H:%M}",
                    )
                    last_notify_time = now
                elif found and now - last_notify_time > args.max_notify_interval:
                    lines = [
                        f"{k}: {v['region']} 空闲 {v['free']}/{v['total']}"
                        for k, v in found.items()
                    ]
                    notify(
                        "[DSH] AutoDL 持续有货 " + ",".join(found),
                        "仍有空闲:\n" + "\n".join(lines) + f"\n时间: {datetime.now():%Y-%m-%d %H:%M}",
                    )
                    last_notify_time = now

                last_found = found
                time.sleep(args.interval)
        except KeyboardInterrupt:
            log("interrupted")
        finally:
            browser.close()
            if STOP_FILE.exists():
                STOP_FILE.unlink()
            log("monitor stopped")


if __name__ == "__main__":
    main()
