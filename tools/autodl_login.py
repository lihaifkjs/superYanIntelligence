"""Open headed Chrome for manual AutoDL login; save session state on success.

Usage: run this script, a Chrome window opens at AutoDL login page.
Log in manually (SMS code / WeChat scan / password), the script detects
success, saves session to tools/autodl_state.json, and exits.
"""

import time
from pathlib import Path

from playwright.sync_api import sync_playwright

STATE_PATH = Path("tools/autodl_state.json")
LOGIN_URL = "https://www.autodl.com/login"
# 登录成功后一般跳转到 /console 或 /home
SUCCESS_HINTS = ("/console", "/home", "/market")


def is_logged_in(page) -> bool:
    url = page.url
    if any(h in url for h in SUCCESS_HINTS):
        return True
    # 登录页消失也算成功
    try:
        body = page.inner_text("body")
        head = body[:200]
        return "登录" not in head and "注册" not in head
    except Exception:
        return False


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=False)
        context = browser.new_context()
        page = context.new_page()
        page.goto(LOGIN_URL, timeout=60000)
        print(f"[autodl-login] 浏览器已打开 {LOGIN_URL}，请手动登录（SMS/微信/密码均可）")
        print("[autodl-login] 等待登录成功，最多 5 分钟...")

        deadline = time.time() + 300
        while time.time() < deadline:
            if is_logged_in(page):
                context.storage_state(path=str(STATE_PATH))
                print(f"[autodl-login] 登录成功，会话已保存到 {STATE_PATH}")
                browser.close()
                return
            time.sleep(2)

        print("[autodl-login] 超时未检测到登录成功，未保存会话")
        browser.close()


if __name__ == "__main__":
    main()
