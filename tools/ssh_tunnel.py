"""SSH 本地端口转发：本机 localhost:<local_port> → 云机 127.0.0.1:<remote_port>。

用途：AutoDL 该地区个人用户无法开外网 HTTP 服务（仅企业认证），
harness（terminal-bench 等）通过本机 SSH 隧道访问云机 vLLM。

用法:
    python tools/ssh_tunnel.py --local-port 8000 --remote-port 8000
    python tools/ssh_tunnel.py --local-port 8001 --remote-port 8001 --host 36.137.238.251 --port 32375

凭据默认从 credentials.md 解析（第一个「实例 SSH」节）。
断线自动重连（最多 RETRIES 次，间隔 10s）。
"""

import argparse
import re
import select
import socket
import sys
import threading
import time
from pathlib import Path

import paramiko

ROOT = Path(__file__).resolve().parent.parent
RETRIES = 1000


def load_credentials():
    text = (ROOT / "credentials.md").read_text(encoding="utf-8")
    host = port = password = None
    in_ssh = False
    for line in text.splitlines():
        if line.startswith("## "):
            if in_ssh and host and port and password:
                break
            in_ssh = "实例 SSH" in line
            continue
        if not in_ssh:
            continue
        m = re.search(r"ssh -p (\d+) root@(\S+)", line)
        if m:
            port, host = int(m.group(1)), m.group(2)
        if line.strip().startswith("- 密码"):
            password = line.split("：")[-1].strip()
    if not (host and port and password):
        raise RuntimeError("credentials.md 里没找到实例 SSH 凭据")
    return host, port, password


def handle(chan, client):
    """在本机 client socket 与 SSH channel 之间双向泵数据。"""
    while True:
        r, _, _ = select.select([client, chan], [], [], 1)
        if client in r:
            data = client.recv(16384)
            if not data:
                break
            chan.send(data)
        if chan in r:
            data = chan.recv(16384)
            if not data:
                break
            client.send(data)
    chan.close()
    client.close()


def serve(transport, local_port, remote_port, bind_host):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((bind_host, local_port))
    server.listen(64)
    server.settimeout(2)
    print(f"[tunnel] {bind_host}:{local_port} -> 127.0.0.1:{remote_port} (remote) listening", flush=True)
    while transport.is_active():
        try:
            client, _ = server.accept()
        except socket.timeout:
            continue
        chan = transport.open_channel("direct-tcpip", ("127.0.0.1", remote_port),
                                      client.getpeername())
        if chan is None:
            client.close()
            continue
        threading.Thread(target=handle, args=(chan, client), daemon=True).start()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-port", type=int, required=True)
    ap.add_argument("--remote-port", type=int, required=True)
    ap.add_argument("--bind", default="127.0.0.1",
                    help="本机监听地址；Docker 容器经 host.docker.internal 访问时需 0.0.0.0")
    ap.add_argument("--host")
    ap.add_argument("--port", type=int)
    ap.add_argument("--password")
    args = ap.parse_args()

    try:
        host, port, password = load_credentials()
    except (RuntimeError, FileNotFoundError):
        host = port = password = None
    host = args.host or host
    port = args.port or port
    password = args.password or password
    if not (host and port and password):
        raise RuntimeError("缺少 SSH 凭据：credentials.md 未找到且未传 --host/--port/--password")

    for attempt in range(1, RETRIES + 1):
        try:
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(host, port=port, username="root", password=password,
                           timeout=15, banner_timeout=30, auth_timeout=30)
            client.get_transport().set_keepalive(30)
            print(f"[tunnel] ssh connected {host}:{port} (attempt {attempt})", flush=True)
            serve(client.get_transport(), args.local_port, args.remote_port, args.bind)
            print("[tunnel] transport closed, reconnecting...", flush=True)
        except Exception as e:
            print(f"[tunnel] attempt {attempt} failed: {e}", file=sys.stderr, flush=True)
        time.sleep(10)
    sys.exit(1)


if __name__ == "__main__":
    main()
