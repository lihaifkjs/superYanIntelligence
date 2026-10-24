"""可复用的云机 SSH 命令执行器。

凭据从 credentials.md 解析（AutoDL 实例 SSH 节），也可 --host/--port/--password 覆盖。

用法:
    python tools/ssh_exec.py --cmd "nvidia-smi"
    python tools/ssh_exec.py --cmd "df -h" --timeout 60
    python tools/ssh_exec.py --file setup.sh          # 上传并执行脚本
"""

import argparse
import re
import sys
import time
from pathlib import Path

import paramiko

ROOT = Path(__file__).resolve().parent.parent


def load_credentials():
    text = (ROOT / "credentials.md").read_text(encoding="utf-8")
    host = port = password = None
    in_ssh = False
    for line in text.splitlines():
        if line.startswith("## "):
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


def run_cmd(client, cmd, timeout=120, stream=False):
    """执行单条命令，返回 (exit_code, stdout)。流式打印 stderr。"""
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    code = stdout.channel.recv_exit_status()
    if err.strip():
        print(err, file=sys.stderr, end="")
    return code, out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmd", help="要在云机执行的命令")
    parser.add_argument("--file", help="上传并执行的本地脚本路径")
    parser.add_argument("--host", help="覆盖主机")
    parser.add_argument("--port", type=int, help="覆盖端口")
    parser.add_argument("--password", help="覆盖密码")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=3, help="连接重试次数")
    args = parser.parse_args()

    if not args.cmd and not args.file:
        parser.error("--cmd 或 --file 至少给一个")

    host, port, password = load_credentials()
    host = args.host or host
    port = args.port or port
    password = args.password or password

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    for attempt in range(1, args.retries + 1):
        try:
            client.connect(host, port=port, username="root", password=password,
                           timeout=15, banner_timeout=30, auth_timeout=30)
            break
        except Exception as e:
            if attempt == args.retries:
                raise
            print(f"connect attempt {attempt} failed: {e}, retrying...", file=sys.stderr)
            time.sleep(3)

    try:
        if args.file:
            local = Path(args.file)
            remote = f"/root/{local.name}"
            sftp = client.open_sftp()
            sftp.put(str(local), remote)
            sftp.close()
            print(f"uploaded {local} -> {remote}")
            code, out = run_cmd(client, f"bash {remote}", timeout=args.timeout)
        else:
            code, out = run_cmd(client, args.cmd, timeout=args.timeout)
        print(out, end="")
        sys.exit(code)
    finally:
        client.close()


if __name__ == "__main__":
    main()
