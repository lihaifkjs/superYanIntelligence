"""只上传文件到云机（不执行），用法: python tools/upload_file.py <local> <remote> [host] [port] [password]"""
import sys
from pathlib import Path

import paramiko

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ssh_exec import load_credentials

local, remote = sys.argv[1], sys.argv[2]
host, port, password = load_credentials()
if len(sys.argv) > 3:
    host = sys.argv[3]
if len(sys.argv) > 4:
    port = int(sys.argv[4])
if len(sys.argv) > 5:
    password = sys.argv[5]

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(host, port=port, username="root", password=password,
               timeout=15, banner_timeout=30, auth_timeout=30)
sftp = client.open_sftp()
sftp.put(local, remote)
sftp.close()
client.close()
print(f"uploaded {local} -> {remote}")
