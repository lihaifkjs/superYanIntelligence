#!/bin/bash
# 云上环境一次性准备。租机后先跑这个（无卡模式也能跑）。
# 用法：bash scripts/cloud/setup_env.sh
set -euo pipefail

cd "$(dirname "$0")/../.."

echo "== [1/4] pip 依赖 =="
pip install -r scripts/requirements.txt
pip install modelscope safetensors

echo "== [2/4] 拉取 teacher 模型（ModelScope） =="
# 已下载则秒过；35B 约 67G，注意数据盘容量
python -c "
from modelscope import snapshot_download
p = snapshot_download('Qwen/Qwen3.6-35B-A3B')
print('model at:', p)
"

echo "== [3/4] 评测仓库（锁 tag，hash 记进 README）与 Docker 沙箱镜像 =="
mkdir -p external
[ -d external/hle ] || git clone --depth 1 https://github.com/centerforaisafety/hle.git external/hle
# Terminal-Bench / SWE-bench Pro / ASI-Bench 仓库在首次评测前补这里，务必锁 tag：
#   git clone -b <tag> <repo> external/terminal-bench && cd external/terminal-bench && git rev-parse HEAD
# agent 沙箱镜像（含 git 任务；distill.py --docker_image 可改）
docker pull "${DOCKER_IMAGE:-alpine/git}" 2>/dev/null || docker pull bash:5 || true

echo "== [4/4] HLE 评测数据（ModelScope 镜像） =="
python scripts/prepare_seeds.py --expert hle --n 300 || true   # 顺带验证数据集链路

echo "setup_env done."
