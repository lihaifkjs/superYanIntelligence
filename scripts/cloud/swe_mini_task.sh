#!/bin/bash
# SWE-bench Pro v2 单题 mini-swe-agent 启动器（与执行环境隔离，避免 pkill 自杀）
pkill -f '[m]ini -c' 2>/dev/null
sleep 1
docker rm -f minisweagent-13928b39 minisweagent-f061e92a 2>/dev/null
mkdir -p /root/w1/outputs/swe-mini3
cd /root/w1/SWE-bench_Pro-os/v2/tasks/instance_NodeBB__NodeBB-00c70ce7b0541cfc94afe567921d7668cdc8f4ac-vnan
MSWEA_CONFIGURED=1 \
MSWEA_MODEL_NAME=openai/qwen35b \
MSWEA_COST_TRACKING=ignore_errors \
OPENAI_API_KEY=dummy \
OPENAI_API_BASE=http://host.docker.internal:8000/v1 \
nohup /root/w1/.venv/bin/mini \
  -c mini.yaml -c swebench.yaml \
  -c environment.image=ghcr.io/scaleapi/swe-bench_pro-v2:instance_NodeBB__NodeBB-00c70ce7b0541cfc94afe567921d7668cdc8f4ac-vnan \
  -c agent.max_consecutive_format_errors=10 \
  -m openai/qwen35b -t instruction.md -y -l 0 \
  -o /root/w1/outputs/swe-mini3/traj.json \
  > /root/w1/outputs/swe-mini3/run.log 2>&1 &
echo "launcher done, mini pid $!"
