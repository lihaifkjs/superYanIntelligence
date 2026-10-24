#!/bin/bash
# W1-Phase0 环境体检：GPU 模式开机后第一件事情
# 用法: bash w1_env_check.sh    任一项失败会 exit 1
set -e
echo "== GPU =="
nvidia-smi -L || { echo "FAIL: 无 GPU，必须在 GPU 模式开机"; exit 1; }
echo "== 磁盘 =="
df -h /dev/shm /root/autodl-tmp | tail -2
echo "== 模型统一视图(应为26) =="
ls /root/autodl-tmp/q36/*.safetensors 2>/dev/null | wc -l
echo "== 分片存活 shm/tmp =="
ls /dev/shm/q36_shards/*.safetensors 2>/dev/null | wc -l
ls /root/autodl-tmp/q36_shards/*.safetensors 2>/dev/null | wc -l
echo "== Python 环境 =="
/root/miniconda3/bin/python -c "import torch, transformers, peft, trl, vllm; print('torch', torch.__version__, '| tf', transformers.__version__, '| vllm', vllm.__version__)"
echo "== 关键脚本在位 =="
for f in /root/scripts/download_model.py /root/hle_baseline.py; do
  [ -f "$f" ] && echo "OK $f" || echo "MISSING $f（从 git 仓库 scripts/cloud/ 补传）"
done
echo "ENV CHECK DONE"
