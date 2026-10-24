#!/bin/bash
# W1-Phase2 起 vLLM 服务（需模型已就位，GPU 模式）
# 用法: bash w1_start_vllm.sh
# 成功输出 VLLM READY + 模型列表；失败输出日志尾部并 exit 1
ps aux | grep 'vllm serve' | grep -v grep | awk '{print $2}' | xargs -r kill 2>/dev/null
sleep 3
nohup /root/miniconda3/bin/vllm serve /root/autodl-tmp/q36 \
  --served-model-name qwen35b \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.97 \
  --enforce-eager \
  --port 8000 > /root/autodl-tmp/vllm.log 2>&1 &
echo "vllm starting, log: /root/autodl-tmp/vllm.log"
for i in $(seq 1 90); do
  sleep 10
  if grep -q "Application startup complete" /root/autodl-tmp/vllm.log 2>/dev/null; then
    echo "VLLM READY"
    curl -s http://127.0.0.1:8000/v1/models
    exit 0
  fi
  if grep -qE "Traceback|CUDA out of memory|ValueError|KeyError" /root/autodl-tmp/vllm.log 2>/dev/null; then
    echo "VLLM FAILED:"; tail -30 /root/autodl-tmp/vllm.log; exit 1
  fi
done
echo "TIMEOUT after 15min:"; tail -20 /root/autodl-tmp/vllm.log; exit 1
