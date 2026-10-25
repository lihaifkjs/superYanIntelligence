#!/bin/bash
# W1-Phase2 起 vLLM（对外 + 可选 eager）：默认对外绑定 0.0.0.0，EAGER=1 时加 --enforce-eager
# 用法: bash w1_start_vllm_perf.sh        # 无 eager（先试性能）
#       EAGER=1 bash w1_start_vllm_perf.sh # 回退 eager 模式
# 成功输出 VLLM READY；失败输出日志尾部并 exit 1
export PATH=/root/miniconda3/bin:$PATH   # flashinfer JIT 需 ninja（conda 的），SSH 非登录 shell 默认无此 PATH
ps aux | grep 'vllm serve' | grep -v grep | awk '{print $2}' | xargs -r kill 2>/dev/null
sleep 3
EAGER_FLAG=""
[ "$EAGER" = "1" ] && EAGER_FLAG="--enforce-eager"
nohup /root/miniconda3/bin/vllm serve /root/autodl-tmp/q36 \
  --served-model-name qwen35b \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.97 \
  --host 0.0.0.0 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  $EAGER_FLAG \
  --port 8000 > /root/autodl-tmp/vllm.log 2>&1 &
echo "vllm starting (eager=${EAGER:-0}), log: /root/autodl-tmp/vllm.log"
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
