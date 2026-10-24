#!/bin/bash
# W1-Phase1 补模型：重跑下载脚本，自动跳过 tmp 里已有分片，缺多少补多少
# 用法: bash w1_refill_model.sh
# 完成标志: grep 'DONE unified' /root/autodl-tmp/refill.log
cd /root
if [ ! -f /root/scripts/download_model.py ]; then
  echo "download_model.py 不在 /root/scripts/，从 git 仓库 scripts/cloud/ 上传后再跑"
  exit 1
fi
nohup /root/miniconda3/bin/python /root/scripts/download_model.py > /root/autodl-tmp/refill.log 2>&1 &
echo "refill started (pid $!), log: /root/autodl-tmp/refill.log"
echo "监控: tail -f /root/autodl-tmp/refill.log   (约 50GB / 30 分钟)"
