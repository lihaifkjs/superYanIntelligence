#!/bin/bash
# W1-Phase3 HLE baseline：先冒烟后子集（全量评测属 W4，不在 W1 范围）
# 用法: bash w1_hle_baseline.sh [N]   默认 10 题冒烟；建议 W1 出分用 200
N=${1:-10}
PARQUET=/root/autodl-tmp/hle/datasets/cais--hle/snapshots/master/data/test-00000-of-00001.parquet
if [ ! -f "$PARQUET" ]; then
  echo "HLE parquet 不在，先下载："
  /root/miniconda3/bin/python -c "from modelscope.hub.snapshot_download import snapshot_download; print(snapshot_download('cais/hle', repo_type='dataset', cache_dir='/root/autodl-tmp/hle'))"
fi
/root/miniconda3/bin/python /root/hle_baseline.py \
  --parquet "$PARQUET" --n "$N" \
  --out /root/autodl-tmp/hle_pred_$N.json
echo "结果文件: /root/autodl-tmp/hle_pred_$N.json"
