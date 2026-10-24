#!/bin/bash
# W2 云主流水线：第0步格式验证 → 种子 → 蒸馏 → 过滤 → 污染检测 → HLE专家训练 → 50题对比评测
# 用法（云机，仓库 clone 在 /root/syi）：
#   cd /root/syi && bash scripts/cloud/w2_pipeline.sh                 # smoke 档
#   cd /root/syi && SCALE=full bash scripts/cloud/w2_pipeline.sh     # 全量档
#   每步可跳：--skip-step0/--skip-data/--skip-train/--skip-eval
# 失败即停；所有日志同时在终端和 /root/autodl-tmp/w2_pipeline.log
set -euo pipefail

cd "$(dirname "$0")/../.."
exec > >(tee -a /root/autodl-tmp/w2_pipeline.log) 2>&1

export PYTHONIOENCODING=utf-8
PY=/root/miniconda3/bin/python
MODEL="${MODEL:-/root/autodl-tmp/q36}"
HLE_PARQUET="${HLE_PARQUET:-/root/autodl-tmp/hle/datasets/caiss--hle/snapshots/master/data/test-00000-of-00001.parquet}"
MS_CACHE="${MS_CACHE:-/root/autodl-tmp/ms_cache}"
SCALE="${SCALE:-smoke}"            # smoke | full
N_HLE_SMOKE=30;  N_HLE_FULL=3000
N_TERM_SMOKE=10; N_TERM_FULL=300
N_EVAL=50

SKIP_STEP0=false; SKIP_DATA=false; SKIP_TRAIN=false; SKIP_EVAL=false
for a in "$@"; do
  case "$a" in
    --skip-step0) SKIP_STEP0=true ;;
    --skip-data) SKIP_DATA=true ;;
    --skip-train) SKIP_TRAIN=true ;;
    --skip-eval) SKIP_EVAL=true ;;
    *) echo "unknown arg: $a"; exit 1 ;;
  esac
done

step() { echo; echo "======== $(date '+%m-%d %H:%M:%S') $* ========"; }

if [ "$SCALE" = full ]; then N_HLE=$N_HLE_FULL; N_TERM=$N_TERM_FULL;
else N_HLE=$N_HLE_SMOKE; N_TERM=$N_TERM_SMOKE; fi

if [ "$SKIP_STEP0" = false ]; then
  step "Phase 0: 35B 蒸馏格式验证（不可跳过）"
  $PY scripts/cloud/w2_step0_format_check.py --model "$MODEL"
fi

if [ "$SKIP_DATA" = false ]; then
  step "seeds"
  $PY scripts/prepare_seeds.py --expert hle --n "$N_HLE" --cache_dir "$MS_CACHE"
  $PY scripts/prepare_seeds.py --expert terminal --n_generated "$N_TERM"

  step "distill (HLE qa $N_HLE / Terminal agent $N_TERM, scale=$SCALE)"
  # 冒烟档 --limit 截断；全量档断点续跑，中断重跑本脚本即可
  LIMIT_FLAG=""; [ "$SCALE" = smoke ] && LIMIT_FLAG="--limit $N_HLE"
  $PY scripts/distill.py --mode qa --model "$MODEL" \
      --seeds data/seeds/hle.jsonl --batch_size 4 $LIMIT_FLAG
  LIMIT_FLAG=""; [ "$SCALE" = smoke ] && LIMIT_FLAG="--limit $N_TERM"
  $PY scripts/distill.py --mode agent --model "$MODEL" \
      --seeds data/seeds/terminal.jsonl --max_turns 12 --sandbox auto $LIMIT_FLAG

  step "build sft + swe/asi"
  $PY scripts/build_sft.py --traj data/raw_traj/hle.jsonl
  $PY scripts/build_sft.py --traj data/raw_traj/terminal.jsonl
  $PY scripts/prepare_swe_asi.py --expert both --n 2000 --cache_dir "$MS_CACHE"

  step "contamination check（exit 非零即停）"
  $PY scripts/check_contamination.py --train data/sft/hle.jsonl --eval "$HLE_PARQUET"
fi

if [ "$SKIP_TRAIN" = false ]; then
  step "train HLE expert (bf16)"
  ps aux | grep 'vllm serve' | grep -v grep | awk '{print $2}' | xargs -r kill || true
  sleep 10
  $PY scripts/train_lora.py --model "$MODEL" --data data/sft/hle.jsonl \
    --output /root/autodl-tmp/lora-hle \
    --target_modules q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,in_proj_a,in_proj_b,out_proj,gate_proj,up_proj,down_proj \
    --epochs 2 --batch_size 1 --gradient_accumulation_steps 4
fi

if [ "$SKIP_EVAL" = false ]; then
  step "eval: Base vs HLE expert ($N_EVAL 题同口径)"
  $PY scripts/merge_lora.py --model "$MODEL" \
    --adapter /root/autodl-tmp/lora-hle/final --output /root/autodl-tmp/q36-hle
  nohup $PY -m vllm.entrypoints.openai.api_server --model /root/autodl-tmp/q36 \
    --served-model-name qwen35b --max-model-len 16384 \
    --gpu-memory-utilization 0.97 --enforce-eager --port 8000 \
    > /root/autodl-tmp/vllm.log 2>&1 &
  for i in $(seq 1 60); do
    sleep 15
    curl -s http://127.0.0.1:8000/v1/models 2>/dev/null | grep -q qwen35b && break
    [ "$i" = 60 ] && { echo "vLLM 启动超时，看 /root/autodl-tmp/vllm.log"; exit 1; }
  done
  $PY scripts/cloud/hle_baseline.py --parquet "$HLE_PARQUET" \
    --url http://127.0.0.1:8000/v1 --model qwen35b --n "$N_EVAL" \
    --out /root/autodl-tmp/hle_base_${N_EVAL}.json
  ps aux | grep 'vllm serve\|api_server' | grep -v grep | awk '{print $2}' | xargs -r kill
  sleep 10
  nohup $PY -m vllm.entrypoints.openai.api_server --model /root/autodl-tmp/q36-hle \
    --served-model-name qwen35b --max-model-len 16384 \
    --gpu-memory-utilization 0.97 --enforce-eager --port 8000 \
    > /root/autodl-tmp/vllm_expert.log 2>&1 &
  for i in $(seq 1 60); do
    sleep 15
    curl -s http://127.0.0.1:8000/v1/models 2>/dev/null | grep -q qwen35b && break
    [ "$i" = 60 ] && { echo "vLLM(专家) 启动超时"; exit 1; }
  done
  $PY scripts/cloud/hle_baseline.py --parquet "$HLE_PARQUET" \
    --url http://127.0.0.1:8000/v1 --model qwen35b --n "$N_EVAL" \
    --out /root/autodl-tmp/hle_expert_${N_EVAL}.json
  ps aux | grep 'vllm serve\|api_server' | grep -v grep | awk '{print $2}' | xargs -r kill || true
  echo "对比结果：/root/autodl-tmp/hle_base_${N_EVAL}.json vs hle_expert_${N_EVAL}.json"
fi

step "W2 PIPELINE DONE (scale=$SCALE)"
echo "下一步：按 docs/cloud-w2-runbook.md Phase 4 落档 + push + 微信通知"
