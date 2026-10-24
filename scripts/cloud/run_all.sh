#!/bin/bash
# 云上主流水线：数据 → 蒸馏 → 过滤 → 训练×4 → 合并矩阵 → 评测。
# 已在本机 3B 全流程验证；上云后只需确认下方 CONFIG 区的变量。
# 用法：bash scripts/cloud/run_all.sh [--skip-data] [--skip-train] [--skip-merge] [--skip-eval]
set -euo pipefail

cd "$(dirname "$0")/../.."
export PYTHONIOENCODING=utf-8

# ============================ CONFIG（上云只改这里） ============================
MODEL="Qwen/Qwen3.6-35B-A3B"        # teacher/base 模型；本机 3B 为 models/models/qwen--Qwen2.5-3B-Instruct/snapshots/master
N_HLE=3000                          # HLE 蒸馏题量
N_TERMINAL_GENERATED=300            # Terminal 模板化生成任务数（内置 30 条固定 smoke 另附）
EPOCHS=2
EXPERTS=(hle terminal swe asi)      # swe/asi 走 prepare_swe_asi.py 开源轨迹
# ==============================================================================

SKIP_DATA=false; SKIP_TRAIN=false; SKIP_MERGE=false; SKIP_EVAL=false
for a in "$@"; do
  case "$a" in
    --skip-data) SKIP_DATA=true ;;
    --skip-train) SKIP_TRAIN=true ;;
    --skip-merge) SKIP_MERGE=true ;;
    --skip-eval) SKIP_EVAL=true ;;
    *) echo "unknown arg: $a"; exit 1 ;;
  esac
done

step() { echo; echo "======== $(date +%H:%M:%S) $* ========"; }

if [ "$SKIP_DATA" = false ]; then
  step "seeds"
  python scripts/prepare_seeds.py --expert hle --n "$N_HLE"
  python scripts/prepare_seeds.py --expert terminal --n_generated "$N_TERMINAL_GENERATED"

  step "distill"
  # 上云第 0 步（不可跳过）：先跑 5 条 qa 人工确认 35B 输出格式被解析器接住，
  # 见 docs/cloud-verify-checklist.md 第 0.5 节，通过后再放开下面全量。
  python scripts/distill.py --mode qa --model "$MODEL" --seeds data/seeds/hle.jsonl
  python scripts/distill.py --mode agent --model "$MODEL" --seeds data/seeds/terminal.jsonl \
      --max_turns 15 --sandbox auto

  step "build sft"
  python scripts/build_sft.py --traj data/raw_traj/hle.jsonl
  python scripts/build_sft.py --traj data/raw_traj/terminal.jsonl
  python scripts/prepare_swe_asi.py --expert both --n 2000

  step "contamination check"
  for e in "${EXPERTS[@]}"; do
    f="data/sft/$e.jsonl"; [ -f "$f" ] || f="data/swe_asi_sample/$e.jsonl"
    python scripts/check_contamination.py --train "$f" \
        --eval data/hle/datasets/cais--hle/snapshots/master/data/test-00000-of-00001.parquet
  done
fi

if [ "$SKIP_TRAIN" = false ]; then
  step "train experts"
  for e in "${EXPERTS[@]}"; do
    f="data/sft/$e.jsonl"; [ -f "$f" ] || f="data/swe_asi_sample/$e.jsonl"
    [ "$e" = terminal ] && TOOLS_FLAG="--tools" || TOOLS_FLAG=""
    # 35B 混合架构 target_modules 以 docs/spike-35b-arch.md 结论为准，必要时加：
    #   --target_modules q_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
    python scripts/train_lora.py --model "$MODEL" --data "$f" \
        --output "outputs/lora-$e" --epochs "$EPOCHS" $TOOLS_FLAG
  done
fi

if [ "$SKIP_MERGE" = false ]; then
  step "merge matrix"
  ADAPTERS=(); for e in "${EXPERTS[@]}"; do ADAPTERS+=("outputs/lora-$e/final"); done
  python scripts/merge_experts.py --base "$MODEL" --experts "${ADAPTERS[@]}" \
      --method task_arithmetic --scaling 1.0 --output outputs/merged-ta
  python scripts/merge_experts.py --base "$MODEL" --experts "${ADAPTERS[@]}" \
      --method ties --density 0.2 --output outputs/merged-ties
  python scripts/merge_experts.py --base "$MODEL" --experts "${ADAPTERS[@]}" \
      --method ties --density 0.2 --freeze_router --output outputs/merged-ties-frz
fi

if [ "$SKIP_EVAL" = false ]; then
  step "eval"
  echo "HLE 官方 harness + 各榜 harness 调用写在这里（锁 tag + 记录 commit hash）。"
  echo "推理服务：GPU 机 vLLM/OpenAI 端点；harness 与 Docker 走 CPU 机，见 PRD 4.6。"
  # python scripts/hle/run_hle_smoke.py --model outputs/merged-ta ...
fi

step "ALL DONE"
