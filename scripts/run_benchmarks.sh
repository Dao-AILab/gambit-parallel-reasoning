#!/usr/bin/env bash
# Run the Gambit tournament across all benchmarks, one GPU per benchmark.
#
#   bash scripts/run_benchmarks.sh
#   MODEL=Qwen/Qwen3-4B-Thinking-2507 SCORER=/path/to.pt GPUS="0 1" bash scripts/run_benchmarks.sh
#
# To average over repeats, re-run with a different SEED each time and pool the
# run directories with gambit/tests/compute_pass_at_n.py --run-dir A --run-dir B.
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

MODEL="${MODEL:-deepseek-ai/DeepSeek-R1-0528-Qwen3-8B}"
SCORER="${SCORER:-${ROOT_DIR}/gambit/step_scorer_checkpoint/DeepSeek-R1-0528-Qwen3-8B_step_scorer.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/gambit/eval_result}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"
N="${N:-256}"
SEED="${SEED:-42}"
# Tournament hyperparameters from the paper (Section 5.1); K is fixed, not a
# function of N.
SWAP_K="${SWAP_K:-16}"
CHECK_INTERVAL="${CHECK_INTERVAL:-200}"
WARMUP_TOKENS="${WARMUP_TOKENS:-12000}"
HARD_FLOOR="${HARD_FLOOR:-0.1}"

BENCHMARKS=(aime_2025 hmmt_2024 hmmt_2025 gpqa_diamond)
read -r -a GPUS <<< "${GPUS:-0 1 2 3}"

[[ -f "${SCORER}" ]] || { echo "Scorer not found: ${SCORER}" >&2; exit 1; }

run_one() {
  local gpu="$1" name="$2"
  local bench="${ROOT_DIR}/gambit/datasets/${name}.jsonl"

  echo "[GPU ${gpu}] start ${name}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${ROOT_DIR}/gambit/tests/benchmark_eval.py" \
    --benchmark "${bench}" \
    --output-dir "${OUTPUT_DIR}" \
    --model-path "${MODEL}" \
    --gambit-step-scorer-path "${SCORER}" \
    --num-traces "${N}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --enable-gambit --enable-branching --no-score-history \
    --tournament-capacity "${N}" \
    --tournament-swap-k "${SWAP_K}" \
    --tournament-check-interval "${CHECK_INTERVAL}" \
    --tournament-warmup-tokens "${WARMUP_TOKENS}" \
    --tournament-hard-floor "${HARD_FLOOR}" \
    --stop-after-completed-traces "${N}" \
    --seed "${SEED}" \
    || echo "[GPU ${gpu}] WARNING: ${name} exited with code $?"
  echo "[GPU ${gpu}] done ${name}"
}

for i in "${!BENCHMARKS[@]}"; do
  gpu="${GPUS[$(( i % ${#GPUS[@]} ))]}"
  run_one "${gpu}" "${BENCHMARKS[$i]}" &
done

wait
echo "All benchmarks complete (seed=${SEED}). Outputs under: ${OUTPUT_DIR}"
