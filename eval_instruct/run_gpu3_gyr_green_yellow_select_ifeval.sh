#!/usr/bin/env bash
set -euo pipefail

# Front-run only (no nohup/tmux/screen). Designed for interactive Ctrl+C.

source /workspace/venvs/real_dreamvenv/bin/activate
cd /workspace/Dream_GYR_v03/eval_instruct

export CUDA_VISIBLE_DEVICES=2
export PYTHONUNBUFFERED=1

# ---- Experiment config (edit here) ----
PRETRAINED="Dream-org/Dream-v0-Instruct-7B"
ALG="entropy"
DIFFUSION_STEPS=1280
MAX_NEW_TOKENS=1280
BATCH_SIZE=1
SEED=1234
LIMIT=50

# Green/red policy toggle (required for yellow to be effective)
ENABLE_GREEN_RED_POLICY=1  # 0/1

# Policy params (match DreamGenerationConfig defaults)
GREEN_CONF_THRESH=-0.01
GREEN_MIN_STABLE_STEPS=2
GREEN_MAX_OSC=256
GREENEST_FORCE_UNMASK=1   # 0/1
GREENEST_SCORE_MODE="confidence"

# ---- Yellow candidate selection (NO COMMIT in this PR) ----
ENABLE_YELLOW_POLICY=1          # 0/1
YELLOW_CONF_THRESH=-0.1
YELLOW_MIN_REMAINING_MASKS=32
YELLOW_FRAC_OF_REMAINING=0.15
YELLOW_MAX_PER_ROW=-1
YELLOW_MIN_SELECTED_CAP=32
YELLOW_MIN_STABLE_STEPS=1  # 안정성은 0이랑 1이랑 동작 똑같음

# Debug logging: print per-step green/red counts (+ yellow when enabled)
ENABLE_GR_STEP_LOGGING=1  # 0/1

if [[ "${ENABLE_GR_STEP_LOGGING}" == "1" && "${ENABLE_GREEN_RED_POLICY}" == "0" ]]; then
  echo "[warn] ENABLE_GR_STEP_LOGGING=1 but ENABLE_GREEN_RED_POLICY=0 -> no GR step logs will be produced."
fi

# fp_stats
ENABLE_FP_STATS=1         # 0/1
FP_STATS_APPEND=1         # 0/1

# Early stop (saves forward passes once all masks are resolved)
ENABLE_EARLY_STOP_WHEN_NO_MASK=1      # 0/1
EARLY_STOP_ONLY_WHEN_GR_ENABLED=1     # 0/1

TS=$(date +"%Y%m%d_%H%M%S")
OUT_ROOT="out/gyr_green_yellow_select/ifeval/policy${ENABLE_GREEN_RED_POLICY}/limit${LIMIT}"
RUN_TAG="alg${ALG}_seed${SEED}_ts${TS}_steps${DIFFUSION_STEPS}_newtok${MAX_NEW_TOKENS}_bs${BATCH_SIZE}"
OUT_DIR="${OUT_ROOT}/${RUN_TAG}"
mkdir -p "${OUT_DIR}"

FP_STATS_PATH="${OUT_DIR}/fp_stats.json"

# Mirror logs under a separate root (out/... -> logs/...)
LOG_DIR="${OUT_DIR/#out\//logs/}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/run.log"

MODEL_ARGS=(
  "pretrained=${PRETRAINED}"
  "trust_remote_code=True"
  "dtype=bfloat16"
  "temperature=0.1"
  "top_p=0.9"
  "alg=${ALG}"
  "diffusion_steps=${DIFFUSION_STEPS}"
  "max_new_tokens=${MAX_NEW_TOKENS}"
  "enable_green_red_policy=${ENABLE_GREEN_RED_POLICY}"
  "green_conf_thresh=${GREEN_CONF_THRESH}"
  "green_min_stable_steps=${GREEN_MIN_STABLE_STEPS}"
  "green_max_osc=${GREEN_MAX_OSC}"
  "greenest_force_unmask=${GREENEST_FORCE_UNMASK}"
  "greenest_score_mode=${GREENEST_SCORE_MODE}"
  "enable_yellow_policy=${ENABLE_YELLOW_POLICY}"
  "yellow_conf_thresh=${YELLOW_CONF_THRESH}"
  "yellow_min_remaining_masks=${YELLOW_MIN_REMAINING_MASKS}"
  "yellow_frac_of_remaining=${YELLOW_FRAC_OF_REMAINING}"
  "yellow_max_per_row=${YELLOW_MAX_PER_ROW}"
  "yellow_min_selected_cap=${YELLOW_MIN_SELECTED_CAP}"
  "yellow_min_stable_steps=${YELLOW_MIN_STABLE_STEPS}"
  "enable_gr_step_logging=${ENABLE_GR_STEP_LOGGING}"
  "enable_fp_stats=${ENABLE_FP_STATS}"
  "fp_stats_path=${FP_STATS_PATH}"
  "fp_stats_append=${FP_STATS_APPEND}"
  "enable_early_stop_when_no_mask=${ENABLE_EARLY_STOP_WHEN_NO_MASK}"
  "early_stop_only_when_gr_enabled=${EARLY_STOP_ONLY_WHEN_GR_ENABLED}"
)

CMD=(
  accelerate launch --main_process_port 12335 -m lm_eval
  --model diffllm
  --model_args "$(IFS=,; echo "${MODEL_ARGS[*]}")"
  --tasks ifeval
  --device cuda
  --batch_size "${BATCH_SIZE}"
  --num_fewshot 0
  --seed "${SEED}"
  --output_path "${OUT_DIR}"
  --log_samples --confirm_run_unsafe_code
  --apply_chat_template
)

if [[ "${LIMIT}" != "0" ]]; then
  CMD+=(--limit "${LIMIT}")
fi

echo "========================================="
echo "Dream_GYR_v03 IFEval green/yellow-select run"
echo "GPU=${CUDA_VISIBLE_DEVICES} ALG=${ALG} STEPS=${DIFFUSION_STEPS} NEWTOK=${MAX_NEW_TOKENS} SEED=${SEED} LIMIT=${LIMIT}"
echo "ENABLE_GREEN_RED_POLICY=${ENABLE_GREEN_RED_POLICY} ENABLE_YELLOW_POLICY=${ENABLE_YELLOW_POLICY}"
echo "output_path=${OUT_DIR}"
echo "fp_stats_path=${FP_STATS_PATH}"
echo "log_file=${LOG_FILE}"
echo "========================================="

PYTHONPATH=. "${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
