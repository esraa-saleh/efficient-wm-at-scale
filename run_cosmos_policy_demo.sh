#!/usr/bin/env bash
set -euo pipefail

# Quick smoke test for a Cosmos Policy install: runs the pretrained LIBERO
# checkpoint for a single trial per task and records MP4 rollout videos.
# Run setup_cosmos_policy_uv.sh first.
#
# Usage:
#   bash run_cosmos_policy_demo.sh [REPO_DIR] [TASK_SUITE]
#
# Example:
#   bash run_cosmos_policy_demo.sh "$PWD/cosmos_policy" libero_spatial
#
# TASK_SUITE defaults to libero_spatial (one of libero_spatial, libero_object,
# libero_goal, libero_10 - see LIBERO.md in the repo).
#
# Videos are written by the eval script itself, always, to:
#   REPO_DIR/rollouts/<date>/*.mp4
# (one per episode - one episode per task in the chosen suite, since
# --num_trials_per_task is fixed at 1 below).
#
# Note: this script runs fully offline (HF_HUB_OFFLINE=1) - the checkpoint
# must already be cached by setup_cosmos_policy_uv.sh. That matters on
# clusters where compute nodes have no internet access; if you see a
# LocalEntryNotFoundError, re-run the setup script (on a node with internet)
# to populate the cache first.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${1:-$SCRIPT_DIR/cosmos_policy}"
TASK_SUITE="${2:-libero_spatial}"

if [[ "$REPO_DIR" != /* ]]; then
  REPO_DIR="$(pwd)/$REPO_DIR"
fi

if [[ ! -f "$REPO_DIR/pyproject.toml" ]]; then
  echo "Error: $REPO_DIR is not a cosmos-policy checkout." >&2
  echo "Run setup_cosmos_policy_uv.sh first (or pass its INSTALL_DIR as \$1)." >&2
  exit 1
fi

ENV_FILE="$REPO_DIR/activate_cuda_env.sh"
if [[ -f "$ENV_FILE" ]]; then
  # shellcheck source=/dev/null
  source "$ENV_FILE"
fi

VENV_DIR="$REPO_DIR/.venv"
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "Error: no venv found at $VENV_DIR. Run setup_cosmos_policy_uv.sh first." >&2
  exit 1
fi

cd "$REPO_DIR"

# All checkpoint/dataset assets are pre-downloaded by setup_cosmos_policy_uv.sh;
# forbid any network fetch here so a missing file fails fast instead of
# hanging (e.g. on a compute node with no internet access).
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

CKPT="nvidia/Cosmos-Policy-LIBERO-Predict2-2B"
RUN_ID="demo_$(printf '%(%Y%m%d_%H%M%S)T' -1)"

# Ensure the directory exists before the first find below - otherwise find
# exits non-zero, and with pipefail+errexit that kills the script silently
# (before any of the echoes further down ever run).
mkdir -p "$REPO_DIR/rollouts"
BEFORE_VIDEOS="$(find "$REPO_DIR/rollouts" -name '*.mp4' | sort)"

echo "Running one LIBERO trial per task on suite '$TASK_SUITE' with $CKPT..."
echo "Rollout videos will be saved under $REPO_DIR/rollouts/"

"$VENV_DIR/bin/python" -m cosmos_policy.experiments.robot.libero.run_libero_eval \
  --config cosmos_predict2_2b_480p_libero__inference_only \
  --ckpt_path "$CKPT" \
  --config_file cosmos_policy/config/config.py \
  --use_wrist_image True \
  --use_proprio True \
  --normalize_proprio True \
  --unnormalize_actions True \
  --dataset_stats_path "$CKPT/libero_dataset_statistics.json" \
  --t5_text_embeddings_path "$CKPT/libero_t5_embeddings.pkl" \
  --trained_with_image_aug True \
  --chunk_size 16 \
  --num_open_loop_steps 16 \
  --task_suite_name "$TASK_SUITE" \
  --num_trials_per_task 1 \
  --local_log_dir cosmos_policy/experiments/robot/libero/logs/ \
  --randomize_seed False \
  --data_collection False \
  --available_gpus "0" \
  --seed 195 \
  --use_variance_scale False \
  --deterministic True \
  --run_id_note "$RUN_ID" \
  --ar_future_prediction False \
  --ar_value_prediction False \
  --use_jpeg_compression True \
  --flip_images True \
  --num_denoising_steps_action 5 \
  --num_denoising_steps_future_state 1 \
  --num_denoising_steps_value 1

AFTER_VIDEOS="$(find "$REPO_DIR/rollouts" -name '*.mp4' 2>/dev/null | sort)"
NEW_VIDEOS="$(comm -13 <(echo "$BEFORE_VIDEOS") <(echo "$AFTER_VIDEOS"))"

echo
echo "Done. New rollout videos from this run:"
if [[ -n "$NEW_VIDEOS" ]]; then
  echo "$NEW_VIDEOS" | sed 's/^/  /'
else
  echo "  (none found - check $REPO_DIR/rollouts/<date>/ manually)"
fi
