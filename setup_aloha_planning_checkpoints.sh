#!/usr/bin/env bash
# setup_aloha_planning_checkpoints.sh
#
# Prefetches the two checkpoints needed to run the ALOHA "main planning config" example
# documented in cosmos_policy/cosmos_policy/experiments/robot/aloha/deploy.py:
#   1. nvidia/Cosmos-Policy-ALOHA-Predict2-2B              -- the acting policy checkpoint
#   2. nvidia/Cosmos-Policy-ALOHA-Planning-Model-Predict2-2B -- the SEPARATE, dedicated
#      value-function scorer, fine-tuned on 648 real rollouts pooled from 5 different
#      policies (Cosmos Policy, pi05, pi0, OpenVLA-OFT+, Diffusion Policy) -- NOT the same
#      training as the base policy checkpoint. This is what makes ALOHA's validated
#      "planning" mode different from just reusing a policy checkpoint as its own scorer.
# Plus the two small support files the deploy.py usage examples reference: the ALOHA
# dataset stats json and t5 text embeddings pkl, both served from the POLICY repo.
#
# Mirrors setup_cosmos_policy_uv.sh's own pattern exactly (same helpers, same env
# handling: source the pinned-HF_HOME env file, then unset HF_HUB_OFFLINE=1 for the
# duration of this script only, since normal training/eval jobs need it back on so
# nothing re-hits the network at runtime).
#
# Run this from the LOGIN NODE (network I/O only, no GPU/heavy compute -- same class of
# operation as the original LIBERO teacher prefetch in setup_cosmos_policy_uv.sh).
#
# Usage:
#   bash setup_aloha_planning_checkpoints.sh

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/cosmos_policy" && pwd)"
ENV_FILE="$REPO_DIR/activate_cuda_env.sh"
VENV_ACTIVATE="$REPO_DIR/.venv/bin/activate"

if [ ! -f "$ENV_FILE" ]; then
  echo "Error: $ENV_FILE not found -- run this from the repo root (next to cosmos_policy/)." >&2
  exit 1
fi
if [ ! -f "$VENV_ACTIVATE" ]; then
  echo "Error: $VENV_ACTIVATE not found -- expected the project venv at cosmos_policy/.venv." >&2
  exit 1
fi

# Both are required: the venv puts `cosmos_policy` (and its deps) on the path, activate_cuda_env.sh
# pins HF_HOME/CUDA env vars. Order matches how submit_sweep.py is invoked elsewhere in this repo.
# shellcheck disable=SC1090
source "$VENV_ACTIVATE"
# shellcheck disable=SC1090
source "$ENV_FILE"
# ENV_FILE pins HF_HOME (so downloads land in the shared hf_cache/, not a stray default
# cache) but also sets HF_HUB_OFFLINE=1 for normal runtime use. Unset it here so this
# script's live downloads actually reach the network.
unset HF_HUB_OFFLINE

echo "HF_HOME=$HF_HOME"
echo "Fetching ALOHA policy + planning-model checkpoints into hf_cache..."
echo

python3 - <<'PYEOF'
import sys
from cosmos_policy.experiments.robot.cosmos_utils import download_hf_checkpoint, download_hf_file

POLICY_REPO = "nvidia/Cosmos-Policy-ALOHA-Predict2-2B"
PLANNING_REPO = "nvidia/Cosmos-Policy-ALOHA-Planning-Model-Predict2-2B"

targets = [
    ("checkpoint", POLICY_REPO),
    ("checkpoint", PLANNING_REPO),
    ("file", f"{POLICY_REPO}/aloha_dataset_statistics.json"),
    ("file", f"{POLICY_REPO}/aloha_t5_embeddings.pkl"),
]

failures = []
for kind, target in targets:
    print(f"--- {kind}: {target} ---")
    try:
        if kind == "checkpoint":
            path = download_hf_checkpoint(target)
        else:
            path = download_hf_file(target)
        print(f"    -> {path}")
    except Exception as e:  # noqa: BLE001 -- report all failures, don't stop at the first
        print(f"    FAILED: {e}", file=sys.stderr)
        if "gated" in str(e).lower() or "403" in str(e) or "401" in str(e):
            repo_id = target.split("/aloha_", 1)[0] if "/aloha_" in target else target
            print(
                f"    If this is a gated-repo error: request access at "
                f"https://huggingface.co/{repo_id} , then re-run this script.",
                file=sys.stderr,
            )
        failures.append(target)

print()
if failures:
    print(f"Done with {len(failures)} failure(s): {failures}", file=sys.stderr)
    sys.exit(1)
print("All ALOHA planning-config assets fetched successfully.")
PYEOF

echo
echo "Next: run the 'main planning config' example from deploy.py's docstring, pointing"
echo "--ckpt_path at $POLICY_REPO and --planning_model_ckpt_path at $PLANNING_REPO."
