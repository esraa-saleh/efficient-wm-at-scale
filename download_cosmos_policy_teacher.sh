#!/usr/bin/env bash
set -euo pipefail

# Download the Cosmos Policy LIBERO teacher checkpoint (nvidia/Cosmos-Policy-LIBERO-Predict2-2B)
# -- plus the base-model / tokenizer checkpoints its experiment config resolves at import/build
# time -- into this project's pinned Hugging Face cache, so KD training/eval jobs can load it as
# the teacher on compute nodes that have no internet.
#
# Everything lands under $HF_HOME (= $COSMOS_POLICY_STORAGE/hf_cache by default -- see
# activate_cuda_env.sh), in the exact sub-locations the runtime's own resolvers look in:
#   - cosmos_utils.download_hf_checkpoint() : $HF_HOME/models--nvidia--Cosmos-Policy-LIBERO-Predict2-2B/
#   - checkpoint_db.get_checkpoint_by_hf()  : $HF_HOME/hub/models--.../   (base model, tokenizer, ALOHA base)
# This is the same prefetch setup_cosmos_policy_uv.sh does; broken out so it can be re-run on its
# own on a new cluster (e.g. after the venv/env were copied over rather than re-installed).
#
# Run ONCE per cluster, from a LOGIN node -- compute nodes here can't reach Hugging Face.
# Idempotent and resumable: safe to re-run if interrupted.
#
# Usage:
#   bash download_cosmos_policy_teacher.sh [REPO_DIR]   # REPO_DIR defaults to ./cosmos_policy
#
# Gated repos: on a 401/403, authenticate and re-run --
#   cosmos_policy/.venv/bin/hf auth login      # needs a read token: https://huggingface.co/settings/tokens

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${1:-$SCRIPT_DIR/cosmos_policy}"
if [[ "$REPO_DIR" != /* ]]; then REPO_DIR="$(pwd)/$REPO_DIR"; fi
VENV_PY="$REPO_DIR/.venv/bin/python"
ENV_FILE="$REPO_DIR/activate_cuda_env.sh"

[[ -x "$VENV_PY" ]] || { echo "Error: venv python not found at $VENV_PY (run setup_cosmos_policy_uv.sh first)." >&2; exit 1; }
[[ -f "$ENV_FILE" ]] || { echo "Error: $ENV_FILE not found (run setup_cosmos_policy_uv.sh first)." >&2; exit 1; }

# Sourcing activate_cuda_env.sh pins HF_HOME (from COSMOS_POLICY_STORAGE) and also sets
# HF_HUB_OFFLINE=1 for later runtime use -- undo that here, since populating the cache is the
# whole point of this script and needs live network.
# shellcheck source=/dev/null
source "$ENV_FILE"
unset HF_HUB_OFFLINE

echo "Storage root : ${COSMOS_POLICY_STORAGE:-<unset>}"
echo "HF_HOME      : $HF_HOME"
mkdir -p "$HF_HOME"

# `hf auth login` in a plain shell writes its token to the DEFAULT HF cache, which this pinned
# HF_HOME never sees -- copy it over if present (same fixup setup_cosmos_policy_uv.sh does).
DEFAULT_HF_TOKEN="$HOME/.cache/huggingface/token"
if [[ ! -f "$HF_HOME/token" && -f "$DEFAULT_HF_TOKEN" ]]; then
  echo "Copying Hugging Face auth token from $DEFAULT_HF_TOKEN into $HF_HOME ..."
  cp "$DEFAULT_HF_TOKEN" "$HF_HOME/token"
fi

"$VENV_PY" - <<'PY'
import os
import sys

from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.errors import GatedRepoError

HF_HOME = os.environ["HF_HOME"]
TEACHER_REPO = "nvidia/Cosmos-Policy-LIBERO-Predict2-2B"

# (repo_id, filename) checkpoints the teacher's experiment config resolves via
# checkpoint_db.get_checkpoint_by_hf() -- default cache, i.e. $HF_HOME/hub. Single files, not
# whole repos (Cosmos-Predict2-2B-Video2World is ~54 GB). Keep this list in sync with
# config/experiment/cosmos_policy_experiment_configs.py + config/defaults/tokenizer.py.
BASE_FILES = [
    ("nvidia/Cosmos-Predict2-2B-Video2World", "model-480p-16fps.pt"),
    ("nvidia/Cosmos-Predict2-2B-Video2World", "tokenizer/tokenizer.pth"),
    ("nvidia/Cosmos-Policy-ALOHA-Predict2-2B", "Cosmos-Policy-ALOHA-Predict2-2B.pt"),
]


def _die_gated(repo: str) -> None:
    sys.exit(
        f"\nError: '{repo}' is gated or requires authentication.\n"
        f"  1. Request access: https://huggingface.co/{repo}  (click 'Agree and access repository')\n"
        f"  2. Authenticate:   {sys.prefix}/bin/hf auth login\n"
        "Then re-run this script.\n"
    )


print(f"\n[1/3] Teacher checkpoint: {TEACHER_REPO}  (~4 GB)")
try:
    path = snapshot_download(repo_id=TEACHER_REPO, cache_dir=HF_HOME)
except GatedRepoError:
    _die_gated(TEACHER_REPO)
print(f"      -> {path}")

print(f"\n[2/3] Teacher-repo eval assets (dataset stats + T5 embeddings)")
for filename in ("libero_dataset_statistics.json", "libero_t5_embeddings.pkl"):
    path = hf_hub_download(repo_id=TEACHER_REPO, filename=filename, cache_dir=HF_HOME)
    print(f"      {filename} -> {path}")

print(f"\n[3/3] Base-model / tokenizer checkpoints resolved by the experiment config")
for repo_id, filename in BASE_FILES:
    try:
        path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="model")
    except GatedRepoError:
        _die_gated(repo_id)
    print(f"      {repo_id}/{filename} -> {path}")

print(f"\nDone. All teacher assets cached under {HF_HOME}")
PY

echo
echo "Verifying offline resolution (how a compute-node KD job will see the cache) ..."
HF_HUB_OFFLINE=1 "$VENV_PY" - <<'PY'
import os

from huggingface_hub import hf_hub_download, snapshot_download

HF_HOME = os.environ["HF_HOME"]
snapshot_download(repo_id="nvidia/Cosmos-Policy-LIBERO-Predict2-2B", cache_dir=HF_HOME)
for repo_id, filename in [
    ("nvidia/Cosmos-Predict2-2B-Video2World", "model-480p-16fps.pt"),
    ("nvidia/Cosmos-Predict2-2B-Video2World", "tokenizer/tokenizer.pth"),
    ("nvidia/Cosmos-Policy-ALOHA-Predict2-2B", "Cosmos-Policy-ALOHA-Predict2-2B.pt"),
]:
    hf_hub_download(repo_id=repo_id, filename=filename, repo_type="model")
print("OK - every teacher asset resolves with HF_HUB_OFFLINE=1; KD jobs can load the teacher.")
PY
