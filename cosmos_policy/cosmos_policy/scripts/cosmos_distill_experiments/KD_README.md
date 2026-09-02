# KD static sweep: full path from a clean checkout to the six launched variants + baselines

This walks through every step actually needed to get from a fresh checkout to (a) the six
`kd_static_500m_std_0_05_*` training jobs (3 agent objectives x 2 data variants) and (b) the
non-KD `baseline_*` comparison runs they're read against, in the order things actually have to
happen. It's the "how do I reproduce this" runbook `README.md` doesn't fully spell out
end-to-end -- that file covers the individual pieces; this stitches them into one sequence.

All commands run from `cosmos_policy/` (the repo clone root) with `.venv/bin/python`, unless noted
otherwise. Steps 0-3 are one-time (needed once ever, not once per variant). Step 2 downloads real
LIBERO benchmark data and can take a while. Steps 4-7 launch real Slurm jobs and consume GPU time.

---

## 0. Environment

**Cleanest path: run `../../../setup_cosmos_policy_uv.sh` from the repo root.** It builds the venv
with the right CUDA extra, writes `activate_cuda_env.sh`, and pre-fetches every HF checkpoint the
config registry needs (see below). If you already have the repo + `$COSMOS_POLICY_STORAGE` data in
place (e.g. a cluster migration) and just need the missing pieces, here they are, in order:

### 0a. Build the venv WITH the CUDA extra + task group

```bash
cd cosmos_policy
uv sync --extra cu130 --group libero --python 3.10
```

Plain `uv sync` is **not enough** -- `transformer-engine` / `flash-attn` / the pinned CUDA libs
only exist under the `cu128` / `cu130` extras (mutually exclusive; see `pyproject.toml`). Without
one, the very first training entrypoint dies at config-load time with
`ModuleNotFoundError: No module named 'transformer_engine'` (`_src/imaginaire/utils/fused_adam.py`
imports it unconditionally). `cu130` (torch 2.9 + CUDA 13) is what Vulcan's L40S driver wants and
what `setup_cosmos_policy_uv.sh` pins; `cu128` (torch 2.7) was the rrg-gberseth setup.

### 0b. Source the env, and clear the CUDA .so shim dir after any CUDA-lib change

```bash
rm -rf .cuda_so_shims        # only needed after a uv sync that changed the nvidia-* libs
source activate_cuda_env.sh
.venv/bin/python -c "import torch, transformer_engine, megatron.core; print(torch.__version__, 'ok')"
```

`activate_cuda_env.sh` rebuilds `.cuda_so_shims/` (unversioned `.so` symlinks the pip CUDA wheels
omit) every time it's sourced, but its `[[ -e link ]] || ln -s` guard is false for a *dangling*
link, so shims left over from a previous CUDA set produce a burst of `ln: File exists` errors and
can misdirect `dlopen`. `rm -rf .cuda_so_shims` first and they rebuild clean.

`activate_cuda_env.sh` also sets the per-cluster knobs every `submit_sweep.py` run reads AT SUBMIT
TIME -- source it before dry runs too, or the generated sbatch gets the wrong paths/GPU:

| env var | drives |
|---|---|
| `COSMOS_POLICY_STORAGE` | `data_root` / `output_root` / `kd_inits` / `hf_cache` |
| `COSMOS_POLICY_ACCOUNT` | `#SBATCH --account` |
| `COSMOS_POLICY_GPU_TYPE` | `#SBATCH --gres=gpu:<type>:<n>` (`l40s` here, was `h100` on rrg-gberseth) |

Per-run overrides still win over those env defaults, e.g. `runs.<name>.gpu_type=h100
runs.<name>.gpus=2` on the `submit_sweep.py` CLI (the torchrun path passes `gpus` through as
`--nproc_per_node`, so >1 there launches that many DDP ranks).

### 0c. Pre-fetch every HF checkpoint the teacher path needs -- one script

`../../../download_cosmos_policy_teacher.sh` (run from a **login node** -- compute nodes have no
internet) fetches all of it into `$COSMOS_POLICY_STORAGE/hf_cache` (the pinned `HF_HOME`), each into
the exact sub-location its runtime resolver looks in, and then re-checks every asset resolves with
`HF_HUB_OFFLINE=1`:

```bash
bash download_cosmos_policy_teacher.sh
```

| asset | resolver | lands in |
|---|---|---|
| `nvidia/Cosmos-Policy-LIBERO-Predict2-2B` (the teacher, ~4 GB, ungated) | `cosmos_utils.download_hf_checkpoint` -> `snapshot_download(cache_dir=HF_HOME)` | `$HF_HOME/models--nvidia--Cosmos-Policy-LIBERO-Predict2-2B/` |
| `Cosmos-Predict2-2B-Video2World/model-480p-16fps.pt` (~3.9 GB, **gated** auto-approve) | `checkpoint_db.get_checkpoint_by_hf` -> `hf_hub_download` (default cache) | `$HF_HOME/hub/models--.../` |
| `Cosmos-Predict2-2B-Video2World/tokenizer/tokenizer.pth` (~0.5 GB, gated; lazy) | same | `$HF_HOME/hub/models--.../` |
| `Cosmos-Policy-ALOHA-Predict2-2B.pt` (~3.9 GB, ungated) | same | `$HF_HOME/hub/models--.../` |

The two `cache_dir` conventions genuinely differ (`$HF_HOME/` vs `$HF_HOME/hub/`) -- a plain
`hf download <repo>` puts the teacher in `hub/`, where `download_hf_checkpoint`'s offline lookup
won't find it. The script uses `snapshot_download(cache_dir=HF_HOME)` for the teacher to match.

`config/experiment/cosmos_policy_experiment_configs.py` calls `get_checkpoint_path("hf://...")` at
**module level** (every experiment registers eagerly on import), so `run_train` / any sweep
entrypoint hits the base-model/ALOHA files the moment the config loads -- skip the prefetch and it's
`LocalEntryNotFoundError` / `OfflineModeIsEnabled` from `checkpoint_db.get_checkpoint_by_hf`. On a
`401`/`403` for the gated Video2World repo: `.venv/bin/hf auth login` (read token from
https://huggingface.co/settings/tokens), then re-run the script.

### 0d. Validate the env with a 1-iteration run before launching anything real

```bash
salloc --account=$COSMOS_POLICY_ACCOUNT --gres=gpu:l40s:1 --cpus-per-task=8 --mem=64G --time=1:00:00
# then, on the node:
export IMAGINAIRE_OUTPUT_ROOT=$COSMOS_POLICY_STORAGE/cosmos_dit_wm_output
export WANDB_MODE=disabled        # wandb is off everywhere; submit_sweep.py bakes this into every sbatch
.venv/bin/torchrun --nproc_per_node=1 --master_port=12345 \
  -m cosmos_policy.scripts.cosmos_distill_experiments.run_train \
  --config=cosmos_policy/config/config.py -- \
  experiment="cosmos_kd_student_500m_libero" job.name=smoke_1iter_500m job.wandb_mode=disabled \
  trainer.max_iter=1 trainer.logging_iter=1 checkpoint.save_iter=1 \
  dataloader_train.batch_size=8 dataloader_train.num_workers=0 dataloader_train.persistent_workers=false \
  dataloader_train.dataset.data_dir=$COSMOS_POLICY_STORAGE/LIBERO-Cosmos-Policy/success_only/libero_object_regen \
  dataloader_train.dataset.rollout_data_dir="" \
  checkpoint.load_path=$COSMOS_POLICY_STORAGE/kd_inits/student_init_500m_dcp
```

### 0e. LIBERO sim setup (only for the `*_eval` companions / any `run_libero_eval`)

The `libero` package needs two things the training path doesn't, and every eval subprocess fails
silently without them (`eval_results.csv` fills with `status=failed`, `success_rate` `nan`;
details in `<eval run_dir>/eval_failure_iter_*.log`):

```bash
# 1. ~/.libero/config.yaml -- else `import libero.libero` hits an interactive input() prompt at
#    import and dies with EOFError in a batch job. Answering "N" writes the default (in-package
#    bddl_files/init_files, assets under ~/.cache/libero):
printf 'N\n' | .venv/bin/python -c "import libero.libero"

# 2. LIBERO sim assets (~400 MB, object meshes/scenes/textures) -- compute nodes are
#    HF_HUB_OFFLINE=1 so they can't self-download. Pre-fetch on a login node:
HF_HUB_OFFLINE=0 .venv/bin/python -c "
from libero.libero.utils.download_utils import download_assets_from_huggingface
print(download_assets_from_huggingface())"
```

`setup_cosmos_policy_uv.sh` already does step 2; step 1's config file is machine-local (`~/.libero/`)
and does not travel with a repo/storage migration, so re-do it on each new cluster.

## 1. Teacher checkpoint + T5 text embeddings

**Teacher checkpoint** (`nvidia/Cosmos-Policy-LIBERO-Predict2-2B`, the released 2B LIBERO policy)
downloads automatically the first time anything calls `load_teacher`/`load_policy_model`
(`kd/teacher_loader.py`'s `resolve_checkpoint_path` -> `download_hf_checkpoint`) -- but compute
nodes have no internet, so **step 0c's `download_cosmos_policy_teacher.sh` already fetched it** (to
`$HF_HOME/models--nvidia--Cosmos-Policy-LIBERO-Predict2-2B/`, the location `download_hf_checkpoint`
resolves against -- not `hub/`). Nothing more to do here for the teacher.

At sbatch time `submit_sweep.py`'s `TEACHER_CHECKPOINT_PREFETCH` step then copies the whole
`hf_cache` to `$SLURM_TMPDIR` for any KD-training/build run (see that constant's docstring for why
the whole cache, not just this repo). Baselines (step 7) don't load the teacher at all.

**T5 text embeddings** (`t5_embeddings.pkl`, one embedding per unique LIBERO task instruction --
shared across every suite/build, not per-task or per-build):

```bash
.venv/bin/python -m cosmos_policy.datasets.save_libero_t5_text_embeddings \
  --data_dir $COSMOS_POLICY_STORAGE/LIBERO-Cosmos-Policy/success_only
```

This is what every `conf/runs/*.yaml`'s `t5_text_embeddings_path` field points at
(`.../success_only/t5_embeddings.pkl`).

The encoder it runs is `google-t5/t5-11b` (`CosmosT5TextEncoder` in
`_src/predict2/inference/get_t5_emb.py`), which HF downloads on first use -- **only needed to
(re)generate the .pkl; nothing at train time touches it.** Three gotchas on this cluster:

- **Cache location.** Put it in the same `$COSMOS_POLICY_STORAGE/hf_cache` (`= HF_HOME` from
  `activate_cuda_env.sh`) as every other checkpoint -- NOT the default `~/.cache/huggingface`
  (`pytorch_model.bin` is ~45 GB and will blow the 50 GB `$HOME` quota).
- **Compute nodes have no internet.** Pre-fetch once from a login node (skip the 45 GB TF
  checkpoint you don't need):
  ```bash
  HF_HOME=$COSMOS_POLICY_STORAGE/hf_cache HF_HUB_OFFLINE=0 .venv/bin/python -c "
  from huggingface_hub import snapshot_download
  snapshot_download('google-t5/t5-11b', allow_patterns=['*.json','*.model','pytorch_model.bin'])"
  ```
  A partial/interrupted download shows up later as a cryptic `TypeError: not a string` from
  SentencePiece (the tokenizer files -- `spiece.model` / `tokenizer.json` -- didn't come down and
  `transformers` silently fell back to the slow tokenizer with `vocab_file=None`). Re-run the
  fetch; it resumes.
- **Host RAM.** `T5EncoderModel.from_pretrained` reads the whole 45 GB fp32 state dict into RAM
  before discarding the decoder half -- a `--mem=32G` job OOM-kills mid-load; use `--mem=90G`.

`get_text_embedding` hardcodes `device="cuda"`, so the actual embedding pass needs a GPU (a short
single-L40S Slurm job -- there are only ~10 unique instructions per suite). `activate_cuda_env.sh`
already sets `HF_HOME` + `HF_HUB_OFFLINE=1`, so once pre-fetched it runs offline.

## 2. Regenerate the LIBERO datasets (once per suite)

The raw LIBERO benchmark release (50 demos/task) isn't usable as-is -- `regenerate_libero_dataset.py`
(vendored NVIDIA code, unmodified) replays every demo in simulation, filters out ones that fail to
replay, filters no-op actions, and re-renders at 256x256 (see its own module docstring). Run once
per suite used by this sweep (only `libero_object` is actually needed for the six variants below,
but all four are needed for the full-suite baseline/tiny-DiT jobs elsewhere in this folder):

```bash
.venv/bin/python -m cosmos_policy.experiments.robot.libero.regenerate_libero_dataset \
  --libero_task_suite libero_object \
  --libero_raw_data_dir <path to the raw LIBERO release's libero_object demos> \
  --libero_target_dir $COSMOS_POLICY_STORAGE/LIBERO-Cosmos-Policy/success_only/libero_object_regen \
  --data_collection True --jpeg_compress True --deterministic True
```

> Not verified in this session against the paper's own stated recipe (arxiv.org/abs/2601.16163,
> Sec 5.1): it says unsuccessful demos should be filtered **only for policy training**, with the
> full unfiltered set used for world-model/value-function training. This script filters
> unconditionally, and every dataset below traces back to this one filtered output. See
> `unresolved_journal.txt` for the open item this created.

`dataset_statistics.json`/`dataset_statistics_post_norm.json` are computed automatically the first
time anything loads this directory (`load_or_compute_dataset_statistics`) -- no separate step.

## 3. One-time: teacher-derived student init (500M)

```bash
.venv/bin/python -m cosmos_policy.scripts.cosmos_distill_experiments.kd.init_student_from_teacher \
  --student_size 500m \
  --out $COSMOS_POLICY_STORAGE/kd_inits/student_init_500m.pt \
  --dcp_out $COSMOS_POLICY_STORAGE/kd_inits/student_init_500m_dcp
```

CPU only, no GPU/Slurm needed -- a one-time weight copy (teacher's own block weights, matching
shapes, no projection), not training. `--out`'s flat `.pt` is what `kd_static_*.yaml`'s own
`student_init_path` field reads directly; `--dcp_out`'s directory is what the plain torchrun path
(`baseline_500m_train.yaml`'s `checkpoint.load_path`) needs instead (DCP vs flat `.pt` loader
routing -- see `teacher_loader.py`'s docstring). Both come from one call so they're guaranteed to
be the exact same initialization.

## 4. Build the three KD datasets (once, shared by all six variants)

All three read from `libero_object_regen` (real demo + real rollout data) and query the teacher;
they differ in what they ask the teacher to produce -- see this folder's own explanation of the
distinction (real ground truth / teacher-native / perturbed-action counterfactual) if you need the
"why", not just the "what."

```bash
# 1. Real data + real ground truth (the term every one of the six variants shares)
.venv/bin/python -m cosmos_policy.scripts.cosmos_distill_experiments.submit_sweep \
  launch.only=[build_distill_dataset_std_0_05] launch.submit=true

# 2. Teacher-native synthetic (action/future-state/value all freely generated by the teacher)
.venv/bin/python -m cosmos_policy.scripts.cosmos_distill_experiments.submit_sweep \
  launch.only=[build_teacher_native_distill_dataset] launch.submit=true

# 3. Perturbed-action synthetic (real action perturbed + held fixed; teacher generates future-state/value)
.venv/bin/python -m cosmos_policy.scripts.cosmos_distill_experiments.submit_sweep \
  launch.only=[build_synthetic_distill_dataset] launch.submit=true
```

Each is a real GPU job (queries the teacher), none is a training run (no checkpoint/resume --
rerunning without `launch.wipe=true` overwrites from scratch). Outputs land at
`distill_dataset_dir`/`teacher_native_dataset_dir`/`synthetic_dataset_dir` as set in each build's
own `conf/runs/datasets/*.yaml`.

### 4b. Merge teacher-native + perturbed-action into the `_pa` dataset

The three `_syn_tn_pa` variants (`syn_tn_pa`/`av_syn_tn_pa`/`action_syn_tn_pa`) train against a
**merged** directory combining builds 2 and 3 above via symlinks -- `batch_prep.py`'s
`split_synthetic_action_distill_loss` docstring describes the expected shape (disjoint filenames,
since both builds independently produce `shard_00000.pt`, `shard_00001.pt`, ... and would collide
if merged as-is), but no committed script performs the merge itself.

> **Not verified against the original historical command** -- no build/merge script for this step
> was found in `kd/`, and the experiment journal doesn't record the exact command used when this
> was first built (predates the earliest journal entry). The following reproduces the shape
> `batch_prep.py` documents (disjoint filenames via a source-tagged prefix, both still matching the
> `shard_*.pt` glob `distill_dataset.py` expects):

```bash
MERGED=$COSMOS_POLICY_STORAGE/kd_synthetic_distill_dataset_merged_std_0_05
mkdir -p "$MERGED"
for f in $COSMOS_POLICY_STORAGE/kd_teacher_native_distill_dataset/shard_*.pt; do
  ln -s "$f" "$MERGED/tn_$(basename "$f")"
done
for f in $COSMOS_POLICY_STORAGE/kd_synthetic_distill_dataset_std_0_05/shard_*.pt; do
  ln -s "$f" "$MERGED/pa_$(basename "$f")"
done
```

## 5. Launch the six variants

Each is `run_type: kd_static` (`kd/train_kd_static[_action|_av].py`), one GPU, no teacher loaded
(all teacher-querying already happened in step 4). Every variant shares the same
`distill_dataset_dir` (step 4.1); `_syn_tn` variants add `synthetic_dataset_dir` = the
teacher-native build (4.2); `_syn_tn_pa` variants add the merged dataset (4b) instead.

```bash
.venv/bin/python -m cosmos_policy.scripts.cosmos_distill_experiments.submit_sweep launch.only=[kd_static_500m_std_0_05_action_syn_tn] launch.submit=true
.venv/bin/python -m cosmos_policy.scripts.cosmos_distill_experiments.submit_sweep launch.only=[kd_static_500m_std_0_05_action_syn_tn_pa] launch.submit=true
.venv/bin/python -m cosmos_policy.scripts.cosmos_distill_experiments.submit_sweep launch.only=[kd_static_500m_std_0_05_av_syn_tn] launch.submit=true
.venv/bin/python -m cosmos_policy.scripts.cosmos_distill_experiments.submit_sweep launch.only=[kd_static_500m_std_0_05_av_syn_tn_pa] launch.submit=true
.venv/bin/python -m cosmos_policy.scripts.cosmos_distill_experiments.submit_sweep launch.only=[kd_static_500m_std_0_05_syn_tn] launch.submit=true
.venv/bin/python -m cosmos_policy.scripts.cosmos_distill_experiments.submit_sweep launch.only=[kd_static_500m_std_0_05_syn_tn_pa] launch.submit=true
```

Or all six in one call: `launch.only=[kd_static_500m_std_0_05_action_syn_tn,kd_static_500m_std_0_05_action_syn_tn_pa,kd_static_500m_std_0_05_av_syn_tn,kd_static_500m_std_0_05_av_syn_tn_pa,kd_static_500m_std_0_05_syn_tn,kd_static_500m_std_0_05_syn_tn_pa]`.

**Historical note**: the real launch took 4 attempts before all six were healthily running (1st:
OOM at `mem: "80G"`; 2nd: `mem: "150G"`, wrongly cancelled on a false-positive OOM extrapolation;
3rd: `mem: "150G"`, actually ran to completion/timeout; 4th: resumed the 3 that timed out, added
the six `*_eval.yaml` continuous-eval companions below). `mem: "150G"` is already baked into each
run's yaml from that experience -- a fresh launch shouldn't need to repeat it. Full detail in
`kd_static_syn_tn_launch_2026-08-15.txt` and `experiment_journal.txt`'s 2026-08-15/17 entries.

## 6. Continuous per-checkpoint eval (optional, but how the actual results were produced)

```bash
.venv/bin/python -m cosmos_policy.scripts.cosmos_distill_experiments.submit_sweep \
  launch.only=[kd_static_500m_std_0_05_action_syn_tn_eval,kd_static_500m_std_0_05_action_syn_tn_pa_eval,kd_static_500m_std_0_05_av_syn_tn_eval,kd_static_500m_std_0_05_av_syn_tn_pa_eval,kd_static_500m_std_0_05_syn_tn_eval,kd_static_500m_std_0_05_syn_tn_pa_eval] \
  launch.submit=true
```

Each polls its training run's `checkpoints/` and full-suite-evaluates every one restart-safely
(`kd/periodic_libero_eval_static.py`) -- a separate Slurm job per variant, never feeds back into
training. See this folder's `README.md` ("Continuous full-suite eval, decoupled from training")
for the restart-safety/cost details.

## 7. Baselines (non-KD comparison points)

None of these are part of the six KD variants -- they're the comparison runs that answer "does
dropping KD entirely still reach high success at this size / with this init?" Each goes through
the plain torchrun/Trainer path unmodified (`run_train.py`, no `kd/` training code at all), with
the ORIGINAL Cosmos Policy objective instead of `combined_kd_loss`.

**`baseline_500m_train`** -- same 500M size, same `libero_object_regen` data, same teacher-derived
init as the six KD variants above, minus the KD loss. The size-matched isolation the 1B baseline
below can't provide by itself.

```bash
.venv/bin/python -m cosmos_policy.scripts.cosmos_distill_experiments.submit_sweep \
  launch.only=[baseline_500m_train,baseline_500m_train_eval] launch.submit=true
```

**`baseline_1b_train`** -- same comparison at 1B instead of 500M (confounds size with loss-type,
since there's no `kd_static_1b_std_0_05` variant in the six above to pair it against 1:1). Needs
its own teacher-derived DCP init first, same pattern as step 3:

```bash
.venv/bin/python -m cosmos_policy.scripts.cosmos_distill_experiments.kd.init_student_from_teacher \
  --student_size 1b \
  --out $COSMOS_POLICY_STORAGE/kd_inits/student_init_1b.pt \
  --dcp_out $COSMOS_POLICY_STORAGE/kd_inits/student_init_1b_dcp

.venv/bin/python -m cosmos_policy.scripts.cosmos_distill_experiments.submit_sweep \
  launch.only=[baseline_1b_train,baseline_1b_train_eval] launch.submit=true
```

**`baseline_1b_train_scratch`** -- identical to `baseline_1b_train` (same architecture/data/
update-count/loss) but with `checkpoint.load_path` left unset entirely, i.e. a genuine random
init instead of the teacher-derived one. Isolates whether the teacher-derived init is doing
anything for standard (non-KD) training at all, vs. training converging to a similar place
regardless of starting point. No extra prerequisite beyond step 0 -- an empty `load_path` is
already the proven from-scratch default path (`net_experiments.py`'s own registration defaults
`checkpoint.load_path=""` for exactly this reason).

```bash
.venv/bin/python -m cosmos_policy.scripts.cosmos_distill_experiments.submit_sweep \
  launch.only=[baseline_1b_train_scratch,baseline_1b_train_scratch_eval] launch.submit=true
```

Each was verified (not just assumed) to actually load its init rather than silently training from
scratch, via a `max_iter=1` smoketest confirming the "Resuming ckpt ..." log line before the real
run was launched -- see `experiment_journal.txt`'s 2026-08-25 entry.

---

## Known open items affecting this recipe

- The `success_only` demo-filtering issue in step 2 (paper says it should only apply to policy
  training, not world-model/value-function training) -- see `unresolved_journal.txt`.
- Step 4b's merge command is a reconstruction, not a verified historical command.
