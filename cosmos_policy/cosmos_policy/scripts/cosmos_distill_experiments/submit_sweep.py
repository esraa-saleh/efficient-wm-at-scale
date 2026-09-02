#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Same generator pattern as cosmos_dit_wm's own submit_sweep.py (moved to ../../../draft_code/
cosmos_dit_wm/submit_sweep.py -- see the repo's own dependency audit), scoped to just this job's
variants (the real run + its smoketest, or any other run you add under conf/runs/) --
deliberately self-contained rather than adding entries to that shared script, so this folder never
has to touch (or wait its turn behind) submissions for the rest of the cosmos_dit_wm family.
This folder's run_train.py is likewise its own copy, not an import of cosmos_dit_wm's -- see that
file's own docstring.

Every *output* a run produces -- checkpoints, loss CSV, wandb offline run, DeviceMonitor, the
resolved-config snapshot, and now Slurm's own stdout/stderr too -- lands under one project-storage
folder per run: job_output_dir(run.output_root, run.job_name), i.e.
{run.output_root}/cosmos_policy/cosmos_v2_finetune/{run.job_name}/. Slurm logs used to go to
./logs/ next to this script (on $HOME, small quota, and separate from everything else the run
produced); they're now under {job_output_dir}/slurm/ instead, alongside the rest of that run's
output. One consequence of this: launch.wipe=true now also deletes that run's Slurm log history,
not just its checkpoints/wandb/etc -- that's intentional (one wipe cleans the whole run), but worth
knowing before you wipe a run whose stdout/stderr you still wanted.

train.sbatch / smoketest.sbatch themselves are deliberately NOT moved into that tree -- they're
generated *launcher code* (reviewable in this folder, in your editor, alongside sweep.py/conf/),
not run output, so they stay here regardless of which account's output_root points at.

Unlike that shared script (a hardcoded Python RUNS list), every field here comes from conf/runs/*.yaml
(validated against sweep.py's RunConfig schema, composed into SweepConfig.runs by conf/config.yaml),
so it's overridable from the command line the same way the rest of this repo overrides training
config -- no editing Python required for a one-off change, and no editing Python required to add a
new run either (add a new conf/runs/<name>.yaml + a `runs@runs.<name>: <name>` line in
conf/config.yaml).

train.sbatch and smoketest.sbatch are *generated*, not hand-maintained -- this file + conf/runs/*.yaml
are their single source of truth. Re-run this after editing a conf/runs/*.yaml file (or via a CLI
override) to regenerate them.

DEFAULT IS DRY RUN: this only (re)writes the sbatch scripts and prints what it *would* submit.
Pass launch.submit=true to actually queue them.

Usage:
    # Regenerate train.sbatch / smoketest.sbatch and show what would be submitted:
    python -m cosmos_policy.scripts.cosmos_distill_experiments.submit_sweep

    # Regenerate + actually submit everything:
    python -m cosmos_policy.scripts.cosmos_distill_experiments.submit_sweep launch.submit=true

    # Just the smoketest, with a one-off override (no need to edit sweep.py for this):
    python -m cosmos_policy.scripts.cosmos_distill_experiments.submit_sweep \\
      launch.only=[smoketest] launch.submit=true runs.smoketest.max_iter=5

    # Smoketest is not idempotent by default -- it reuses job.name, so job.name auto-resume means a
    # second run just picks up wherever the first one's checkpoint left off (and short-circuits to
    # an instant no-op once its checkpoint iteration >= max_iter). launch.wipe=true deletes the
    # selected run(s)' existing checkpoint dir first, so it actually starts from iteration 0 again --
    # this prints exactly what's about to be deleted and requires typing "yes" at an interactive
    # prompt before it actually removes anything (aborts safely if there's no TTY to prompt on):
    python -m cosmos_policy.scripts.cosmos_distill_experiments.submit_sweep \\
      launch.only=[smoketest] launch.wipe=true launch.submit=true
"""

import pathlib
import shutil
import subprocess

import hydra
from omegaconf import OmegaConf

from cosmos_policy.scripts.cosmos_distill_experiments.sweep import RunConfig, SweepConfig

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
# .../cosmos_policy/cosmos_policy/scripts/cosmos_distill_experiments -> .../cosmos_policy (the repo
# clone root containing .venv/, activate_cuda_env.sh). Derived, not a conf/ value: it's a
# structural fact about where this repo was cloned, not a per-account setting -- see RunConfig's
# data_root/output_root fields for the values that ARE per-account.
REPO_ROOT = SCRIPT_DIR.parent.parent.parent


def resolve_data_dir(data_dir: str, data_root: str) -> str:
    return data_dir if data_dir.startswith("/") else f"{data_root}/{data_dir}"


def resolve_dataset_dir(suites: list, data_root: str, output_root: str) -> pathlib.Path:
    """A single suite passes straight through as an absolute path -- LIBERODataset.data_dir already
    accepts exactly that. Multiple suites can't: LIBERODataset takes one data_dir, not a list. So
    for >1 suite, build (or reuse) a small directory of per-suite symlinks under
    output_root/dataset_combos/ and point data_dir at THAT instead --
    get_hdf5_files(data_dir) (datasets/dataset_utils.py) walks with followlinks=True, so it reads
    the symlinked suites as their union. Deterministically named from the sorted suite list, so
    re-running this for the same suites reuses (not duplicates) the same combo dir."""
    resolved = [pathlib.Path(resolve_data_dir(s, data_root)) for s in suites]
    if len(resolved) == 1:
        return resolved[0]

    names = sorted(p.name for p in resolved)
    if len(set(names)) != len(names):
        raise SystemExit(f"suites must have distinct directory names to combine, got: {sorted(suites)}")

    combo_dir = pathlib.Path(output_root) / "dataset_combos" / "+".join(names)
    combo_dir.mkdir(parents=True, exist_ok=True)
    for suite_path in resolved:
        link = combo_dir / suite_path.name
        if not link.is_symlink():
            link.symlink_to(suite_path)
    return combo_dir


def _write_generated_params_yaml(values: dict, path: pathlib.Path) -> pathlib.Path:
    """Writes `values` as a plain YAML file at `path`, the same shape kd/params.py's
    load_params/load_static_params/load_build_params already know how to OmegaConf.load() -- this
    is what makes conf/runs/<name>.yaml the one place any of these run types' hyperparameters get
    edited, without touching train_kd.py/train_kd_static.py/build_distill_dataset.py's own
    `--kd_params`/`--static_params`/`--build_params` CLI at all: they still just load a file, it's
    only generated now instead of hand-maintained. Path-valued fields (data_dir,
    t5_text_embeddings_path, rollout_data_dir, out_dir/distill_dataset_dir, run_dir) are
    deliberately NOT included here -- those still arrive via the same CLI overrides
    build_sbatch_script() already passed before this existed (see e.g. train_kd.py's own --data_dir/
    --run_dir flags), since they're resolved per-run (dataset_combos/, job_output_dir(...)) rather
    than being plain values a user would tune."""
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(values), path)
    return path


def _kd_live_params(run: RunConfig) -> dict:
    values = dict(
        student_net_experiment_name=run.student_net_experiment_name,
        student_init_path=run.student_init_path,
        lr=run.lr,
        max_iter=run.max_iter,
        batch_size=run.batch_size,
        log_every=run.log_every,
        checkpoint_every=run.checkpoint_every,
        teacher_device=run.teacher_device,
        student_device=run.student_device,
        ground_truth_loss_weight=run.ground_truth_loss_weight,
        seed=run.seed,
    )
    # Empty means "use teacher_loader.py's own default" -- KDParams itself already defaults these
    # the same way, so omitting them (rather than writing an empty string into the yaml) preserves
    # that default instead of overriding it with "".
    if run.teacher_experiment_name:
        values["teacher_experiment_name"] = run.teacher_experiment_name
    if run.teacher_checkpoint:
        values["teacher_checkpoint"] = run.teacher_checkpoint
    return values


def _kd_static_params(run: RunConfig) -> dict:
    return dict(
        student_net_experiment_name=run.student_net_experiment_name,
        student_init_path=run.student_init_path,
        student_device=run.student_device,
        batch_size=run.batch_size,
        lr=run.lr,
        max_iter=run.max_iter,
        log_every=run.log_every,
        checkpoint_every=run.checkpoint_every,
        ground_truth_loss_weight=run.ground_truth_loss_weight,
        seed=run.seed,
    )


def _build_distill_dataset_params(run: RunConfig) -> dict:
    values = dict(
        num_batches=run.num_batches,
        noise_draws_per_batch=run.noise_draws_per_batch,
        batch_size=run.batch_size,
        examples_per_shard=run.examples_per_shard,
        device=run.device,
        seed=run.seed,
    )
    if run.teacher_experiment_name:
        values["teacher_experiment_name"] = run.teacher_experiment_name
    if run.teacher_checkpoint:
        values["teacher_checkpoint"] = run.teacher_checkpoint
    return values


def _build_synthetic_distill_dataset_params(run: RunConfig) -> dict:
    values = dict(
        num_batches=run.num_batches,
        batch_size=run.batch_size,
        action_perturbation_std=run.action_perturbation_std,
        num_denoising_steps=run.num_denoising_steps,
        examples_per_shard=run.examples_per_shard,
        device=run.device,
        seed=run.seed,
    )
    if run.teacher_experiment_name:
        values["teacher_experiment_name"] = run.teacher_experiment_name
    if run.teacher_checkpoint:
        values["teacher_checkpoint"] = run.teacher_checkpoint
    return values


def _build_teacher_native_distill_dataset_params(run: RunConfig) -> dict:
    values = dict(
        num_batches=run.num_batches,
        batch_size=run.batch_size,
        num_denoising_steps=run.num_denoising_steps,
        examples_per_shard=run.examples_per_shard,
        device=run.device,
        seed=run.seed,
    )
    if run.teacher_experiment_name:
        values["teacher_experiment_name"] = run.teacher_experiment_name
    if run.teacher_checkpoint:
        values["teacher_checkpoint"] = run.teacher_checkpoint
    return values


def _confirm(prompt: str) -> bool:
    """Blocking yes/no prompt. Fails safe (returns False, i.e. "don't proceed") if there's no TTY
    to prompt on, rather than hanging or silently proceeding."""
    try:
        return input(prompt).strip().lower() == "yes"
    except EOFError:
        return False


def _existing_checkpoint_marker(run: RunConfig, run_dir: pathlib.Path) -> pathlib.Path:
    """Where to look for "does this run already have a checkpoint to resume from" -- the normal
    torchrun/Trainer path writes checkpoints/latest_checkpoint.txt; kd_live writes train_state.pt
    directly under run_dir instead (see kd/checkpoint_io.py's save_student/load_student_for_resume).
    kd_static ALSO ends up at checkpoints/latest_checkpoint.txt, same relative path as the torchrun
    default, but via a different mechanism (checkpoint_io.save_versioned_checkpoint -- every
    checkpoint kept, not overwritten, see train_kd_static.py's module docstring for why), so it
    falls through to the same `return` below rather than needing its own branch.
    build_distill_dataset/kd_static_eval runs have no checkpoint concept at all (not a training
    loop) -- see main()'s own dedicated handling for build_distill_dataset instead of this
    function; kd_static_eval's fall-through here is harmless (checkpoint_marker.exists() is always
    False for it, so the auto-resume messaging below just never fires). "kd_live_av"/"kd_live_action"
    write via the exact same checkpoint_io.save_student() call as "kd_live" (train_kd_av.py/
    train_kd_action.py mirror train_kd.py byte-for-byte on this path), so they take the same
    branch; "kd_static_av"/"kd_static_action" fall through the same way "kd_static" does, for the
    same reason.
    """
    if run.run_type in ("kd_live", "kd_live_av", "kd_live_action"):
        return run_dir / "train_state.pt"
    return run_dir / "checkpoints" / "latest_checkpoint.txt"


def job_output_dir(output_root: str, job_name: str) -> pathlib.Path:
    """{output_root}/cosmos_policy/cosmos_v2_finetune/{job_name} -- must match job.project /
    job.group / job.name exactly (see _src/imaginaire/config.py's JobConfig.path_local), since this
    is where the *training framework itself* actually writes checkpoints/wandb/etc, not an
    independently-chosen location. "cosmos_policy"/"cosmos_v2_finetune" come from job.project /
    job.group, set on the base experiment this job inherits from
    (config/experiment/cosmos_policy_experiment_configs.py) -- duplicated here, not derived, so if
    that base ever changes its job.group this needs updating too."""
    return pathlib.Path(output_root) / "cosmos_policy" / "cosmos_v2_finetune" / job_name


# run_types whose Python entry point actually loads the ~3.7GB teacher checkpoint (via
# teacher_loader.load_teacher -> download_hf_checkpoint -> huggingface_hub's snapshot_download) --
# see build_sbatch_script's TEACHER_CHECKPOINT_PREFETCH below for why this matters. kd_static* and
# kd_static_eval never load the teacher at all (see sweep.py's RunConfig.student_device
# docstring); the torchrun path loads its own base checkpoint through a different, unrelated
# mechanism (checkpoint.load_path), not this one.
TEACHER_LOADING_RUN_TYPES = frozenset(
    {
        "kd_live",
        "kd_live_av",
        "kd_live_action",
        "build_distill_dataset",
        "build_synthetic_distill_dataset",
        "build_teacher_native_distill_dataset",
    }
)

# Prepended (only for TEACHER_LOADING_RUN_TYPES) right before `source activate_cuda_env.sh`, so
# activate_cuda_env.sh's own `export HF_HOME="${COSMOS_POLICY_HF_HOME:-<shared Lustre path>}"`
# picks up this job's local override instead of its shared-storage default. Motivated by a real,
# measured problem: loading the teacher checkpoint took 9-10+ minutes in practice, NOT explained
# by raw filesystem throughput (a plain `dd` read of the checkpoint file hit 470 MB/s / 8.3s from
# the login node, but only 50 MB/s / 77.7s from an actual compute node under real cluster load --
# ~10x slower, evidently from shared Lustre/network contention specific to whichever node a job
# lands on, not anything about our own code). Copying the checkpoint once, up front, to
# $SLURM_TMPDIR (this cluster's fast local-NVMe per-job scratch, auto-cleaned at job exit) and
# redirecting HF's own cache lookup there means huggingface_hub's snapshot_download() finds it
# already present locally and never touches the network filesystem for it again during this job.
#
# Copies the WHOLE shared hf_cache/ dir (~9GB as of writing), not just the one model repo this
# folder's own scripts name directly -- confirmed the hard way: an earlier version of this that
# copied only `models--nvidia--Cosmos-Policy-LIBERO-Predict2-2B` (2.9GB) broke a real run, because
# the base experiment config ALSO resolves a second, separate HF repo
# (`nvidia/Cosmos-Predict2-2B-Video2World`, the pretrain-only checkpoint --
# cosmos_policy_experiment_configs.py's own `checkpoint.load_path` -- see teacher_loader.py's
# module docstring) that lived under hf_cache/hub/, not the one copied dir. Redirecting HF_HOME
# away from the shared cache made that second repo invisible, and since HF_HUB_OFFLINE is set in
# this environment, there was no network fallback -- the job crashed with
# `OfflineModeIsEnabled`/`LocalEntryNotFoundError`. Copying the whole directory (hub/ included)
# avoids leaving any HF-cached repo behind, whichever ones a given run type happens to touch.
#
# Fails safe: if $SLURM_TMPDIR isn't set, or the copy itself fails for any reason, this leaves
# HF_HOME pointing at the original shared location untouched (just slow, not broken) rather than
# ever redirecting to an incomplete/empty local cache dir that could strand a job with no
# checkpoint to load at all (compute nodes on this cluster have no general internet egress, so a
# fresh from-HuggingFace download isn't a fallback that would actually work here).
TEACHER_CHECKPOINT_PREFETCH = """# Prefetch the full HF cache to this node's fast local scratch -- see submit_sweep.py's
# TEACHER_CHECKPOINT_PREFETCH docstring for why (including why this copies the WHOLE cache dir,
# not just one model repo). Safe no-op if SLURM_TMPDIR is unset or the copy fails: HF_HOME then
# just keeps pointing at the original shared (slower) location.
if [ -n "${SLURM_TMPDIR:-}" ]; then
    _shared_hf_home="${COSMOS_POLICY_HF_HOME:-${COSMOS_POLICY_STORAGE:-/project/rrg-gberseth/esraa1/cosmos_policy_storage}/hf_cache}"
    _local_hf_home="$SLURM_TMPDIR/hf_cache"
    if cp -r "$_shared_hf_home" "$_local_hf_home" 2>/dev/null; then
        export COSMOS_POLICY_HF_HOME="$_local_hf_home"
        echo "Prefetched HF cache to local scratch: $_local_hf_home"
    else
        echo "WARNING: failed to prefetch HF cache to local scratch -- falling back to shared HF_HOME=$_shared_hf_home" >&2
    fi
fi
"""


def build_sbatch_script(run: RunConfig) -> str:
    run_dir = job_output_dir(run.output_root, run.job_name)
    # Slurm's own stdout/stderr, alongside everything else this run produces (checkpoints, loss
    # CSV, wandb, DeviceMonitor, config.yaml) -- see module docstring for why this moved off
    # $HOME/./logs/, and the launch.wipe consequence of that.
    log_dir = run_dir / "slurm"
    data_stage = ""  # set by the torchrun branch when run.stage_data_to_tmpdir is on

    if run.run_type == "kd_static":
        # Static-KD runs (kd/train_kd_static.py): no raw LIBERO data, no teacher -- trains purely
        # from a dataset build_distill_dataset.py already built once, so `suites`/`resolve_dataset_
        # dir` (which resolves *raw* LIBERO data, irrelevant here) is never called at all.
        # run_dir/params.yaml is generated fresh from RunConfig every time this runs -- it, not any
        # hand-maintained file, is train_kd_static.py's actual --static_params source of truth.
        params_path = _write_generated_params_yaml(_kd_static_params(run), run_dir / "params.yaml")
        cmd = (
            "python -m cosmos_policy.scripts.cosmos_distill_experiments.kd.train_kd_static "
            f"--static_params {params_path} "
            f"--distill_dataset_dir {run.distill_dataset_dir} "
            f'--synthetic_dataset_dir "{run.synthetic_dataset_dir}" '
            f"--run_dir {run_dir}"
        )
        gres = f"gpu:{run.gpu_type}:{run.gpus}"
    elif run.run_type == "kd_static_av":
        # Same as "kd_static" above, just dispatching to train_kd_static_av.py instead --
        # action/value-only loss (see that script's own module docstring), otherwise byte-for-byte
        # identical params/launch shape, so it reuses _kd_static_params() unchanged.
        params_path = _write_generated_params_yaml(_kd_static_params(run), run_dir / "params.yaml")
        cmd = (
            "python -m cosmos_policy.scripts.cosmos_distill_experiments.kd.train_kd_static_av "
            f"--static_params {params_path} "
            f"--distill_dataset_dir {run.distill_dataset_dir} "
            f'--synthetic_dataset_dir "{run.synthetic_dataset_dir}" '
            f"--run_dir {run_dir}"
        )
        gres = f"gpu:{run.gpu_type}:{run.gpus}"
    elif run.run_type == "kd_static_action":
        # Same as "kd_static" above, just dispatching to train_kd_static_action.py instead --
        # action-only loss (see that script's own module docstring), otherwise byte-for-byte
        # identical params/launch shape, so it reuses _kd_static_params() unchanged.
        params_path = _write_generated_params_yaml(_kd_static_params(run), run_dir / "params.yaml")
        cmd = (
            "python -m cosmos_policy.scripts.cosmos_distill_experiments.kd.train_kd_static_action "
            f"--static_params {params_path} "
            f"--distill_dataset_dir {run.distill_dataset_dir} "
            f'--synthetic_dataset_dir "{run.synthetic_dataset_dir}" '
            f"--run_dir {run_dir}"
        )
        gres = f"gpu:{run.gpu_type}:{run.gpus}"
    elif run.run_type == "kd_live":
        # KD runs (scripts/cosmos_distill_experiments/kd/): a plain `python -m ...kd.train_kd`
        # process, not torchrun/the imaginaire Trainer -- train_kd.py places the frozen teacher and
        # trainable student on two explicit CUDA devices itself, so this needs `gpus` (2) H100s
        # instead of 1, and no --nproc_per_node/config.py Hydra CLI overrides (train_kd.py takes its
        # own --kd_params file plus a few path overrides -- see its own module docstring). t5
        # embeddings path is always {data_root}/t5_embeddings.pkl regardless of single- vs.
        # multi-suite data_dir (data_root is always the shared "success_only" root -- see
        # RunConfig's data_root field), same reasoning kd/params.py's KDParams.t5_text_embeddings_path
        # documents for why this isn't derived from data_dir instead.
        data_dir = resolve_dataset_dir(run.suites, run.data_root, run.output_root)
        t5_text_embeddings_path = pathlib.Path(run.data_root) / "t5_embeddings.pkl"
        params_path = _write_generated_params_yaml(_kd_live_params(run), run_dir / "params.yaml")
        cmd = (
            "python -m cosmos_policy.scripts.cosmos_distill_experiments.kd.train_kd "
            f"--kd_params {params_path} "
            f"--data_dir {data_dir} "
            f"--t5_text_embeddings_path {t5_text_embeddings_path} "
            f'--rollout_data_dir "{run.rollout_data_dir}" '
            f'--synthetic_dataset_dir "{run.synthetic_dataset_dir}" '
            f"--run_dir {run_dir}"
        )
        gres = f"gpu:{run.gpu_type}:{run.gpus}"
    elif run.run_type == "kd_live_av":
        # Same as "kd_live" above, just dispatching to train_kd_av.py instead -- action/value-only
        # loss (see that script's own module docstring), otherwise byte-for-byte identical params/
        # launch shape, so it reuses _kd_live_params() unchanged.
        data_dir = resolve_dataset_dir(run.suites, run.data_root, run.output_root)
        t5_text_embeddings_path = pathlib.Path(run.data_root) / "t5_embeddings.pkl"
        params_path = _write_generated_params_yaml(_kd_live_params(run), run_dir / "params.yaml")
        cmd = (
            "python -m cosmos_policy.scripts.cosmos_distill_experiments.kd.train_kd_av "
            f"--kd_params {params_path} "
            f"--data_dir {data_dir} "
            f"--t5_text_embeddings_path {t5_text_embeddings_path} "
            f'--rollout_data_dir "{run.rollout_data_dir}" '
            f'--synthetic_dataset_dir "{run.synthetic_dataset_dir}" '
            f"--run_dir {run_dir}"
        )
        gres = f"gpu:{run.gpu_type}:{run.gpus}"
    elif run.run_type == "kd_live_action":
        # Same as "kd_live" above, just dispatching to train_kd_action.py instead -- action-only
        # loss (see that script's own module docstring), otherwise byte-for-byte identical
        # params/launch shape, so it reuses _kd_live_params() unchanged.
        data_dir = resolve_dataset_dir(run.suites, run.data_root, run.output_root)
        t5_text_embeddings_path = pathlib.Path(run.data_root) / "t5_embeddings.pkl"
        params_path = _write_generated_params_yaml(_kd_live_params(run), run_dir / "params.yaml")
        cmd = (
            "python -m cosmos_policy.scripts.cosmos_distill_experiments.kd.train_kd_action "
            f"--kd_params {params_path} "
            f"--data_dir {data_dir} "
            f"--t5_text_embeddings_path {t5_text_embeddings_path} "
            f'--rollout_data_dir "{run.rollout_data_dir}" '
            f'--synthetic_dataset_dir "{run.synthetic_dataset_dir}" '
            f"--run_dir {run_dir}"
        )
        gres = f"gpu:{run.gpu_type}:{run.gpus}"
    elif run.run_type == "build_distill_dataset":
        # One-shot data-generation job (kd/build_distill_dataset.py), not a training run -- no loss
        # curve, no resume, real output is run.distill_dataset_dir (the shards), not run_dir (which
        # here holds only Slurm logs + this generated params.yaml, for provenance).
        data_dir = resolve_dataset_dir(run.suites, run.data_root, run.output_root)
        t5_text_embeddings_path = pathlib.Path(run.data_root) / "t5_embeddings.pkl"
        params_path = _write_generated_params_yaml(_build_distill_dataset_params(run), run_dir / "params.yaml")
        task_names_flag = f"--task_names {' '.join(run.task_names)} " if run.task_names else ""
        cmd = (
            "python -m cosmos_policy.scripts.cosmos_distill_experiments.kd.build_distill_dataset "
            f"--build_params {params_path} "
            f"--data_dir {data_dir} "
            f"--t5_text_embeddings_path {t5_text_embeddings_path} "
            f'--rollout_data_dir "{run.rollout_data_dir}" '
            f"{task_names_flag}"
            f"--out_dir {run.distill_dataset_dir}"
        )
        gres = f"gpu:{run.gpu_type}:{run.gpus}"
    elif run.run_type == "build_synthetic_distill_dataset":
        # One-shot data-generation job (kd/build_synthetic_distill_dataset.py) -- a SEPARATE
        # pipeline from "build_distill_dataset" above, not a mode of it (see that script's own
        # module docstring). Real output is run.synthetic_dataset_dir (the shards), not run_dir
        # (Slurm logs + this generated params.yaml only, for provenance).
        data_dir = resolve_dataset_dir(run.suites, run.data_root, run.output_root)
        t5_text_embeddings_path = pathlib.Path(run.data_root) / "t5_embeddings.pkl"
        params_path = _write_generated_params_yaml(
            _build_synthetic_distill_dataset_params(run), run_dir / "params.yaml"
        )
        task_names_flag = f"--task_names {' '.join(run.task_names)} " if run.task_names else ""
        cmd = (
            "python -m cosmos_policy.scripts.cosmos_distill_experiments.kd.build_synthetic_distill_dataset "
            f"--build_params {params_path} "
            f"--data_dir {data_dir} "
            f"--t5_text_embeddings_path {t5_text_embeddings_path} "
            f'--rollout_data_dir "{run.rollout_data_dir}" '
            f"{task_names_flag}"
            f"--out_dir {run.synthetic_dataset_dir}"
        )
        gres = f"gpu:{run.gpu_type}:{run.gpus}"
    elif run.run_type == "build_teacher_native_distill_dataset":
        # One-shot data-generation job (kd/build_teacher_native_distill_dataset.py) -- a SEPARATE
        # pipeline from both "build_distill_dataset" and "build_synthetic_distill_dataset" above,
        # not a mode of either (see that script's own module docstring). Real output is
        # run.teacher_native_dataset_dir (the shards), not run_dir (Slurm logs + this generated
        # params.yaml only, for provenance).
        data_dir = resolve_dataset_dir(run.suites, run.data_root, run.output_root)
        t5_text_embeddings_path = pathlib.Path(run.data_root) / "t5_embeddings.pkl"
        params_path = _write_generated_params_yaml(
            _build_teacher_native_distill_dataset_params(run), run_dir / "params.yaml"
        )
        task_names_flag = f"--task_names {' '.join(run.task_names)} " if run.task_names else ""
        cmd = (
            "python -m cosmos_policy.scripts.cosmos_distill_experiments.kd.build_teacher_native_distill_dataset "
            f"--build_params {params_path} "
            f"--data_dir {data_dir} "
            f"--t5_text_embeddings_path {t5_text_embeddings_path} "
            f'--rollout_data_dir "{run.rollout_data_dir}" '
            f"{task_names_flag}"
            f"--out_dir {run.teacher_native_dataset_dir}"
        )
        gres = f"gpu:{run.gpu_type}:{run.gpus}"
    elif run.run_type == "kd_static_eval":
        # Continuous full-suite eval companion to a kd_static run (kd/periodic_libero_eval_static.py)
        # -- a SEPARATE Slurm job, not a training run itself and never feeds back into training:
        # `run_dir` here is THIS job's own output tree (eval_results.csv, run_libero_eval.py's own
        # per-call logs, Slurm logs), while `monitored_run_dir` is the kd_static run being watched
        # (its checkpoints/) -- see that script's module docstring for why coordination is
        # file-based rather than a subprocess/PID link, and for how it restarts safely from
        # eval_results.csv if this job itself gets halted and resubmitted.
        monitored_run_dir = job_output_dir(run.output_root, run.monitored_job_name)
        task_keyword_flag = f"--task_keyword {run.eval_task_keyword} " if run.eval_task_keyword else ""
        eval_every_n_iters_flag = f"--eval_every_n_iters {run.eval_every_n_iters} " if run.eval_every_n_iters else ""
        cmd = (
            "python -m cosmos_policy.scripts.cosmos_distill_experiments.kd.periodic_libero_eval_static "
            f"--run_dir {monitored_run_dir} "
            f"--inference_experiment {run.eval_inference_experiment} "
            f"--task_suite_name {run.eval_task_suite_name} "
            f"--num_trials_per_task {run.eval_num_trials_per_task} "
            f"--dataset_stats_path {run.eval_dataset_stats_path} "
            f"--t5_text_embeddings_path {run.eval_t5_text_embeddings_path} "
            f"--local_log_dir {run_dir} "
            f"--poll_seconds {run.eval_poll_seconds} "
            f"{task_keyword_flag}"
            f"{eval_every_n_iters_flag}"
            f"--checkpoint_format {run.eval_checkpoint_format}"
        )
        gres = f"gpu:{run.gpu_type}:{run.gpus}"
    else:
        data_dir = resolve_dataset_dir(run.suites, run.data_root, run.output_root)
        loss_csv_path = run_dir / "train_loss.csv"

        # Optionally stage the dataset to node-local scratch first (see RunConfig.stage_data_to_tmpdir).
        # cp -rL dereferences the symlink-combo dir resolve_dataset_dir builds for multi-suite runs.
        # Falls back to the /project path if $SLURM_TMPDIR is unset or the copy fails.
        if run.stage_data_to_tmpdir:
            data_stage = (
                f'_src_data_dir="{data_dir}"\n'
                f'export COSMOS_TRAIN_DATA_DIR="$_src_data_dir"\n'
                f'if [ -n "${{SLURM_TMPDIR:-}}" ] && cp -rL "$_src_data_dir" "$SLURM_TMPDIR/train_data" 2>/dev/null; then\n'
                f'    export COSMOS_TRAIN_DATA_DIR="$SLURM_TMPDIR/train_data"\n'
                f'    echo "Staged dataset to $COSMOS_TRAIN_DATA_DIR"\n'
                f'else\n'
                f'    echo "WARNING: dataset staging to \\$SLURM_TMPDIR skipped/failed -- reading from $_src_data_dir" >&2\n'
                f'fi\n\n'
            )
            data_dir_override = "$COSMOS_TRAIN_DATA_DIR"
        else:
            data_stage = ""
            data_dir_override = str(data_dir)

        persistent_workers = "true" if run.num_workers > 0 else "false"
        overrides = {
            "job.wandb_mode": "disabled",
            "job.name": run.job_name,
            "trainer.max_iter": str(run.max_iter),
            "trainer.logging_iter": "10",
            "dataloader_train.batch_size": str(run.batch_size),
            "dataloader_train.num_workers": str(run.num_workers),
            "dataloader_train.persistent_workers": persistent_workers,
            "checkpoint.save_iter": str(run.checkpoint_save_iter),
            "dataloader_train.dataset.data_dir": data_dir_override,
            "dataloader_train.dataset.rollout_data_dir": f'"{run.rollout_data_dir}"',
            "trainer.callbacks.loss_csv.csv_path": loss_csv_path,
            **dict(run.extra_overrides),
        }
        if run.task_names:
            overrides["dataloader_train.dataset.task_names"] = "[" + ",".join(run.task_names) + "]"
        override_str = " ".join(f"{k}={v}" for k, v in overrides.items())
        cmd = (
            f"torchrun --nproc_per_node={run.gpus} --master_port={run.master_port} "
            "-m cosmos_policy.scripts.cosmos_distill_experiments.run_train "
            "--config=cosmos_policy/config/config.py -- "
            f'experiment="{run.experiment}" {override_str}'
        )
        gres = f"gpu:{run.gpu_type}:{run.gpus}"

    teacher_prefetch = TEACHER_CHECKPOINT_PREFETCH if run.run_type in TEACHER_LOADING_RUN_TYPES else ""

    return f"""#!/bin/bash
#SBATCH --account={run.account}
#SBATCH --gres={gres}
#SBATCH --cpus-per-task={run.cpus}
#SBATCH --mem={run.mem}
#SBATCH --time={run.time_limit}
#SBATCH --job-name={run.job_name}
#SBATCH --output={log_dir}/{run.job_name}_%j.out
#SBATCH --error={log_dir}/{run.job_name}_%j.err

set -euo pipefail
cd {REPO_ROOT}
source .venv/bin/activate
{teacher_prefetch}source activate_cuda_env.sh
export WANDB_MODE=disabled
export IMAGINAIRE_OUTPUT_ROOT={run.output_root}

{data_stage}{cmd}
"""


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: SweepConfig) -> None:
    job_names = [run.job_name for run in cfg.runs.values()]
    if len(job_names) != len(set(job_names)):
        raise SystemExit(f"job_name values must be unique across runs: {job_names}")

    names = cfg.launch.only or list(cfg.runs.keys())
    unknown = set(names) - set(cfg.runs.keys())
    if unknown:
        raise SystemExit(f"launch.only names not found in runs: {sorted(unknown)} (have: {list(cfg.runs.keys())})")

    for name in names:
        run = cfg.runs[name]
        run_dir = job_output_dir(run.output_root, run.job_name)

        if run.run_type in (
            "build_distill_dataset",
            "build_synthetic_distill_dataset",
            "build_teacher_native_distill_dataset",
        ):
            # Not a training run: no checkpoint, no resume. Real output is run.distill_dataset_dir /
            # run.synthetic_dataset_dir / run.teacher_native_dataset_dir (the shards) -- run_dir here
            # only ever holds Slurm logs + the generated params.yaml, so wipe/overwrite risk has to
            # be checked against that directory instead, not the torchrun/KD-training
            # checkpoint-marker logic below. All three write via the same distill_dataset.py
            # ShardWriter (build_synthetic_distill_dataset.py/build_teacher_native_distill_dataset.py
            # reuse it directly -- see their own module docstrings for why), same shard_*.pt naming
            # and no-resume-just-overwrites semantics, so this check applies identically to all three.
            if run.run_type == "build_synthetic_distill_dataset":
                dataset_dir = pathlib.Path(run.synthetic_dataset_dir)
            elif run.run_type == "build_teacher_native_distill_dataset":
                dataset_dir = pathlib.Path(run.teacher_native_dataset_dir)
            else:
                dataset_dir = pathlib.Path(run.distill_dataset_dir)
            existing_shards = sorted(dataset_dir.glob("shard_*.pt")) if dataset_dir.exists() else []
            if cfg.launch.wipe:
                if existing_shards:
                    print(
                        f"[WARNING] {name} ({run.job_name}): about to PERMANENTLY DELETE "
                        f"{len(existing_shards)} existing shard(s) under {dataset_dir}"
                    )
                    if not _confirm("Type 'yes' to permanently delete this, anything else to abort: "):
                        raise SystemExit(f"Aborted: {name} ({run.job_name}) wipe not confirmed.")
                    shutil.rmtree(dataset_dir)
                    print(f"[WIPED] {name} ({run.job_name}): removed {dataset_dir}")
                else:
                    print(f"[WIPED] {name} ({run.job_name}): {dataset_dir} had no shards, nothing to remove")
            elif existing_shards:
                # ShardWriter has no resume logic (see distill_dataset.py's module docstring) --
                # re-running without wiping doesn't continue, it silently starts overwriting
                # shard_00000.pt onward while any shards past what this run produces stay stale.
                print(
                    f"[WARNING] {name} ({run.job_name}): {dataset_dir} already has "
                    f"{len(existing_shards)} shard(s) -- this run will start OVERWRITING from "
                    f"shard_00000.pt, not append or resume. Pass launch.wipe=true "
                    f"launch.only=[{name}] first if you meant a clean rebuild."
                )
                if not _confirm(f"Type 'yes' to overwrite {name} ({run.job_name})'s existing shards, anything else to abort: "):
                    raise SystemExit(f"Aborted: {name} ({run.job_name}) overwrite not confirmed.")
        elif cfg.launch.wipe:
            if run_dir.exists():
                checkpoint_marker = _existing_checkpoint_marker(run, run_dir)
                if checkpoint_marker.exists():
                    # The torchrun path's marker is a small text file worth reading for the exact
                    # iteration; kd_static's marker is now the same kind of small text file (see
                    # _existing_checkpoint_marker), so only kd_live's multi-GB train_state.pt still
                    # needs this not-worth-deserializing-just-for-this-message treatment.
                    if run.run_type == "kd_live":
                        what = f"a checkpoint ({checkpoint_marker.name} exists)"
                    else:
                        what = f"a checkpoint (latest: {checkpoint_marker.read_text().strip()})"
                else:
                    what = "no checkpoint yet, but other files"
                contents = ", ".join(sorted(p.name for p in run_dir.iterdir())) or "(empty)"
                print(
                    f"[WARNING] {name} ({run.job_name}): about to PERMANENTLY DELETE {run_dir} "
                    f"({what}) -- contents: {contents}"
                )
                if not _confirm("Type 'yes' to permanently delete this, anything else to abort: "):
                    raise SystemExit(f"Aborted: {name} ({run.job_name}) wipe not confirmed.")
                shutil.rmtree(run_dir)
                print(f"[WIPED] {name} ({run.job_name}): removed {run_dir}")
            else:
                print(f"[WIPED] {name} ({run.job_name}): {run_dir} did not exist, nothing to remove")
        else:
            # job.name auto-resume means this run will pick up wherever an existing checkpoint at
            # this same run_dir left off, rather than starting fresh -- surface that BEFORE
            # submitting (not after), since it's easy to not notice a stale job_name reused from an
            # earlier edit of conf/runs/*.yaml. Only checked when NOT wiping -- wipe's own [WIPED]
            # message above already reports what existed.
            checkpoint_marker = _existing_checkpoint_marker(run, run_dir)
            if checkpoint_marker.exists():
                is_kd_live = run.run_type in ("kd_live", "kd_live_av", "kd_live_action")
                latest = checkpoint_marker.read_text().strip() if not is_kd_live else checkpoint_marker.name
                print(
                    f"[WARNING] {name} ({run.job_name}): {run_dir} already has a checkpoint "
                    f"(latest: {latest}) -- this run will AUTO-RESUME from there, not start from "
                    f"iteration 0. Pass launch.wipe=true launch.only=[{name}] first if you meant a "
                    f"clean run."
                )
                if not _confirm(f"Type 'yes' to resume {name} ({run.job_name}) from {latest}, anything else to abort: "):
                    raise SystemExit(f"Aborted: {name} ({run.job_name}) resume not confirmed.")
            elif run_dir.exists() and any(run_dir.iterdir()):
                print(
                    f"[WARNING] {name} ({run.job_name}): {run_dir} already exists and is "
                    f"non-empty (no checkpoint yet, but other files present, e.g. from a "
                    f"previously wiped or interrupted run) -- new output will be added alongside "
                    f"whatever's already there."
                )

        # Slurm needs --output/--error's directory to exist before the job starts writing to it --
        # create it (and thus run_dir, its parent) up front rather than relying on the training
        # job itself to have created run_dir first.
        (run_dir / "slurm").mkdir(parents=True, exist_ok=True)

        script = build_sbatch_script(run)
        script_path = SCRIPT_DIR / (run.filename or f"{run.job_name}.sbatch")
        script_path.write_text(script)
        script_path.chmod(0o755)

        if cfg.launch.submit:
            result = subprocess.run(["sbatch", str(script_path)], capture_output=True, text=True)
            if result.returncode != 0:
                print(f"[FAILED] {name} ({run.job_name}): {result.stderr.strip()}")
            else:
                print(f"[SUBMITTED] {name} ({run.job_name}): {result.stdout.strip()}")
        else:
            print(f"[DRY RUN] Would submit: sbatch {script_path}")

    if not cfg.launch.submit:
        print(f"\n{len(names)} sbatch script(s) written to {SCRIPT_DIR}/ -- review, then re-run with launch.submit=true.")


if __name__ == "__main__":
    main()
