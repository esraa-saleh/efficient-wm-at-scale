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
Same generator pattern as ../cosmos_dit_wm/submit_sweep.py, scoped to just this job's variants
(the real run + its smoketest, or any other run you add under conf/runs/) -- deliberately
self-contained rather than adding entries to that shared script, so this folder never has to touch
(or wait its turn behind) submissions for the rest of the cosmos_dit_wm family.

Every *output* a run produces -- checkpoints, loss CSV, wandb offline run, DeviceMonitor, the
resolved-config snapshot, and now Slurm's own stdout/stderr too -- lands under one project-storage
folder per run: job_output_dir(paths, run.job_name), i.e.
{paths.output_root}/cosmos_policy/cosmos_v2_finetune/{run.job_name}/. Slurm logs used to go to
./logs/ next to this script (on $HOME, small quota, and separate from everything else the run
produced); they're now under {job_output_dir}/slurm/ instead, alongside the rest of that run's
output. One consequence of this: launch.wipe=true now also deletes that run's Slurm log history,
not just its checkpoints/wandb/etc -- that's intentional (one wipe cleans the whole run), but worth
knowing before you wipe a run whose stdout/stderr you still wanted.

train.sbatch / smoketest.sbatch themselves are deliberately NOT moved into that tree -- they're
generated *launcher code* (reviewable in this folder, in your editor, alongside sweep.py/conf/),
not run output, so they stay here regardless of which account's storage paths.yaml points at.

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

from cosmos_policy.scripts.cosmos_distill_experiments.sweep import PathsConfig, RunConfig, SweepConfig

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
# .../cosmos_policy/cosmos_policy/scripts/cosmos_distill_experiments -> .../cosmos_policy (the repo
# clone root containing .venv/, activate_cuda_env.sh). Derived, not a conf/ value: it's a
# structural fact about where this repo was cloned, not a per-account setting -- see
# conf/paths.yaml for the values that ARE per-account (data_root, output_root).
REPO_ROOT = SCRIPT_DIR.parent.parent.parent


def resolve_data_dir(data_dir: str, data_root: str) -> str:
    return data_dir if data_dir.startswith("/") else f"{data_root}/{data_dir}"


def _confirm(prompt: str) -> bool:
    """Blocking yes/no prompt. Fails safe (returns False, i.e. "don't proceed") if there's no TTY
    to prompt on, rather than hanging or silently proceeding."""
    try:
        return input(prompt).strip().lower() == "yes"
    except EOFError:
        return False


def job_output_dir(paths: PathsConfig, job_name: str) -> pathlib.Path:
    """{paths.output_root}/cosmos_policy/cosmos_v2_finetune/{job_name} -- must match job.project /
    job.group / job.name exactly (see _src/imaginaire/config.py's JobConfig.path_local), since this
    is where the *training framework itself* actually writes checkpoints/wandb/etc, not an
    independently-chosen location. "cosmos_policy"/"cosmos_v2_finetune" come from job.project /
    job.group, set on the base experiment this job inherits from
    (config/experiment/cosmos_policy_experiment_configs.py) -- duplicated here, not derived, so if
    that base ever changes its job.group this needs updating too."""
    return pathlib.Path(paths.output_root) / "cosmos_policy" / "cosmos_v2_finetune" / job_name


def build_sbatch_script(run: RunConfig, paths: PathsConfig) -> str:
    data_dir = resolve_data_dir(run.data_dir, paths.data_root)
    run_dir = job_output_dir(paths, run.job_name)
    loss_csv_path = run_dir / "train_loss.csv"
    # Slurm's own stdout/stderr, alongside everything else this run produces (checkpoints, loss
    # CSV, wandb, DeviceMonitor, config.yaml) -- see module docstring for why this moved off
    # $HOME/./logs/, and the launch.wipe consequence of that.
    log_dir = run_dir / "slurm"

    overrides = {
        "job.wandb_mode": "offline",
        "job.name": run.job_name,
        "trainer.max_iter": str(run.max_iter),
        "trainer.logging_iter": "10",
        "dataloader_train.batch_size": str(run.batch_size),
        "dataloader_train.num_workers": "0",
        "dataloader_train.persistent_workers": "false",
        "checkpoint.save_iter": str(run.checkpoint_save_iter),
        "dataloader_train.dataset.data_dir": data_dir,
        "dataloader_train.dataset.rollout_data_dir": f'"{run.rollout_data_dir}"',
        "trainer.callbacks.loss_csv.csv_path": loss_csv_path,
        **dict(run.extra_overrides),
    }
    override_str = " ".join(f"{k}={v}" for k, v in overrides.items())

    torchrun_cmd = (
        "torchrun --nproc_per_node=1 --master_port=12341 "
        "-m cosmos_policy.scripts.cosmos_dit_wm.run_train "
        "--config=cosmos_policy/config/config.py -- "
        f'experiment="{run.experiment}" {override_str}'
    )

    return f"""#!/bin/bash
#SBATCH --account={run.account}
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task={run.cpus}
#SBATCH --mem={run.mem}
#SBATCH --time={run.time_limit}
#SBATCH --job-name={run.job_name}
#SBATCH --output={log_dir}/{run.job_name}_%j.out
#SBATCH --error={log_dir}/{run.job_name}_%j.err

set -euo pipefail
cd {REPO_ROOT}
source .venv/bin/activate
source activate_cuda_env.sh
export WANDB_MODE=offline
export IMAGINAIRE_OUTPUT_ROOT={paths.output_root}

{torchrun_cmd}
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
        run_dir = job_output_dir(cfg.paths, run.job_name)

        if cfg.launch.wipe:
            if run_dir.exists():
                latest_checkpoint_file = run_dir / "checkpoints" / "latest_checkpoint.txt"
                if latest_checkpoint_file.exists():
                    what = f"a checkpoint (latest: {latest_checkpoint_file.read_text().strip()})"
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
            latest_checkpoint_file = run_dir / "checkpoints" / "latest_checkpoint.txt"
            if latest_checkpoint_file.exists():
                latest = latest_checkpoint_file.read_text().strip()
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

        script = build_sbatch_script(run, cfg.paths)
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
