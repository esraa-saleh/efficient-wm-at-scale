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
Continuously sim-evals a kd_static training run's checkpoints as they appear -- by default a
full-suite pass (every task in `--task_suite_name`, `--num_trials_per_task` rollouts each) per
checkpoint, not a narrow subset: with `checkpoint_io.save_versioned_checkpoint` keeping every
checkpoint rather than overwriting (see train_kd_static.py's module docstring), there's no
in-training stopping decision this needs to feed cheaply -- every checkpoint gets a real,
full-coverage number, logged for you to review, not used to control training in any way.
`--task_keyword` opts into restricting to just one task instead (e.g. for a single-task training
run where full-suite coverage isn't meaningful) -- see its own help text.

Launched as a genuinely SEPARATE Slurm job (`run_type: kd_static_eval`, see submit_sweep.py), not a
subprocess of train_kd_static.py: the two may land on different compute nodes, so coordination is
entirely file-based (shared storage), not PID-based. Reuses the torchrun/Trainer path's own
`checkpoints/iter_NNNNNNNNN/` + `latest_checkpoint.txt` convention (train_kd_static.py now writes
this format too -- see checkpoint_io.py's module docstring) rather than inventing a new one, so
`find_unevaluated_checkpoints`/`read_latest_checkpoint_iteration` below are close ports of
../../periodic_libero_eval.py's own (also fixing that script's `--task_id` vs. the real
`--task_ids` flag mismatch -- see run_libero_eval.py's PolicyEvalConfig.task_ids).

RESTART SAFETY: if this job is halted (preemption, walltime, manual cancel) and resubmitted, it
must not silently re-evaluate checkpoints it already scored, nor silently skip ones it hasn't --
`main()` reads its own `eval_results.csv` at startup and seeds `evaluated` from every iteration
already logged there (regardless of status), before ever comparing it against
`find_unevaluated_checkpoints`. This is the one thing ../../periodic_libero_eval.py's own design
does NOT do (its `evaluated` set is purely in-memory, reset on every restart) -- deliberately fixed
here since this script is explicitly meant to survive being halted and relaunched, unlike that one.

Usage (normally launched via submit_sweep.py, not run directly -- see conf/runs/*_eval.yaml):
    python -m cosmos_policy.scripts.cosmos_distill_experiments.kd.periodic_libero_eval_static \\
        --run_dir <kd_static run's run_dir> \\
        --inference_experiment cosmos_kd_student_500m_libero__inference_only \\
        --task_suite_name libero_object --num_trials_per_task 3 \\
        --dataset_stats_path <path> --t5_text_embeddings_path <path> \\
        --local_log_dir <where eval_results.csv and run_libero_eval.py's own per-call logs go>
"""

import argparse
import csv
import datetime
import math
import os
import pathlib
import re
import sys
import time

from cosmos_policy.scripts.cosmos_distill_experiments.kd.eval_sim_crash_resume import (
    SUCCESS_RATE_RE,
    run_libero_eval_with_sim_crash_resume,
)
from cosmos_policy.scripts.periodic_libero_eval import resolve_task_id

CHECKPOINT_NAME_RE = re.compile(r"^iter_(\d+)$")
CSV_FIELDNAMES = ["iteration", "success_rate", "num_trials", "status", "timestamp"]


def append_csv_row(csv_path: pathlib.Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def load_already_evaluated(csv_path: pathlib.Path) -> set[int]:
    """Every iteration this script has ever logged a row for (any status) -- read at startup so a
    restart resumes from where a prior run of this same job left off, instead of re-evaluating
    everything from scratch or losing track of what's already done. See module docstring."""
    if not csv_path.exists():
        return set()
    with open(csv_path, newline="") as f:
        return {int(row["iteration"]) for row in csv.DictReader(f)}


def find_unevaluated_checkpoints(
    checkpoint_dir: pathlib.Path, evaluated: set[int], eval_every_n_iters: int | None = None
) -> list[tuple[int, pathlib.Path]]:
    if not checkpoint_dir.is_dir():
        return []
    found = []
    for path in checkpoint_dir.glob("iter_*"):
        match = CHECKPOINT_NAME_RE.match(path.name)
        if match is None:
            continue
        iteration = int(match.group(1))
        if iteration in evaluated:
            continue
        if eval_every_n_iters is not None and iteration % eval_every_n_iters != 0:
            continue
        found.append((iteration, path))
    return sorted(found)


def read_latest_checkpoint_iteration(checkpoint_dir: pathlib.Path) -> int | None:
    """The most recent iteration checkpoint_io.save_versioned_checkpoint confirms is FULLY written
    (latest_checkpoint.txt is written only once both model.pt and train_state.pt inside that
    iteration's folder are complete), or None if no checkpoint has been confirmed written yet. An
    `iter_*` directory that exists on disk but isn't (yet) named here may still be mid-write."""
    latest_file = checkpoint_dir / "latest_checkpoint.txt"
    if not latest_file.is_file():
        return None
    match = CHECKPOINT_NAME_RE.match(latest_file.read_text().strip())
    return int(match.group(1)) if match else None


def _write_failure_log(local_log_dir: pathlib.Path, iteration: int, reason: str, command: list, result) -> pathlib.Path:
    """Writes the FULL subprocess output (not just a stdout-bound tail) to its own file, opened and
    closed immediately (no long-lived buffered handle) so it's on disk regardless of whatever
    buffering this script's own parent stdout is subject to -- see this module's own investigation
    that motivated this: real failures were undiagnosable because `print(...)`'s output sat in an
    unflushed buffer for the SLURM `.out` file, sometimes for the job's ENTIRE runtime, while the
    per-iteration `eval_results.csv` row already recorded `status=failed`, giving no way to tell
    "the eval subprocess crashed" apart from "it finished but the regex didn't match" after the
    fact."""
    log_path = local_log_dir / f"eval_failure_iter_{iteration:09d}.log"
    with open(log_path, "w") as f:
        f.write(f"reason: {reason}\n")
        f.write(f"returncode: {result.returncode}\n")
        f.write("command:\n" + " \\\n    ".join(command) + "\n\n")
        f.write("=== stdout ===\n" + result.stdout + "\n")
        f.write("=== stderr ===\n" + result.stderr + "\n")
        f.flush()
        os.fsync(f.fileno())
    return log_path


def run_eval(
    *,
    config_file: str,
    inference_experiment: str,
    checkpoint_path: pathlib.Path,
    task_suite_name: str,
    num_trials_per_task: int,
    dataset_stats_path: str,
    t5_text_embeddings_path: str,
    seed: int,
    local_log_dir: pathlib.Path,
    iteration: int,
    max_sim_crash_restarts: int = 25,
    checkpoint_format: str = "kd_static",
    task_ids: str = "",
) -> float:
    progress_path = local_log_dir / f"eval_progress_iter_{iteration:09d}.json"
    # Fresh suite for this checkpoint -- wipe any leftover breadcrumbs from a prior attempt.
    if progress_path.is_file():
        progress_path.unlink()

    # kd_static: train_kd_static.py's own flat iter_NNNNNNNNN/model.pt (checkpoint_io.py). dcp: the
    # standard torchrun/DistributedCheckpointer sharded-directory format a plain non-KD run
    # produces -- load_model_from_checkpoint auto-detects DCP vs .pt by whether the path ends in
    # ".pt" (model_loader.py), so passing the bare iteration directory (not .../model.pt) is what
    # routes it to the DCP loader, which itself appends "model" internally (dcp.py:549).
    if checkpoint_format == "dcp":
        ckpt_path = checkpoint_path
    elif checkpoint_format == "kd_static":
        ckpt_path = checkpoint_path / "model.pt"
    else:
        raise ValueError(f"Unknown checkpoint_format: {checkpoint_format!r} (expected 'kd_static' or 'dcp')")

    command = [
        sys.executable,
        "-m",
        "cosmos_policy.experiments.robot.libero.run_libero_eval",
        f"--config={inference_experiment}",
        f"--ckpt_path={ckpt_path}",
        f"--config_file={config_file}",
        f"--task_suite_name={task_suite_name}",
        f"--num_trials_per_task={num_trials_per_task}",
        f"--dataset_stats_path={dataset_stats_path}",
        f"--t5_text_embeddings_path={t5_text_embeddings_path}",
        f"--local_log_dir={local_log_dir}",
        f"--seed={seed}",
        *([f"--task_ids={task_ids}"] if task_ids else []),
        "--randomize_seed=False",
        "--data_collection=False",
        "--use_wandb=False",
        f"--eval_progress_path={progress_path}",
    ]

    returncode, success_rate, output = run_libero_eval_with_sim_crash_resume(
        command,
        progress_path,
        max_restarts=max_sim_crash_restarts,
        capture_output=True,
    )

    # Synthesize a CompletedProcess-like object for the existing failure-log helper.
    class _Result:
        def __init__(self, returncode: int, output: str):
            self.returncode = returncode
            self.stdout = output
            self.stderr = ""

    result = _Result(returncode, output)

    if returncode != 0 or math.isnan(success_rate):
        reason = (
            "subprocess exited non-zero"
            if returncode != 0
            else "no success-rate summary line found"
        )
        log_path = _write_failure_log(local_log_dir, iteration, reason, command, result)
        print(
            f"[WARNING] Eval subprocess failed (exit {returncode}) for {checkpoint_path} "
            f"-- full output: {log_path}",
            flush=True,
        )
        return float("nan")

    # Confirm the summary line was present (success_rate already parsed by the helper).
    if SUCCESS_RATE_RE.search(output) is None:
        log_path = _write_failure_log(
            local_log_dir, iteration, "no success-rate summary line found", command, result
        )
        print(
            f"[WARNING] Could not find a success-rate summary line in eval output for "
            f"{checkpoint_path} -- full output: {log_path}",
            flush=True,
        )
        return float("nan")
    return success_rate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run_dir", required=True, help="The kd_static training run's run_dir (contains checkpoints/).")
    parser.add_argument("--config_file", default="cosmos_policy/config/config.py")
    parser.add_argument("--inference_experiment", required=True, help="e.g. cosmos_kd_student_500m_libero__inference_only")
    parser.add_argument("--task_suite_name", required=True)
    parser.add_argument("--num_trials_per_task", type=int, default=3, help="Every task in task_suite_name is always evaluated, unless --task_keyword restricts to one.")
    parser.add_argument(
        "--task_keyword",
        default=None,
        help="Restrict eval to the single task in task_suite_name whose language description "
        "contains this (case-insensitive, underscores/spaces both match, e.g. 'ketchup') -- see "
        "periodic_libero_eval.py's resolve_task_id. Default (unset) evaluates every task, "
        "unchanged from before.",
    )
    parser.add_argument("--dataset_stats_path", required=True)
    parser.add_argument("--t5_text_embeddings_path", required=True)
    parser.add_argument("--local_log_dir", required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--poll_seconds", type=float, default=30.0)
    parser.add_argument(
        "--eval_every_n_iters",
        type=int,
        default=None,
        help="Only evaluate checkpoints whose iteration is a multiple of this (e.g. 1000). "
        "Default (unset) evaluates every checkpoint that lands, per checkpoint_every at train time.",
    )
    parser.add_argument(
        "--checkpoint_format",
        default="kd_static",
        choices=["kd_static", "dcp"],
        help="'kd_static' (default): flat iter_NNNNNNNNN/model.pt, as train_kd_static.py writes. "
        "'dcp': the standard torchrun/DistributedCheckpointer sharded-directory format a plain "
        "non-KD run (e.g. baseline_1b_train) writes instead.",
    )
    args = parser.parse_args()

    task_ids = ""
    if args.task_keyword:
        resolved_task_id, resolved_description = resolve_task_id(args.task_suite_name, args.task_keyword)
        task_ids = str(resolved_task_id)
        print(f"--task_keyword={args.task_keyword!r} resolved to task_id={resolved_task_id} ({resolved_description!r}) in suite {args.task_suite_name!r}")

    run_dir = pathlib.Path(args.run_dir)
    checkpoint_dir = run_dir / "checkpoints"
    training_done_path = run_dir / "TRAINING_DONE"
    local_log_dir = pathlib.Path(args.local_log_dir)
    csv_path = local_log_dir / "eval_results.csv"

    evaluated = load_already_evaluated(csv_path)
    if evaluated:
        print(f"Resuming: {len(evaluated)} checkpoint(s) already evaluated per {csv_path} -- {sorted(evaluated)}")

    while True:
        pending = find_unevaluated_checkpoints(checkpoint_dir, evaluated, args.eval_every_n_iters)
        training_done = training_done_path.exists()
        latest_written_iteration = read_latest_checkpoint_iteration(checkpoint_dir)

        for iteration, checkpoint_path in pending:
            if latest_written_iteration is None or iteration > latest_written_iteration:
                continue  # still being written (or latest_checkpoint.txt hasn't caught up); retry next poll

            success_rate = run_eval(
                config_file=args.config_file,
                inference_experiment=args.inference_experiment,
                checkpoint_path=checkpoint_path,
                task_suite_name=args.task_suite_name,
                num_trials_per_task=args.num_trials_per_task,
                dataset_stats_path=args.dataset_stats_path,
                t5_text_embeddings_path=args.t5_text_embeddings_path,
                seed=args.seed,
                local_log_dir=local_log_dir,
                iteration=iteration,
                checkpoint_format=args.checkpoint_format,
                task_ids=task_ids,
            )
            scope = f"task_keyword={args.task_keyword!r}" if args.task_keyword else f"full-suite ({args.task_suite_name})"
            print(
                f"[periodic eval] iter {iteration}: {scope} success rate "
                f"over {args.num_trials_per_task} trials{'' if args.task_keyword else '/task'} = {success_rate:.4f}",
                flush=True,
            )
            append_csv_row(
                csv_path,
                {
                    "iteration": iteration,
                    "success_rate": "" if math.isnan(success_rate) else success_rate,
                    "num_trials": args.num_trials_per_task,
                    "status": "failed" if math.isnan(success_rate) else "ok",
                    "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                },
            )
            evaluated.add(iteration)

        if training_done and not find_unevaluated_checkpoints(checkpoint_dir, evaluated, args.eval_every_n_iters):
            print("TRAINING_DONE present and no unevaluated checkpoints remain; stopping periodic eval.")
            break

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
