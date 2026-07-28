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
Periodically evaluate a running BC training job's checkpoints in the LIBERO simulator.

Launched by `train_from_scratch_bc_demo.py` (via `--eval-every-n-steps`) as a *separate*
process alongside the `torchrun`-launched training subprocess, not as a training callback --
running the LIBERO/robosuite/MuJoCo simulator in the training process itself would mutate
global RNG state (`set_seed_everywhere`) and require careful train/eval mode toggling around
every rollout; staying out-of-process avoids both and means a crash in eval can never take
down training (or vice versa).

Steps, in a loop:
    1. Poll `--checkpoint-dir` for `iter_*` entries that haven't been evaluated yet -- these can
       be either a flat `iter_NNNNNNNNN.pt` file (the plain Checkpointer, checkpointer.py) or an
       `iter_NNNNNNNNN` DIRECTORY (the DCP/distributed checkpointer, checkpointer/dcp.py, used by
       e.g. the LIBERO experiment with FSDP) -- `find_unevaluated_checkpoints` matches both.
    2. For each one whose iteration is a positive multiple of `--eval-every-n-steps`, confirm it's
       actually finished writing by checking `checkpoint_dir/latest_checkpoint.txt` (written by the
       DCP checkpointer only once a checkpoint is fully saved, per its own module docstring: "Points
       to most recent checkpoint folder") names this iteration or a later one -- an `iter_*` entry
       that exists on disk but isn't (yet) named there may still be mid-write. Then shell out to
       `cosmos_policy.experiments.robot.libero.run_libero_eval`, restricted via `--task_id` to
       just the one task this training run is on (resolved from `--task-keyword` against the
       LIBERO benchmark's task language descriptions -- see `resolve_task_id`), for
       `--num-trials` rollouts.
    3. Parse the "Overall success rate: X (Y%)" summary line run_libero_eval.py prints and log
       it via plain `log.info` (no wandb) as this run's average return proxy for that iteration.
       Also append a row to `<local-log-dir>/eval_results.csv` (iteration, success_rate,
       num_trials, task_id, task_description, status, timestamp) -- a durable, structured record
       independent of whether the run's stdout was piped through `tee`. `status` is "ok" or
       "failed" (eval subprocess crashed, or its output couldn't be parsed for a success rate);
       failed rows still get a row, with success_rate left blank, so the CSV is a complete record
       of every iteration eval was attempted at.
    4. Stop once the checkpoint at (or past) `--max-iter` has been evaluated, or once the
       training process (`--training-pid`) has exited and there are no more unevaluated
       checkpoints to catch up on (covers a crashed/interrupted training run).

Usage (normally invoked by train_from_scratch_bc_demo.py, not run directly):
    python -m cosmos_policy.scripts.periodic_libero_eval \
        --checkpoint-dir <IMAGINAIRE_OUTPUT_ROOT>/<JOB_PROJECT>/from_scratch_bc_demo/<job_name>/checkpoints \
        --config-file cosmos_policy/config/config.py \
        --inference-experiment cosmos_predict2_2b_480p_libero__inference_only \
        --task-suite-name libero_object --task-keyword alphabet_soup \
        --eval-every-n-steps 100 --max-iter 1000 --num-trials 10 \
        --dataset-stats-path nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_dataset_statistics.json \
        --t5-text-embeddings-path /path/to/success_only/t5_embeddings.pkl \
        --local-log-dir /path/to/work_dir/eval_logs

Caveats (untested end-to-end against a real training run -- there was no GPU/simulator
available to verify this against):
    - `--dataset-stats-path` defaults to the pretrained checkpoint's stats file, on the
      assumption that LIBERODataset's action/proprio normalization is a fixed, dataset-wide
      scheme independent of which task subset is in `data_dir` -- not independently confirmed.
    - Eval always targets `cuda:0` (hardcoded in run_libero_eval.py's DEVICE), so unless
      `--eval-gpu` (train_from_scratch_bc_demo.py) points it at a spare GPU, it contends for
      GPU memory/compute with the training process during each eval window.
"""

import argparse
import csv
import datetime
import math
import os
import pathlib
import re
import subprocess
import sys
import time

from cosmos_policy._src.imaginaire.utils import log

CHECKPOINT_NAME_RE = re.compile(r"^iter_(\d+)(?:\.pt)?$")
SUCCESS_RATE_RE = re.compile(r"Overall success rate: ([\d.]+) \([\d.]+%\)")
CSV_FIELDNAMES = ["iteration", "success_rate", "num_trials", "task_id", "task_description", "status", "timestamp"]


def append_csv_row(csv_path: pathlib.Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def resolve_task_id(task_suite_name: str, task_keyword: str) -> tuple[int, str]:
    """Find the single task in `task_suite_name` whose language description matches `task_keyword`.

    Mirrors `find_task_hdf5_file`'s substring-match semantics in train_from_scratch_bc_demo.py,
    but matched against the LIBERO benchmark's task descriptions (`task.language`, e.g. "pick up
    the alphabet soup and place it in the basket") rather than demo HDF5 filenames (e.g.
    "...alphabet_soup_demo.hdf5") -- filenames use underscores where descriptions use spaces, so
    both sides are normalized (underscores -> spaces) before matching.
    """
    from libero.libero import benchmark

    task_suite = benchmark.get_benchmark_dict()[task_suite_name]()
    descriptions = [task_suite.get_task(i).language for i in range(task_suite.n_tasks)]
    normalized_keyword = task_keyword.lower().replace("_", " ")
    matches = [
        i for i, description in enumerate(descriptions) if normalized_keyword in description.lower().replace("_", " ")
    ]

    if len(matches) == 1:
        return matches[0], descriptions[matches[0]]

    listing = "\n".join(f"  [{i}] {description}" for i, description in enumerate(descriptions))
    if not matches:
        raise ValueError(
            f"No task in suite {task_suite_name!r} has a language description matching "
            f"task_keyword={task_keyword!r}. Available tasks:\n{listing}"
        )
    raise ValueError(f"task_keyword={task_keyword!r} matched multiple tasks in suite {task_suite_name!r}:\n{listing}")


def find_unevaluated_checkpoints(checkpoint_dir: pathlib.Path, evaluated: set[int]) -> list[tuple[int, pathlib.Path]]:
    if not checkpoint_dir.is_dir():
        return []
    found = []
    for path in checkpoint_dir.glob("iter_*"):
        match = CHECKPOINT_NAME_RE.match(path.name)
        if match is not None and int(match.group(1)) not in evaluated:
            found.append((int(match.group(1)), path))
    return sorted(found)


def read_latest_checkpoint_iteration(checkpoint_dir: pathlib.Path) -> int | None:
    """The most recent iteration the checkpointer confirms is FULLY written, or None if unknown.

    Both checkpointer variants (checkpointer.py, checkpointer/dcp.py) write
    "{checkpoint_dir}/latest_checkpoint.txt" naming the checkpoint folder/file (e.g.
    "iter_000000005") only once that checkpoint is completely saved -- an iter_* entry that
    exists on disk but isn't (yet) named here (or is named an earlier iteration) may still be
    mid-write, which matters most for the DCP directory format (a partially-written directory
    has no single "final size" to poll for stability the way a flat .pt file would).
    """
    latest_file = checkpoint_dir / "latest_checkpoint.txt"
    if not latest_file.is_file():
        return None
    match = CHECKPOINT_NAME_RE.match(latest_file.read_text().strip())
    return int(match.group(1)) if match else None


def process_is_alive(pid: int | None) -> bool:
    if pid is None:
        return True  # no PID given -- caller can't check, so never treat training as "done" this way
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # process exists, just owned by someone else (shouldn't happen for our own child)
    return True


def run_eval(
    *,
    config_file: str,
    inference_experiment: str,
    checkpoint_path: pathlib.Path,
    task_suite_name: str,
    task_id: int,
    num_trials: int,
    dataset_stats_path: str,
    t5_text_embeddings_path: str,
    net_overrides: str,
    seed: int,
    local_log_dir: pathlib.Path,
) -> float:
    command = [
        sys.executable,
        "-m",
        "cosmos_policy.experiments.robot.libero.run_libero_eval",
        f"--config={inference_experiment}",
        f"--ckpt_path={checkpoint_path}",
        f"--config_file={config_file}",
        f"--task_suite_name={task_suite_name}",
        f"--task_id={task_id}",
        f"--num_trials_per_task={num_trials}",
        f"--dataset_stats_path={dataset_stats_path}",
        f"--t5_text_embeddings_path={t5_text_embeddings_path}",
        f"--local_log_dir={local_log_dir}",
        f"--seed={seed}",
        "--randomize_seed=False",
        "--data_collection=False",
        "--use_wandb=False",
    ]
    if net_overrides:
        command.append(f"--extra_model_overrides={net_overrides}")
    log.info("Eval command:\n" + " \\\n    ".join(command))

    result = subprocess.run(command, capture_output=True, text=True, check=False)
    output = result.stdout + result.stderr
    if result.returncode != 0:
        log.warning(f"Eval subprocess failed (exit {result.returncode}) for {checkpoint_path}:\n{output[-4000:]}")
        return float("nan")

    match = SUCCESS_RATE_RE.search(output)
    if match is None:
        log.warning(f"Could not find a success-rate summary line in eval output for {checkpoint_path}")
        return float("nan")
    return float(match.group(1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--inference-experiment", required=True)
    parser.add_argument("--task-suite-name", required=True)
    parser.add_argument("--task-keyword", required=True)
    parser.add_argument("--eval-every-n-steps", type=int, required=True)
    parser.add_argument("--max-iter", type=int, required=True)
    parser.add_argument("--num-trials", type=int, default=10)
    parser.add_argument("--dataset-stats-path", required=True)
    parser.add_argument("--t5-text-embeddings-path", required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--local-log-dir", required=True)
    parser.add_argument(
        "--net-overrides",
        default="",
        help="Comma-separated model.config.net.key=value overrides matching whatever shape the "
        "training checkpoints were built with, e.g. 'model_channels=512,num_heads=8,num_blocks=6'.",
    )
    parser.add_argument("--training-pid", type=int, default=None, help="PID of the training process to watch for exit.")
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    args = parser.parse_args()

    net_overrides = ",".join(
        f"model.config.net.{item.strip()}" for item in args.net_overrides.split(",") if item.strip()
    )

    task_id, task_description = resolve_task_id(args.task_suite_name, args.task_keyword)
    log.info(f"Periodic eval resolved task_id={task_id} ({task_description!r}) in suite {args.task_suite_name!r}")

    checkpoint_dir = pathlib.Path(args.checkpoint_dir)
    local_log_dir = pathlib.Path(args.local_log_dir)
    csv_path = local_log_dir / "eval_results.csv"
    evaluated: set[int] = set()
    reached_max_iter = False

    while not reached_max_iter:
        pending = find_unevaluated_checkpoints(checkpoint_dir, evaluated)
        training_alive = process_is_alive(args.training_pid)
        latest_written_iteration = read_latest_checkpoint_iteration(checkpoint_dir)

        for iteration, checkpoint_path in pending:
            if iteration == 0 or iteration % args.eval_every_n_steps != 0:
                evaluated.add(iteration)
                continue
            if latest_written_iteration is None or iteration > latest_written_iteration:
                continue  # still being written (or latest_checkpoint.txt hasn't caught up); retry next poll

            success_rate = run_eval(
                config_file=args.config_file,
                inference_experiment=args.inference_experiment,
                checkpoint_path=checkpoint_path,
                task_suite_name=args.task_suite_name,
                task_id=task_id,
                num_trials=args.num_trials,
                dataset_stats_path=args.dataset_stats_path,
                t5_text_embeddings_path=args.t5_text_embeddings_path,
                net_overrides=net_overrides,
                seed=args.seed,
                local_log_dir=local_log_dir,
            )
            log.info(
                f"[periodic eval] iter {iteration}: average success rate over {args.num_trials} "
                f"rollouts of task {task_description!r} = {success_rate:.4f}"
            )
            append_csv_row(
                csv_path,
                {
                    "iteration": iteration,
                    "success_rate": "" if math.isnan(success_rate) else success_rate,
                    "num_trials": args.num_trials,
                    "task_id": task_id,
                    "task_description": task_description,
                    "status": "failed" if math.isnan(success_rate) else "ok",
                    "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                },
            )
            evaluated.add(iteration)
            if iteration >= args.max_iter:
                reached_max_iter = True

        if reached_max_iter:
            break
        if not training_alive and not find_unevaluated_checkpoints(checkpoint_dir, evaluated):
            log.info("Training process has exited and no unevaluated checkpoints remain; stopping periodic eval.")
            break
        time.sleep(args.poll_seconds)
