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
Outer-wrapper resume for LIBERO evals that die mid-episode with MuJoCo SIGABRT.

SIGABRT cannot be caught inside run_libero_eval.py (it kills the process). The eval script
persists per-episode progress via --eval_progress_path; this module relaunches the same command
after an abort, marking the in-flight episode as a failure so the suite can finish.

Used by periodic_libero_eval_static.py and kd_static_final_ckpt_eval.sbatch.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess
from typing import Optional

SUCCESS_RATE_RE = re.compile(r"Overall success rate: ([\d.]+) \([\d.]+%\)")

# Python subprocess typically reports -6 for SIGABRT; some environments surface 128+6=134.
SIM_CRASH_EXIT_CODES = {-6, 134, 6}


def is_sim_crash_exit(returncode: int) -> bool:
    return returncode in SIM_CRASH_EXIT_CODES or abs(returncode) == 6


def load_progress(progress_path: pathlib.Path) -> dict:
    if not progress_path.is_file():
        return {"completed": [], "in_progress": None}
    with open(progress_path) as f:
        data = json.load(f)
    data.setdefault("completed", [])
    data.setdefault("in_progress", None)
    return data


def save_progress(progress_path: pathlib.Path, data: dict) -> None:
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = progress_path.with_suffix(progress_path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    tmp_path.replace(progress_path)


def mark_in_progress_episode_as_failed(progress_path: pathlib.Path) -> Optional[tuple[int, int]]:
    """If progress has an in_progress episode, record it as success=False / sim_crash.

    Returns (task_id, episode_idx) when a mark was written, else None.
    """
    progress = load_progress(progress_path)
    in_progress = progress.get("in_progress")
    if not in_progress:
        return None
    task_id = int(in_progress["task_id"])
    episode_idx = int(in_progress["episode_idx"])
    progress["completed"] = [
        c
        for c in progress.get("completed", [])
        if not (int(c["task_id"]) == task_id and int(c["episode_idx"]) == episode_idx)
    ]
    progress["completed"].append(
        {
            "task_id": task_id,
            "episode_idx": episode_idx,
            "success": False,
            "reason": "sim_crash",
        }
    )
    progress["in_progress"] = None
    save_progress(progress_path, progress)
    return task_id, episode_idx


def run_libero_eval_with_sim_crash_resume(
    command: list[str],
    progress_path: pathlib.Path,
    *,
    max_restarts: int = 25,
    capture_output: bool = True,
) -> tuple[int, float, str]:
    """Run run_libero_eval, relaunching after MuJoCo SIGABRT using the progress file.

    Returns (final_returncode, success_rate_or_nan, combined_output).
    Does NOT wipe progress_path -- caller should delete it before a fresh suite if desired.
    Ensures --eval_progress_path=<progress_path> is present on the command.

    max_restarts caps how many SIGABRT recoveries we attempt. Each recovery marks one
    in-flight episode failed and continues; a 30-episode suite can hit several landmines,
    so the default is high. The cap mainly guards load-time aborts that leave no
    in_progress breadcrumb (those refuse to retry) and true infinite loops.
    """
    progress_flag = f"--eval_progress_path={progress_path}"
    cmd = list(command)
    if not any(arg.startswith("--eval_progress_path=") for arg in cmd):
        cmd.append(progress_flag)

    combined_output_parts: list[str] = []
    last_returncode = 1
    restarts = 0

    while True:
        print(
            f"[sim-crash-resume] attempt={restarts + 1}/{max_restarts + 1} "
            f"progress={progress_path}",
            flush=True,
        )
        print("Eval command:\n" + " \\\n    ".join(cmd), flush=True)

        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        output = (result.stdout or "") + (result.stderr or "")
        if not capture_output:
            print(output, end="", flush=True)

        combined_output_parts.append(output)
        last_returncode = result.returncode

        if last_returncode == 0:
            match = SUCCESS_RATE_RE.search(output)
            if match is None:
                return last_returncode, float("nan"), "\n".join(combined_output_parts)
            return last_returncode, float(match.group(1)), "\n".join(combined_output_parts)

        if is_sim_crash_exit(last_returncode):
            marked = mark_in_progress_episode_as_failed(progress_path)
            if marked is None:
                print(
                    f"[sim-crash-resume] exit {last_returncode} looks like sim crash but no "
                    f"in_progress episode in {progress_path} -- not retrying",
                    flush=True,
                )
                return last_returncode, float("nan"), "\n".join(combined_output_parts)
            task_id, episode_idx = marked
            if restarts < max_restarts:
                restarts += 1
                print(
                    f"[sim-crash-resume] SIGABRT-like exit {last_returncode}: marked "
                    f"task={task_id} episode_idx={episode_idx} as failed (sim_crash); "
                    f"relaunching ({restarts}/{max_restarts})",
                    flush=True,
                )
                continue
            print(
                f"[sim-crash-resume] SIGABRT-like exit {last_returncode}: marked "
                f"task={task_id} episode_idx={episode_idx} as failed (sim_crash); "
                f"giving up (restarts={restarts}, max_restarts={max_restarts})",
                flush=True,
            )
            return last_returncode, float("nan"), "\n".join(combined_output_parts)

        print(
            f"[sim-crash-resume] giving up after exit {last_returncode} "
            f"(restarts={restarts}, max_restarts={max_restarts})",
            flush=True,
        )
        return last_returncode, float("nan"), "\n".join(combined_output_parts)


def parse_success_rate(output: str) -> float:
    match = SUCCESS_RATE_RE.search(output)
    if match is None:
        return float("nan")
    return float(match.group(1))
