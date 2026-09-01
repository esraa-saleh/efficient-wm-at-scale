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
Checkpoint I/O for the KD student.

`save_student`/`load_student_for_resume` (train_kd.py's live-KD path): writes two files per
checkpoint, overwritten in place every `checkpoint_every` iterations (only the latest ever matters
-- train_kd.py's own resume support, nothing else reads this run_dir's checkpoints).
- `model.pt`: a bare `student_model.state_dict()`, nothing else. This is the file
  `run_libero_eval.py`'s `--ckpt_path` should point at -- `load_model_state_dict_from_checkpoint`
  (cosmos_policy/_src/predict2/utils/model_loader.py:195-274) branches on a `.pt` suffix, loads it
  via `easy_io.load()`, and calls `model.load_state_dict(local_state_dict, strict=False)` directly
  on whatever that file contains -- so it must be the flat state dict, not wrapped in another dict
  level.
- `train_state.pt`: `{"model": ..., "optimizer": ..., "iteration": ...}`. Kept separate from
  model.pt (rather than one format serving both purposes) so the eval-facing file never has to
  special-case a wrapper key.

`save_versioned_checkpoint`/`load_latest_versioned_checkpoint_for_resume` (train_kd_static.py):
a DIFFERENT scheme for a different requirement -- every checkpoint is kept, not overwritten, so a
separate `kd_static_eval` job (periodic_libero_eval_static.py) can sim-eval each one independently
of training, on its own schedule, without training's own checkpointing racing to overwrite whatever
that job is mid-read on. Mirrors the torchrun/Trainer path's own `checkpoints/iter_NNNNNNNNN/` +
`latest_checkpoint.txt` convention (see ../../periodic_libero_eval.py, written for that path) rather
than inventing a new one -- same `model.pt`/`train_state.pt` pair as above, just one directory per
checkpoint instead of overwritten in place, so the same "is this one safe to read yet" logic
(`latest_checkpoint.txt` named LAST, after both files inside are complete) already proven for that
path applies here unchanged.
"""

import pathlib
from typing import Optional, Tuple

import torch


def save_student(student_model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, run_dir: pathlib.Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(student_model.state_dict(), run_dir / "model.pt")
    torch.save(
        {"model": student_model.state_dict(), "optimizer": optimizer.state_dict(), "iteration": iteration},
        run_dir / "train_state.pt",
    )


def save_versioned_checkpoint(
    student_model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, run_dir: pathlib.Path
) -> pathlib.Path:
    """Writes `run_dir/checkpoints/iter_{iteration:09d}/{model.pt,train_state.pt}`, then updates
    `run_dir/checkpoints/latest_checkpoint.txt` to name it -- written LAST, only once both files
    inside are complete, so a poller checking that marker (see
    periodic_libero_eval_static.read_latest_checkpoint_iteration) never sees a partially-written
    checkpoint. Nothing here ever gets deleted or overwritten -- every checkpoint this is called for
    persists, by design (see train_kd_static.py's module docstring)."""
    checkpoint_dir = run_dir / "checkpoints" / f"iter_{iteration:09d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(student_model.state_dict(), checkpoint_dir / "model.pt")
    torch.save(
        {"model": student_model.state_dict(), "optimizer": optimizer.state_dict(), "iteration": iteration},
        checkpoint_dir / "train_state.pt",
    )
    (run_dir / "checkpoints" / "latest_checkpoint.txt").write_text(checkpoint_dir.name)
    return checkpoint_dir


def load_latest_versioned_checkpoint_for_resume(run_dir: pathlib.Path) -> Optional[Tuple[dict, dict, int]]:
    """Reads `run_dir/checkpoints/latest_checkpoint.txt` (if present) and loads that checkpoint's
    train_state.pt for train_kd_static.py's own resume. Returns None if there's nothing to resume
    from yet (fresh run) -- distinct from `load_student_for_resume` below, which assumes the caller
    already checked existence itself."""
    latest_file = run_dir / "checkpoints" / "latest_checkpoint.txt"
    if not latest_file.is_file():
        return None
    train_state = torch.load(run_dir / "checkpoints" / latest_file.read_text().strip() / "train_state.pt", map_location="cpu")
    return train_state["model"], train_state["optimizer"], train_state["iteration"]


def load_student_for_resume(path: pathlib.Path) -> Tuple[dict, dict, int]:
    train_state = torch.load(path, map_location="cpu")
    return train_state["model"], train_state["optimizer"], train_state["iteration"]


def load_student_strict(model: torch.nn.Module, model_pt_path: pathlib.Path) -> None:
    """Deliberately `strict=True` -- stronger than the shared eval loader's `strict=False` (which
    exists to tolerate unrelated keys like TransformerEngine FP8 padding on OTHER checkpoints, not
    KD-script-produced ones) -- so any mismatch between what save_student wrote and what `model`
    actually expects is caught immediately, not silently swallowed."""
    state_dict = torch.load(model_pt_path, map_location=next(model.parameters()).device)
    model.load_state_dict(state_dict, strict=True)
