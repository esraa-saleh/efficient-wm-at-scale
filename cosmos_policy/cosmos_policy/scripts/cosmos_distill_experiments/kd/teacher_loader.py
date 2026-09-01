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
Shared model-loading helper for every KD script (init_student_from_teacher.py, train_kd.py,
offline_eval.py, batch_prep_equivalence_test.py) -- used for both the teacher (via `load_teacher`,
which supplies the teacher's own experiment name and defaults its checkpoint to the released HF
repo) and the student (via the lower-level `load_policy_model`, called directly with the student's
registered net_experiments.py name and its teacher-derived-init .pt path). Exists because the base
experiment
(`cosmos_predict2_2b_480p_libero`)'s OWN `checkpoint.load_path` -- inherited from
config/experiment/cosmos_policy_experiment_configs.py:148 --
(`get_checkpoint_path("hf://nvidia/Cosmos-Predict2-2B-Video2World/model-480p-16fps.pt")`) is the
*pretrain-only* checkpoint, not the released LIBERO-finetuned policy: `run_libero_eval.py`'s own
usage examples always pass `--ckpt_path nvidia/Cosmos-Policy-LIBERO-Predict2-2B` explicitly for
exactly this reason (never relying on the experiment's own default load_path), and
`cosmos_policy/experiments/robot/cosmos_utils.py`'s `get_model()` does the same. So every KD script
needs to resolve a HF repo id (like the one above) to a local path *before* calling
`load_model_from_checkpoint`, since that function does not do HF-repo-id resolution itself -- it
only accepts a path it can hand to `easy_io.load()`/DCP directly. `get_model()` is the precedent
this mirrors exactly (`is_hf_checkpoint_path()` -> `download_hf_checkpoint()` -> pass the resolved
local path as `s3_checkpoint_dir`).
"""

import contextlib
from typing import Optional

import torch

from cosmos_policy._src.predict2.utils.model_loader import load_model_from_checkpoint
from cosmos_policy.experiments.robot.cosmos_utils import download_hf_checkpoint, is_hf_checkpoint_path

TEACHER_EXPERIMENT_NAME = "cosmos_predict2_2b_480p_libero"
DEFAULT_TEACHER_CHECKPOINT = "nvidia/Cosmos-Policy-LIBERO-Predict2-2B"
CONFIG_FILE = "cosmos_policy/config/config.py"


def resolve_checkpoint_path(checkpoint_path: str) -> str:
    """Downloads `checkpoint_path` from HuggingFace and returns the local path if it looks like a
    HF repo id (e.g. "nvidia/Cosmos-Policy-LIBERO-Predict2-2B"); returns it unchanged otherwise
    (already-local paths, s3:// paths)."""
    if is_hf_checkpoint_path(checkpoint_path):
        return download_hf_checkpoint(checkpoint_path)
    return checkpoint_path


def load_policy_model(
    to_device: str,
    experiment_name: str,
    checkpoint: Optional[str] = None,
    skip_load_model: bool = False,
):
    """Generic loader used for both teacher and student: resolves `checkpoint` if it's an HF repo
    id (see module docstring), then delegates to `load_model_from_checkpoint`
    (cosmos_policy/_src/predict2/utils/model_loader.py). `checkpoint=None` with
    `skip_load_model=False` falls back to the named experiment's own `checkpoint.load_path` --
    only appropriate when that default is actually what you want (it usually isn't for the
    teacher; see `load_teacher` below).

    Pins the current CUDA device to `to_device` for the whole call: `load_model_from_checkpoint`
    correctly does `model.to(torch.device(to_device))` up front, but its own `model.on_train_start()`
    call right after (`text2world_model.py`'s `self.net = self.net.to(**self.tensor_kwargs)`,
    where `tensor_kwargs["device"]` is the bare string `"cuda"`, not an indexed device) silently
    re-homes `self.net` -- the actual transformer weights, not just some auxiliary buffer -- onto
    `torch.cuda.current_device()`, undoing that placement whenever `to_device` isn't the current
    device. Confirmed against the real released teacher/student checkpoints: loading a model with
    `to_device="cuda:1"` while cuda:0 was current left `net`'s weights on cuda:0, causing "mat2 is
    on cuda:0" RuntimeErrors on the very first forward pass through a two-GPU KD step. Pinning the
    current device here fixes this at the source for every caller instead of requiring each one to
    reason about it."""
    s3_checkpoint_dir = None if skip_load_model else (resolve_checkpoint_path(checkpoint) if checkpoint else None)
    device_ctx = torch.cuda.device(to_device) if to_device.startswith("cuda") else contextlib.nullcontext()
    with device_ctx:
        return load_model_from_checkpoint(
            experiment_name=experiment_name,
            s3_checkpoint_dir=s3_checkpoint_dir,
            config_file=CONFIG_FILE,
            to_device=to_device,
            instantiate_ema=False,
            skip_load_model=skip_load_model,
        )


def load_teacher(
    to_device: str,
    checkpoint: str = DEFAULT_TEACHER_CHECKPOINT,
    experiment_name: str = TEACHER_EXPERIMENT_NAME,
    skip_load_model: bool = False,
):
    """Loads the released Cosmos Policy LIBERO teacher (or, with `skip_load_model=True`, an
    uninitialized instance of the same architecture -- not used for the student in practice, since
    the student needs a different, registered net_experiments.py experiment name; kept here mainly
    for the teacher's own default checkpoint convenience)."""
    return load_policy_model(
        to_device=to_device,
        experiment_name=experiment_name,
        checkpoint=checkpoint,
        skip_load_model=skip_load_model,
    )
