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
Shared helpers for the kd/*_test.py files. Named `support.py`, not `test_support.py` --
pytest's default `python_files` pattern matches both `test_*.py` AND `*_test.py`, so a leading
`test_` (even without a matching test function inside) still gets collected and, in this repo,
fails at import time (see the "Test first (KD)" section of ../README.md for why).
"""

import pathlib
from typing import Tuple

import torch
from omegaconf import OmegaConf

from cosmos_policy.scripts.cosmos_distill_experiments.kd.teacher_loader import load_policy_model

# Smallest registered student size -- used across tests purely for instantiation speed, never for
# real KD quality (these tests check plumbing, not fidelity).
SMALL_STUDENT_EXPERIMENT = "cosmos_kd_student_500m_libero"

_TRAIN_YAML = pathlib.Path(__file__).resolve().parents[1] / "conf" / "runs" / "train.yaml"


def resolve_test_data_paths(suite: str = "libero_object_regen") -> Tuple[str, str]:
    """Returns (data_dir, t5_text_embeddings_path) for `suite`, resolved against
    ../conf/runs/train.yaml's data_root -- the same cluster storage location every other run in
    this folder already reads from (data_root/output_root are plain RunConfig fields now, no
    separate paths.yaml). Tests using this need to run somewhere that path is mounted (the cluster,
    not this repo's own scratch space)."""
    train_run = OmegaConf.load(_TRAIN_YAML)
    data_dir = str(pathlib.Path(train_run.data_root) / suite)
    t5_text_embeddings_path = str(pathlib.Path(train_run.data_root) / "t5_embeddings.pkl")
    return data_dir, t5_text_embeddings_path


def write_fake_student_init(out_path: pathlib.Path, experiment_name: str = SMALL_STUDENT_EXPERIMENT) -> pathlib.Path:
    """Writes a fresh (randomly initialized, not teacher-derived) student's own state dict to
    `out_path` -- a real, loadable model.pt for tests that need *some* valid student_init_path but
    don't care about its actual weights (teacher_frozen_test, identity_kd_test's non-identity
    plumbing checks, etc). Real teacher-derived init is student_init_test.py's job, not this."""
    student, _ = load_policy_model(to_device="cpu", experiment_name=experiment_name, skip_load_model=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(student.state_dict(), out_path)
    return out_path
