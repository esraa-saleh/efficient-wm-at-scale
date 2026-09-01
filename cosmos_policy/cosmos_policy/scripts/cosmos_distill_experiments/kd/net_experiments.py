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
Registers one Hydra *experiment* per KD student size -- the architecture recipe shared by three
separate consumers that all need to resolve the identical `num_blocks` for a given size by name
rather than duplicating the literal value: init_student_from_teacher.py (build the empty student
skeleton), the optional same-size non-KD baseline run (a normal torchrun/Trainer job, which can only
take an `experiment=<name>` string on its CLI), and run_libero_eval.py's `--config` for the final
policy eval.

Follows exactly the same plain-dict-overlay `LazyDict` pattern ../experiment.py already uses (see
that file's own module docstring and ../params.py's docstring for why this mechanism -- not a
Hydra `defaults`-composed structured config -- is the one proven to work end-to-end in this
codebase's LazyConfig system, job 18781181). Registered with Hydra via the same import-shim
mechanism experiment.py uses: config/experiment/cosmos_distill_experiments_registration.py imports
this module for its side effect (populating the ConfigStore) -- Hydra's own
`import_all_modules_from_package("cosmos_policy.config.experiment", ...)` only scans modules
physically inside that package, not this one.

Per the KD plan's finding #2, the teacher net is fixed-width (COSMOS_V2_2B_NET:
model_channels=2048, num_heads=16, num_blocks=28 -- see
cosmos_policy/_src/predict2/configs/text2world/defaults/net.py:80-96) and the "depth-reduced
student" design keeps width/heads unchanged and only reduces `num_blocks`. So every entry below
overrides `model.config.net.num_blocks` alone -- nothing else about the architecture changes.

`num_blocks` values below are confirmed against a real run of
`python -m cosmos_policy.scripts.cosmos_distill_experiments.kd.student_sizes` on an H100 node
(teacher: 28 blocks, 1.957B params): 14 blocks -> 0.988B ("1b"), 10 blocks -> 0.711B ("700m"), 7
blocks -> 0.504B ("500m") -- all close to their targets, no adjustment needed. Only "1b" is needed
to start (see the KD plan's recommended sequencing); 700m/500m are registered now for convenience
but shouldn't be launched until the 1b result is promising.
"""

import os

from hydra.core.config_store import ConfigStore

from cosmos_policy._src.imaginaire.lazy_config import LazyCall as L
from cosmos_policy._src.imaginaire.lazy_config import LazyDict
from cosmos_policy.scripts.cosmos_distill_experiments.loss_csv_callback import LossCsvCallback

TEACHER_EXPERIMENT_NAME = "cosmos_predict2_2b_480p_libero"

# Same env var ../experiment.py reads for the same reason (_src/imaginaire/config.py's
# Config.path_local checkpoint default) -- only used as this callback's OWN default csv_path below;
# ../submit_sweep.py's (non-KD) sbatch path always overrides trainer.callbacks.loss_csv.csv_path to
# the actual job_name-based run dir at launch time, same as it does for the existing tiny-net job.
_OUTPUT_ROOT = os.environ.get("IMAGINAIRE_OUTPUT_ROOT", "/tmp/imaginaire4-output")

# name -> num_blocks. Public (imported by init_student_from_teacher.py to enumerate valid
# --student_size choices) -- not just this module's own registration loop below. Confirmed against
# a real kd/student_sizes.py run -- see module docstring.
STUDENT_SIZES = {
    "1b": 14,  # 0.988B
    "700m": 10,  # 0.711B
    "500m": 7,  # 0.504B
}


def _make_student_experiment(size_name: str, num_blocks: int) -> tuple[LazyDict, LazyDict]:
    experiment_name = f"cosmos_kd_student_{size_name}_libero"

    student = LazyDict(
        dict(
            defaults=[
                f"/experiment/{TEACHER_EXPERIMENT_NAME}",
                "_self_",
            ],
            model=dict(
                config=dict(
                    # fsdp_shard_size=8 inherited from the base experiment assumes 8 GPUs; both the
                    # KD script (train_kd.py, plain single-process 2-GPU) and the optional baseline
                    # run (single-GPU torchrun) need this degenerated to no sharding, same reasoning
                    # as ../experiment.py's own tiny-net job.
                    fsdp_shard_size=1,
                    net=dict(num_blocks=num_blocks),
                ),
            ),
            checkpoint=dict(
                # Each consumer sets its own load_path: init_student_from_teacher.py never loads
                # one (skip_load_model=True), the baseline run points this at
                # init_student_from_teacher.py's output, and run_libero_eval.py overrides it via
                # --ckpt_path. Empty here so nothing loads implicitly if a consumer forgets to.
                load_path="",
            ),
            trainer=dict(
                callbacks=dict(
                    # Registered so a baseline (non-KD) torchrun run of this experiment can use the
                    # exact same submit_sweep.py override
                    # (trainer.callbacks.loss_csv.csv_path=<job_name-based path>) the existing
                    # tiny-net job already relies on -- without this, that override would target a
                    # config path that doesn't exist on this experiment and fail at launch. Same
                    # callback, same reasoning as ../experiment.py's own registration.
                    loss_csv=L(LossCsvCallback)(
                        csv_path=f"{_OUTPUT_ROOT}/cosmos_policy/cosmos_v2_finetune/{experiment_name}/train_loss.csv"
                    ),
                ),
            ),
            job=dict(
                name=experiment_name,
            ),
        )
    )

    # Mirrors experiment.py's own __inference_only pattern (cosmos_dit_wm_tiny_libero_from_scratch
    # -> its __inference_only variant): narrows the SDE's sigma range for inference-time sampling,
    # same as the released checkpoint's own inference config.
    inference_only = LazyDict(
        dict(
            defaults=[
                f"/experiment/{experiment_name}",
                "_self_",
            ],
            model=dict(
                config=dict(
                    sde=dict(
                        sigma_max=80.0,
                        sigma_min=4.0,
                    ),
                ),
            ),
            job=dict(
                group="cosmos_v2_inference",
                name=f"{experiment_name}__inference_only",
            ),
        )
    )

    return student, inference_only


_cs = ConfigStore.instance()
for _size_name, _num_blocks in STUDENT_SIZES.items():
    _student, _inference_only = _make_student_experiment(_size_name, _num_blocks)
    for _experiment in (_student, _inference_only):
        _cs.store(group="experiment", package="_global_", name=_experiment["job"]["name"], node=_experiment)
