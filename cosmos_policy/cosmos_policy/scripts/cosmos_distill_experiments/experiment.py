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
Hydra experiment definition for the `cosmos_dit_wm_tiny_libero_object_suite` job (Stage 5 of the
CosmosDiTFrameReplaceBC -> real Cosmos Policy roadmap, see ../cosmos_dit_wm/cosmos_dit_frame_replace_stages.tex
Section 5): the real production recipe (WAN2.1 VAE tokenizer, real 9-token frame-replace layout,
real `HybridEDMSDE`, real *unmasked* joint loss across all 9 tokens) at "tiny" net size, restricted
to the `libero_object` LIBERO suite. Every tunable value -- including the identifiers Hydra needs
before composition starts (experiment_name, base_experiment) -- lives in conf/runs/train.yaml, not as a
literal here; see params.py's docstring for the schema/loader and for why these get applied to the
training LazyDict below as a plain dict overlay rather than composed in via a `defaults` entry (a
`defaults`-composed structured config was tried and confirmed broken end-to-end -- job 18781181).

This registers the same two experiment names previously defined by `_make_small_libero_experiment`
in config/experiment/cosmos_dit_wm_experiments.py (values unchanged, just inlined for one preset
instead of parameterized across tiny/small/base) -- see README.md in this folder for the launch
command and how this differs from the small/base variants that stayed in that shared file.

This file physically lives under scripts/cosmos_distill_experiments/ (not config/experiment/) so
this job's config and launcher live in one place. It still gets registered with Hydra via a one-line
import shim at config/experiment/cosmos_distill_experiments_registration.py, because Hydra's config
registration (import_all_modules_from_package("cosmos_policy.config.experiment", ...) in
config_v2.py) only scans modules physically inside that package.

The Hydra *experiment* names registered below (cosmos_dit_wm_tiny_libero_from_scratch and its
__inference_only variant) are unchanged from before this folder was renamed from
scripts/cosmos_dit_wm_tiny_libero_object_suite/ -- scripts/cosmos_dit_wm/submit_sweep.py launches a
separate real run (job_name=cosmos_dit_wm_tiny_libero_object_suite_v2) against that exact experiment
name, so it has to keep resolving to this file regardless of what this folder itself is called.
"""

import os

from hydra.core.config_store import ConfigStore

from cosmos_policy._src.imaginaire.lazy_config import LazyCall as L
from cosmos_policy._src.imaginaire.lazy_config import LazyDict
from cosmos_policy.scripts.cosmos_distill_experiments.loss_csv_callback import LossCsvCallback
from cosmos_policy.scripts.cosmos_distill_experiments import params

# _src/imaginaire/config.py's Config.path_local reads this same env var for checkpoints, defaulting
# to /tmp/imaginaire4-output (node-local, ephemeral -- gone if the job lands on a different node
# next time). Point both checkpoints and our loss CSV at the same *persistent* project-storage
# location by setting this before launching, e.g.:
#   export IMAGINAIRE_OUTPUT_ROOT="$COSMOS_POLICY_STORAGE/cosmos_dit_wm_output"
_OUTPUT_ROOT = os.environ.get("IMAGINAIRE_OUTPUT_ROOT", "/tmp/imaginaire4-output")

_PARAMS = params.load_params()

cosmos_dit_wm_tiny_libero_from_scratch = LazyDict(
    dict(
        defaults=[
            f"/experiment/{_PARAMS.base_experiment}",
            "_self_",
        ],
        model=dict(
            config=dict(
                fsdp_shard_size=_PARAMS.fsdp_shard_size,
                net=dict(
                    model_channels=_PARAMS.model_channels,
                    num_blocks=_PARAMS.num_blocks,
                    num_heads=_PARAMS.num_heads,
                ),
            ),
        ),
        checkpoint=dict(
            load_path=_PARAMS.checkpoint_load_path,
        ),
        trainer=dict(
            callbacks=dict(
                # Pure observability (doesn't touch model/data/optimization) -- appends
                # (iteration, loss, timestamp) to a CSV every step, for later plotting. The real
                # Trainer only logs loss to the console (IterSpeed) and to wandb, neither of which
                # is a convenient flat file to plot from -- especially with wandb forced offline on
                # this cluster (no internet access on compute nodes).
                loss_csv=L(LossCsvCallback)(
                    csv_path=f"{_OUTPUT_ROOT}/cosmos_policy/cosmos_v2_finetune/{_PARAMS.experiment_name}/train_loss.csv"
                ),
            ),
        ),
        job=dict(
            name=_PARAMS.experiment_name,
        ),
    )
)

# Mirrors cosmos_predict2_2b_480p_libero__inference_only's pattern: narrow the SDE's sigma range
# for inference-time sampling (vs. the wider range used for training), same as the released
# checkpoint's inference config. checkpoint.load_path gets overridden separately by
# run_libero_eval.py's --ckpt_path at eval time, not baked in here.
cosmos_dit_wm_tiny_libero_from_scratch__inference_only = LazyDict(
    dict(
        defaults=[
            f"/experiment/{_PARAMS.experiment_name}",
            "_self_",
        ],
        model=dict(
            config=dict(
                sde=dict(
                    sigma_max=_PARAMS.inference_sigma_max,
                    sigma_min=_PARAMS.inference_sigma_min,
                ),
            ),
        ),
        job=dict(group=_PARAMS.inference_job_group, name=f"{_PARAMS.experiment_name}__inference_only"),
    )
)

_cs = ConfigStore.instance()
for _experiment in [
    cosmos_dit_wm_tiny_libero_from_scratch,
    cosmos_dit_wm_tiny_libero_from_scratch__inference_only,
]:
    _cs.store(group="experiment", package="_global_", name=_experiment["job"]["name"], node=_experiment)
