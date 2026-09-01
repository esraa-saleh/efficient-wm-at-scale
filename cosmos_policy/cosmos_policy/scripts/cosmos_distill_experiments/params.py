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
Schema + loader for this folder's (cosmos_distill_experiments, formerly
cosmos_dit_wm_tiny_libero_object_suite) experiment values. The actual values live in
conf/runs/train.yaml (the same file sweep.py's RunConfig/submit_sweep.py validate and launch
train.yaml from), not a separate params.yaml -- see RunConfig's own docstring on why: train.yaml's
`experiment` field is both the Hydra experiment name IT defines (via these architecture fields) and
launches, and every other torchrun run (smoketest.yaml, baseline_1b_train.yaml) references that
already-registered experiment by name instead of duplicating these values. load_params() below
reads conf/runs/train.yaml directly and filters it down to just the fields this schema declares
(mapping train.yaml's `experiment` field to `experiment_name` -- the one name mismatch between the
two schemas).

experiment.py plugs every field of the loaded params into its LazyDict as a plain dict overlay --
NOT a `defaults`-composed Hydra structured config. That was tried twice and confirmed broken both
times by actually running the pipeline (not just checking config resolution):

1. Composing a *partial* dataclass (e.g. just model_channels/num_blocks/num_heads) into `net` via a
   `defaults` entry makes that DictConfig struct-locked to `object_type=TinyNetSize` -- even though
   `OmegaConf.to_container(resolve=True)` shows all of net's other fields correctly inherited (that
   call flattens away the struct typing, which is what made the first attempt at this *look* fine),
   the *live* DictConfig stays struct-locked to only the 3 fields the dataclass declares. Real
   framework code somewhere in model instantiation probes `net_config.parameters` (unrelated to any
   field we touch) and that raises `ConfigAttributeError: Key 'parameters' not in 'TinyNetSize'`
   instead of the silent `None` a plain (non-structured) DictConfig would give -- confirmed via an
   actual smoketest run (job 18781181), which FAILED with exactly that traceback partway through
   model construction. Checking only `load_config(...)` + `to_container(...)` was not enough to catch
   this; only an actual end-to-end run surfaced it.
2. Separately (and unrelated to the above): composing a foreign dataclass onto `fsdp_shard_size`
   (plain `int` on the model config dataclass) or `checkpoint.load_path` (on `CheckpointConfig`, an
   attrs dataclass) fails OmegaConf's structured-merge type check outright at config-resolution time
   already ("X is not a subclass of Y") -- confirmed separately, before even getting to run anything.

So for every value here, the only way that's actually been proven to work end-to-end in this
codebase's LazyConfig system is a plain dict overlay -- the exact same mechanism every other
experiment file already uses for this kind of override (e.g.
`conditioner=dict(text=dict(dropout_rate=0.0))` in cosmos_policy_experiment_configs.py). This module
exists so those overlay values are named, documented, and configurable in one place (conf/runs/
train.yaml) instead of scattered as bare literals through experiment.py's LazyDict -- not so they're
independently Hydra-composable via a separate `key=value` syntax the way sweep.py's RunConfig fields
are (that would need `@hydra.main`/`hydra.compose()`, and experiment.py runs at Hydra *registration*
time -- before "experiment=..." is even parsed -- so there's no active Hydra context here to compose
against regardless of the struct-typing issue above). load_params() below sidesteps that entirely by
using plain `OmegaConf.load()` on conf/runs/train.yaml -- no Hydra context required, no `defaults`
composition into the training config's DictConfig, so none of the struct-locking failure modes above
apply to it.

Overriding any of these at launch time still goes through the same CLI mechanism as every other
training config value in this repo -- e.g. `model.config.net.model_channels=256
model.config.net.num_blocks=6` on the training command line (see train.sbatch), exactly like
`trainer.max_iter` or `checkpoint.save_iter` are already overridden there. To change a *default*
(not a one-off launch override), edit conf/runs/train.yaml.
"""

import pathlib
from dataclasses import dataclass, fields

from omegaconf import MISSING, OmegaConf

_TRAIN_YAML = pathlib.Path(__file__).resolve().parent / "conf" / "runs" / "train.yaml"


@dataclass(frozen=True)
class TinyLiberoObjectSuiteParams:
    experiment_name: str = MISSING
    base_experiment: str = MISSING
    fsdp_shard_size: int = MISSING
    model_channels: int = MISSING
    num_blocks: int = MISSING
    num_heads: int = MISSING
    checkpoint_load_path: str = MISSING
    inference_sigma_max: float = MISSING
    inference_sigma_min: float = MISSING
    inference_job_group: str = MISSING


def load_params() -> TinyLiberoObjectSuiteParams:
    """Loads conf/runs/train.yaml, filtered down to just this schema's fields (train.yaml also
    carries plenty of RunConfig-only fields -- job_name, data_root, suites, max_iter, ... -- that
    aren't part of this schema and must not reach OmegaConf.to_object() below, which validates
    against exactly TinyLiberoObjectSuiteParams's fields and nothing else), validated against
    TinyLiberoObjectSuiteParams's schema (raises if a field is missing or the wrong type). Maps
    train.yaml's `experiment` to `experiment_name` -- the one field name that differs between
    RunConfig and this schema."""
    raw = OmegaConf.load(_TRAIN_YAML)
    field_names = {f.name for f in fields(TinyLiberoObjectSuiteParams)} - {"experiment_name"}
    values = {name: raw[name] for name in field_names}
    values["experiment_name"] = raw["experiment"]
    schema = OmegaConf.structured(TinyLiberoObjectSuiteParams)
    merged = OmegaConf.merge(schema, OmegaConf.create(values))
    return OmegaConf.to_object(merged)
