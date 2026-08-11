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
Structured-config *schema* for what submit_sweep.py can launch -- types and docs only, no values.
The actual runs (`train`, `smoketest`) are defined in conf/runs/*.yaml, composed into
`SweepConfig.runs` by conf/config.yaml's `defaults` list (each `runs@runs.<name>: <name>` entry).
Adding a new launchable run means adding a new conf/runs/<name>.yaml + one `defaults` line, not
editing this file. Every field remains overridable from the command line the normal Hydra way, e.g.:

    python -m cosmos_policy.scripts.cosmos_distill_experiments.submit_sweep \\
      launch.only=[smoketest] launch.submit=true runs.smoketest.max_iter=5

This is the one place in this folder that uses vanilla Hydra (@hydra.main + ConfigStore) rather
than this repo's own LazyConfig/override() machinery in _src/imaginaire/config.py. That machinery
(used by experiment.py) builds the *training* Config/Trainer object and is specific to that job;
submit_sweep.py is a plain submission script with no relationship to it, so it uses Hydra directly
instead of going through that layer.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass(kw_only=True)
class RunConfig:
    job_name: str = MISSING  # Becomes job.name, checkpoint dir, and loss CSV path -- must be unique across `runs`.
    experiment: str = MISSING  # Hydra experiment name (see experiment.py / conf/params.yaml's experiment_name).
    data_dir: str = MISSING  # LIBERO suite directory (relative to paths.data_root unless it starts with "/").
    max_iter: int = MISSING
    batch_size: int = MISSING
    checkpoint_save_iter: int = MISSING
    rollout_data_dir: str = ""
    extra_overrides: Dict[str, str] = field(default_factory=dict)  # Any other config.path=value overrides.
    time_limit: str = MISSING  # sbatch --time
    mem: str = MISSING  # sbatch --mem
    cpus: int = MISSING  # sbatch --cpus-per-task
    account: str = MISSING  # sbatch --account
    filename: str = ""  # Output .sbatch filename; defaults to f"{job_name}.sbatch" if empty.


@dataclass(kw_only=True)
class PathsConfig:
    # Cluster-storage paths, specific to this account's allocation -- see conf/paths.yaml (the
    # `# @package paths` header there is what places its fields under cfg.paths instead of the
    # config root). repo_root is NOT a field here -- submit_sweep.py derives it from its own file
    # location instead, since it's not a per-account value.
    data_root: str = MISSING
    output_root: str = MISSING


@dataclass
class LaunchOptions:
    # Which run(s) (keys of `runs` below) to process. Empty/unset means all of them.
    only: Optional[List[str]] = None
    # Actually call `sbatch`. Default False: (re)write the .sbatch scripts and print what would be
    # submitted, without touching the queue.
    submit: bool = False
    # Delete the selected run(s)' existing checkpoint dir (model/optim/scheduler/trainer shards,
    # latest_checkpoint.txt, DeviceMonitor, wandb offline run, loss CSV) before doing anything else,
    # so job.name auto-resume finds nothing and the run starts from iteration 0. Off by default --
    # this is a real rm -rf on real training artifacts. Only ever wipes the run(s) selected by
    # `only`, never everything, so e.g. `only=[smoketest] wipe=true` cannot touch train's checkpoints.
    wipe: bool = False


@dataclass
class SweepConfig:
    launch: LaunchOptions = field(default_factory=LaunchOptions)
    # Populated by conf/config.yaml's `- paths` defaults entry -- see conf/paths.yaml.
    paths: PathsConfig = field(default_factory=PathsConfig)
    # Populated by conf/config.yaml's `runs@runs.<name>: <name>` defaults entries, not here -- see
    # conf/runs/*.yaml for the actual train/smoketest values.
    runs: Dict[str, RunConfig] = field(default_factory=dict)


cs = ConfigStore.instance()
cs.store(name="sweep_config", node=SweepConfig)
