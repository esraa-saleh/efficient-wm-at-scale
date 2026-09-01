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
Thin launcher around `cosmos_policy.scripts.train`, needed to work around a gap in
_src/imaginaire/utils/distributed.py's distributed.init() on Slurm allocations that grant a
cgroup-restricted subset of a node's CPUs (e.g. an interactive job with 8 of a node's CPUs, rather
than the whole node): NVML's nvmlDeviceGetCpuAffinity() returns an *empty* affinity list in that
case, and the unguarded `os.sched_setaffinity(0, [])` call then raises
`OSError: [Errno 22] Invalid argument`, crashing before training starts -- on any experiment, not
just ours.

Rather than editing that vendored NVIDIA source, this monkeypatches `os.sched_setaffinity` (before
the real train script is even imported) so an empty CPU list is a no-op -- leaving the process's
existing, cgroup-assigned affinity untouched -- instead of crashing. On a normal full-node
allocation (non-empty affinity list), behavior is unchanged.

Copied from ../cosmos_dit_wm/run_train.py rather than imported from there, so this folder doesn't
depend on a sibling directory outside itself (that sibling is untracked/moved to draft_code/ -- see
the repo's own dependency audit) -- keep the two in sync by hand if either one changes, they're not
meant to diverge. This folder's own baseline_*.sbatch/train.sbatch scripts depend on this copy
directly (`-m cosmos_policy.scripts.cosmos_distill_experiments.run_train`).

Usage: identical to `python -m cosmos_policy.scripts.train`, e.g.:
    torchrun --nproc_per_node=1 --master_port=12341 -m cosmos_policy.scripts.cosmos_distill_experiments.run_train \
      --config=cosmos_policy/config/config.py -- \
      experiment="cosmos_dit_wm_tiny_libero_from_scratch" trainer.max_iter=3
"""

import os
import runpy

_real_sched_setaffinity = os.sched_setaffinity


def _safe_sched_setaffinity(pid: int, mask) -> None:
    if not mask:
        return
    _real_sched_setaffinity(pid, mask)


os.sched_setaffinity = _safe_sched_setaffinity

if __name__ == "__main__":
    runpy.run_module("cosmos_policy.scripts.train", run_name="__main__")
