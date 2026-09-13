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
Param-count calibration for KD students: the depth-reduction plan (see
../kd/net_experiments.py's module docstring) only ever varies MiniTrainDIT's `num_blocks` --
width/heads stay fixed at the real teacher's values (model_channels=2048, num_heads=16, the same
COSMOS_V2_2B_NET this repo already uses for the released 2B checkpoint, see
cosmos_policy/_src/predict2/configs/text2world/defaults/net.py:80-96). This script exists so
picking a `num_blocks` for a "~1B"/"~700M"/"~500M" student -- or a sub-500M one, see
net_experiments.py's STUDENT_SIZES entries down to "90m" -- is a lookup against real parameter
counts instead of a guess.

Swept down to num_blocks=1: the architectural floor for this depth-reduction scheme (fixed
width/heads, only depth varies), since a "0-block" student would be just the shared
patchify/embed/output trunk with no transformer processing at all -- not meaningful, not swept.
Needs a GPU node to run (this import chain pulls in transformer_engine, which needs a CUDA
context to import even for a meta-device build that does no forward/backward pass -- CPU-only
fails at import time, not at the actual param count).

Usage:
    python -m cosmos_policy.scripts.cosmos_distill_experiments.kd.student_sizes
"""

import torch

from cosmos_policy._src.imaginaire.utils.count_params import count_params
from cosmos_policy._src.predict2.networks.minimal_v4_dit import MiniTrainDIT

# Exact non-depth args of COSMOS_V2_2B_NET (net.py:80-96) -- only num_blocks varies here.
_TEACHER_NET_KWARGS = dict(
    max_img_h=240,
    max_img_w=240,
    max_frames=128,
    in_channels=16,
    out_channels=16,
    patch_spatial=2,
    patch_temporal=1,
    model_channels=2048,
    num_heads=16,
    concat_padding_mask=True,
    pos_emb_cls="rope3d",
    pos_emb_learnable=True,
    pos_emb_interpolation="crop",
    use_adaln_lora=True,
    adaln_lora_dim=256,
    extra_per_block_abs_pos_emb=True,
)

TEACHER_NUM_BLOCKS = 28


def count_student_params(num_blocks: int) -> int:
    """Meta-device MiniTrainDIT instantiation at the given depth (no real allocation -- meta
    tensors carry shape/dtype only, which is all count_params needs)."""
    with torch.device("meta"):
        net = MiniTrainDIT(num_blocks=num_blocks, **_TEACHER_NET_KWARGS)
    return count_params(net)


if __name__ == "__main__":
    teacher_params = count_student_params(TEACHER_NUM_BLOCKS)
    print(f"teacher (num_blocks={TEACHER_NUM_BLOCKS}): {teacher_params / 1e9:.3f}B params\n")
    print(f"{'num_blocks':>10}  {'params':>12}  {'params (B)':>10}")
    for num_blocks in range(1, TEACHER_NUM_BLOCKS + 1):
        params = count_student_params(num_blocks)
        print(f"{num_blocks:>10}  {params:>12,}  {params / 1e9:>10.3f}")
