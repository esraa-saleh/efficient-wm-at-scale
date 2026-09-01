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

"""A minimal Callback that appends (iteration, loss, timestamp) to a CSV every training step, for
later plotting -- the real Trainer logs loss to the console (via IterSpeed) and to wandb, but
neither is a convenient flat file to plot from, especially with wandb offline on this cluster.

Copied from ../cosmos_dit_wm/loss_csv_callback.py rather than imported from there, so this folder
doesn't depend on a sibling directory outside itself (that sibling is untracked -- see the repo's
own dependency audit) -- keep the two in sync by hand if either one changes, they're not meant to
diverge.

Pure observability: doesn't touch the model, data, or optimization in any way.
"""

import csv
import os
import time

import torch

from cosmos_policy._src.imaginaire.model import ImaginaireModel
from cosmos_policy._src.imaginaire.utils.callback import Callback
from cosmos_policy._src.imaginaire.utils.distributed import rank0_only


class LossCsvCallback(Callback):
    def __init__(self, csv_path: str):
        super().__init__()
        self.csv_path = csv_path
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        if not os.path.exists(csv_path):
            with open(csv_path, "w", newline="") as f:
                csv.writer(f).writerow(["iteration", "loss", "timestamp"])

    @rank0_only
    def on_training_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow([iteration, loss.item(), time.strftime("%Y-%m-%dT%H:%M:%S")])
