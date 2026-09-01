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
Builds a static KD distillation dataset once, so train_kd_static.py can train (possibly several)
students against it repeatedly without ever re-paying the teacher's forward-pass cost. Exists
because train_kd.py's live loop re-queries the full 2B teacher on every single iteration of every
single run -- fine for one run, wasteful across a sweep of student sizes/configs against the same
fixed data distribution.

For each of `--num_batches` raw batches drawn from LIBERODataset (pure real demo/rollout data --
no synthetic/perturbed-action augmentation; that's a separate, standalone pipeline now, see
build_synthetic_distill_dataset.py), queries the teacher `--noise_draws_per_batch` times --
`teacher.training_step(...)` redraws its
own sigma/epsilon internally on every call, so calling it repeatedly on the SAME raw batch already
gives distinct noise levels/difficulties for free, no extra code needed to vary that. This is what
makes the resulting static set span a spread of denoising difficulty per example instead of
freezing in whatever single noise draw happened to be sampled once -- the exact risk flagged before
this was built (see distill_dataset.py's module docstring for the storage format this writes).

Usage:
    python -m cosmos_policy.scripts.cosmos_distill_experiments.kd.build_distill_dataset \\
        --build_params <path/to/build_params.yaml>
"""

import argparse

import torch
from torch.utils.data import DataLoader

from cosmos_policy.datasets.libero_dataset import LIBERODataset
from cosmos_policy.scripts.cosmos_distill_experiments.kd import batch_prep
from cosmos_policy.scripts.cosmos_distill_experiments.kd import params as kd_params
from cosmos_policy.scripts.cosmos_distill_experiments.kd.distill_dataset import ShardWriter
from cosmos_policy.scripts.cosmos_distill_experiments.kd.teacher_loader import load_teacher


def build(
    *,
    teacher_checkpoint: str,
    teacher_experiment_name: str,
    data_dir: str,
    t5_text_embeddings_path: str,
    rollout_data_dir: str,
    task_names: list,
    out_dir: str,
    num_batches: int,
    noise_draws_per_batch: int,
    batch_size: int,
    examples_per_shard: int,
    device: str,
    seed: int,
) -> None:
    torch.manual_seed(seed)

    print(f"Loading teacher ({teacher_checkpoint}) on {device}...")
    teacher, _ = load_teacher(
        to_device=device,
        checkpoint=teacher_checkpoint,
        experiment_name=teacher_experiment_name,
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    dataset = LIBERODataset(
        data_dir=data_dir,
        t5_text_embeddings_path=t5_text_embeddings_path,
        rollout_data_dir=rollout_data_dir,
        task_names=task_names or None,
        **batch_prep.DATASET_KWARGS,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    batch_iter = batch_prep.endless_batches(loader)

    writer = ShardWriter(out_dir, examples_per_shard=examples_per_shard)
    total_examples = num_batches * noise_draws_per_batch * batch_size
    print(
        f"Building {total_examples} individual examples ({num_batches} raw batches x "
        f"{noise_draws_per_batch} noise draws each x batch_size={batch_size}) into {out_dir}..."
    )

    for batch_idx in range(num_batches):
        data_batch = next(batch_iter)
        data_batch = batch_prep.move_batch_to_device(data_batch, device)

        for draw in range(noise_draws_per_batch):
            with torch.no_grad(), batch_prep.policy_autocast():
                output_batch, _ = teacher.training_step(data_batch, batch_idx)
            writer.add(
                xt=output_batch["xt"],
                sigma=output_batch["sigma"],
                condition=output_batch["condition"],
                teacher_x0=output_batch["model_pred"].x0,
                x0=output_batch["x0"],  # ground truth -- see batch_prep.py's combined_kd_loss
            )

        if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
            print(f"  raw batch {batch_idx + 1}/{num_batches} ({writer.num_written} examples flushed so far)")

    writer.close()
    print(f"Done: {writer.num_written} individual examples written to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build_params", required=True)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--t5_text_embeddings_path", default=None)
    parser.add_argument("--rollout_data_dir", default=None)
    parser.add_argument("--task_names", nargs="+", default=None, help="Restrict to tasks whose filename contains one of these (case-insensitive), e.g. --task_names ketchup")
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    overrides = {
        k: v
        for k, v in dict(
            data_dir=args.data_dir,
            t5_text_embeddings_path=args.t5_text_embeddings_path,
            rollout_data_dir=args.rollout_data_dir,
            task_names=args.task_names,
            out_dir=args.out_dir,
        ).items()
        if v is not None
    }
    params = kd_params.load_build_params(args.build_params, overrides=overrides or None)

    build(
        teacher_checkpoint=params.teacher_checkpoint,
        teacher_experiment_name=params.teacher_experiment_name,
        data_dir=params.data_dir,
        t5_text_embeddings_path=params.t5_text_embeddings_path,
        rollout_data_dir=params.rollout_data_dir,
        task_names=params.task_names,
        out_dir=params.out_dir,
        num_batches=params.num_batches,
        noise_draws_per_batch=params.noise_draws_per_batch,
        batch_size=params.batch_size,
        examples_per_shard=params.examples_per_shard,
        device=params.device,
        seed=params.seed,
    )


if __name__ == "__main__":
    main()
