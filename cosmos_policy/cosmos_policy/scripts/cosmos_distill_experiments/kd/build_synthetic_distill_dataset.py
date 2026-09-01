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
Builds a standalone synthetic KD dataset: for each of `--num_batches` raw batches drawn from
LIBERODataset (both demo AND rollout examples -- both are valid source pools for synthesis, no
filtering to one or the other), every example's real action is discarded and replaced with a
perturbed one, and the teacher is queried for the resulting future state and value -- JOINTLY, via
one real multi-step `generate_samples_from_batch` call with the perturbed action held fixed as
conditioning (`synthetic_generation.generate_synthetic_targets` -- see its own module docstring for
exactly how, and why this can't reuse training's single-`denoise()`-call shortcut: there is no
ground truth to noise for a hypothetical action nobody actually took).

This is a deliberately SEPARATE pipeline from build_distill_dataset.py, not an extension of it:
that script's static dataset is pure real demo/rollout data, no perturbation, matching the exact
Cosmos-Policy-recipe sample-type proportions the live path also trains on -- see its own module
docstring for why any perturbation mechanism was removed from it entirely rather than kept
alongside this one.

On-disk format: after generation, `synthetic_generation.splice_synthetic_targets_into_batch` builds
a new LIBERODataset-shaped batch with the perturbed action/synthetic future-state/value spliced in
(everything else -- current images/proprio/text embedding -- untouched), forces
`world_model_sample_mask=1`/`value_function_sample_mask=0` (action given, future+value are this
example's actual denoising targets), then this script runs THAT through `teacher.training_step(...)`
-- the exact same call build_distill_dataset.py already makes on real batches -- and writes the
result via distill_dataset.py's existing `ShardWriter`, with `x0` set to the TEACHER's own
prediction instead of a real ground truth (there isn't one). Passing the same tensor as both
`teacher_x0` and `x0` makes `combined_kd_loss`/`combined_kd_loss_action_value_only`/
`combined_kd_loss_action_only` collapse to pure distillation loss automatically, regardless of
whatever `ground_truth_loss_weight` a training script's real-data term uses -- exactly the
distill-only behavior this synthetic term needs (see this project's own design discussion: total
loss = loss(real batch) + loss(synthetic batch), the second term never blended with a ground-truth
target). This also means EVERY existing consumer of `ShardWriter`'s format -- `DistillShardDataset`,
`join_examples_into_micro_batch`, train_kd_static*.py's own dataset loading -- already knows how to
read a synthetic dataset built by this script, with zero new consumption code required for either
the live or static training paths.

Usage:
    python -m cosmos_policy.scripts.cosmos_distill_experiments.kd.build_synthetic_distill_dataset \\
        --build_params <path/to/build_params.yaml>
"""

import argparse

import torch
from torch.utils.data import DataLoader

from cosmos_policy.datasets.libero_dataset import LIBERODataset
from cosmos_policy.scripts.cosmos_distill_experiments.kd import batch_prep
from cosmos_policy.scripts.cosmos_distill_experiments.kd import params as kd_params
from cosmos_policy.scripts.cosmos_distill_experiments.kd.distill_dataset import ShardWriter
from cosmos_policy.scripts.cosmos_distill_experiments.kd.synthetic_generation import (
    generate_synthetic_targets,
    splice_synthetic_targets_into_batch,
)
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
    batch_size: int,
    action_perturbation_std: float,
    num_denoising_steps: int,
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

    # No filtering to demo-only or rollout-only: LIBERODataset already draws from both pools every
    # batch (DATASET_KWARGS' demonstration_sampling_prob/success_rollout_sampling_prob), and both
    # are valid sources to synthesize off of.
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
    total_examples = num_batches * batch_size
    print(f"Building {total_examples} synthetic examples ({num_batches} raw batches x batch_size={batch_size}) into {out_dir}...")

    for batch_idx in range(num_batches):
        data_batch = next(batch_iter)
        data_batch = batch_prep.move_batch_to_device(data_batch, device)

        targets = generate_synthetic_targets(
            teacher,
            data_batch,
            action_perturbation_std=action_perturbation_std,
            num_steps=num_denoising_steps,
            seed=seed + batch_idx,  # varies per batch -- otherwise every batch samples identically
        )
        synthetic_batch = splice_synthetic_targets_into_batch(data_batch, targets)

        with torch.no_grad(), batch_prep.policy_autocast():
            output_batch, _ = teacher.training_step(synthetic_batch, batch_idx)
        writer.add(
            xt=output_batch["xt"],
            sigma=output_batch["sigma"],
            condition=output_batch["condition"],
            teacher_x0=output_batch["model_pred"].x0,
            x0=output_batch["model_pred"].x0,  # no real ground truth -- see this module's own
            # docstring for why reusing the teacher's own prediction here is what makes the
            # downstream loss collapse to pure distillation automatically.
        )

        if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
            print(f"  raw batch {batch_idx + 1}/{num_batches} ({writer.num_written} examples flushed so far)")

    writer.close()
    print(f"Done: {writer.num_written} synthetic examples written to {out_dir}")


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
    params = kd_params.load_build_synthetic_params(args.build_params, overrides=overrides or None)

    build(
        teacher_checkpoint=params.teacher_checkpoint,
        teacher_experiment_name=params.teacher_experiment_name,
        data_dir=params.data_dir,
        t5_text_embeddings_path=params.t5_text_embeddings_path,
        rollout_data_dir=params.rollout_data_dir,
        task_names=params.task_names,
        out_dir=params.out_dir,
        num_batches=params.num_batches,
        batch_size=params.batch_size,
        action_perturbation_std=params.action_perturbation_std,
        num_denoising_steps=params.num_denoising_steps,
        examples_per_shard=params.examples_per_shard,
        device=params.device,
        seed=params.seed,
    )


if __name__ == "__main__":
    main()
