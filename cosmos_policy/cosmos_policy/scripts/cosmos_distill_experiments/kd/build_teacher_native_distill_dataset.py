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
Builds a standalone teacher-native KD dataset: for each of `--num_batches` raw batches drawn from
LIBERODataset (both demo AND rollout examples -- both are valid source pools for synthesis, no
filtering to one or the other), every example's real action is discarded and replaced with the
TEACHER'S OWN freely-generated action for that state -- action, future state, and value are all
sampled JOINTLY, via one real multi-step `generate_samples_from_batch` call with nothing held fixed
(`synthetic_generation.generate_teacher_native_targets` -- see its own module docstring for exactly
how, and why this can't reuse training's single-`denoise()`-call shortcut: there is no ground truth
to noise for a state/action pair nobody actually executed).

This is a deliberately SEPARATE pipeline from both build_distill_dataset.py (pure real demo/rollout
data, real recorded action, no teacher-imagined action) and build_synthetic_distill_dataset.py (the
recorded action perturbed and held FIXED, only future state/value generated) -- see
synthetic_generation.py's module docstring for how the three relate. Here the action is neither the
real recorded one nor a perturbation of it: it's whatever the teacher itself would have done from
this state, unconstrained.

On-disk format: after generation, `synthetic_generation.splice_teacher_native_targets_into_batch`
builds a new LIBERODataset-shaped batch with the teacher's action/future-state/value spliced in
(everything else -- current images/proprio/text embedding -- untouched), forces
`world_model_sample_mask=0`/`value_function_sample_mask=0` (the "bc" category: action, future state,
AND value are all genuine denoising targets here, since all three are the teacher's own
self-consistent answer for this state -- see that splice function's own docstring for why this
differs from the counterfactual pipeline's `world_model_sample_mask=1`), then this script runs THAT
through `teacher.training_step(...)` -- the exact same call build_distill_dataset.py and
build_synthetic_distill_dataset.py already make -- and writes the result via distill_dataset.py's
existing `ShardWriter`, with `x0` set to the TEACHER's own prediction instead of a real ground truth
(there isn't one -- same reasoning as build_synthetic_distill_dataset.py's own module docstring for
why `teacher_x0`/`x0` are identical here too, making the downstream loss collapse to pure
distillation regardless of `ground_truth_loss_weight`). Every existing consumer of `ShardWriter`'s
format already knows how to read this dataset, with zero new consumption code required.

Usage:
    python -m cosmos_policy.scripts.cosmos_distill_experiments.kd.build_teacher_native_distill_dataset \\
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
    generate_teacher_native_targets,
    splice_teacher_native_targets_into_batch,
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
    print(f"Building {total_examples} teacher-native examples ({num_batches} raw batches x batch_size={batch_size}) into {out_dir}...")

    for batch_idx in range(num_batches):
        data_batch = next(batch_iter)
        data_batch = batch_prep.move_batch_to_device(data_batch, device)

        targets = generate_teacher_native_targets(
            teacher,
            data_batch,
            num_steps=num_denoising_steps,
            seed=seed + batch_idx,  # varies per batch -- otherwise every batch samples identically
        )
        native_batch = splice_teacher_native_targets_into_batch(data_batch, targets)

        with torch.no_grad(), batch_prep.policy_autocast():
            output_batch, _ = teacher.training_step(native_batch, batch_idx)
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
    print(f"Done: {writer.num_written} teacher-native examples written to {out_dir}")


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
    params = kd_params.load_build_teacher_native_params(args.build_params, overrides=overrides or None)

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
        num_denoising_steps=params.num_denoising_steps,
        examples_per_shard=params.examples_per_shard,
        device=params.device,
        seed=params.seed,
    )


if __name__ == "__main__":
    main()
