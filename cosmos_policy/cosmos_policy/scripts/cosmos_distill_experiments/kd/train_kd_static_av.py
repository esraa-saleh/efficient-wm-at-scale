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
Identical to train_kd_static.py in every respect (same StaticTrainParams schema, same
DistillShardDataset, same resume/checkpoint/logging shape) except the loss: this uses
batch_prep.combined_kd_loss_action_value_only instead of batch_prep.combined_kd_loss, so only the
action-chunk and value slots of the shared video latent are supervised -- the future-state slots
(current/future proprio, wrist image, third-person image) are excluded from the loss entirely, not
just down-weighted (see combined_kd_loss_action_value_only's own docstring for why this is the only
lever available here: the static dataset's xt/sigma were already noised at build time with every
slot as a denoising target, and rebuilding it is out of scope for this variant).

The student's forward pass (student.denoise(...)) is completely unchanged -- it still denoises
every slot of the video latent every step, since there's nothing else to occupy those token
positions in the real teacher/student's fixed grid. Only the loss stops supervising the
future-state ones.

If `params.synthetic_dataset_dir` is set, a SECOND `sample_micro_batch` call (against a
build_synthetic_distill_dataset.py-built dataset) contributes a second, always-distill-only
action/value-only loss term -- same two-term total loss as train_kd_static.py's own module
docstring, just narrowed to action+value like the real term here.

Usage:
    python -m cosmos_policy.scripts.cosmos_distill_experiments.kd.train_kd_static_av \
        --static_params <path/to/static_params.yaml>
"""

import argparse
import itertools
import pathlib

import torch

from cosmos_policy.scripts.cosmos_distill_experiments.kd import checkpoint_io
from cosmos_policy.scripts.cosmos_distill_experiments.kd import params as kd_params
from cosmos_policy.scripts.cosmos_distill_experiments.kd.batch_prep import (
    append_loss_csv_with_slots,
    combined_kd_loss_action_value_only,
    compute_action_value_latent_idx,
    per_slot_kd_losses,
    policy_autocast,
    sample_type_proportions_from_condition,
    split_synthetic_action_distill_loss,
)
from cosmos_policy.scripts.cosmos_distill_experiments.kd.distill_dataset import DistillShardDataset, sample_micro_batch
from cosmos_policy.scripts.cosmos_distill_experiments.kd.teacher_loader import load_policy_model


def train(params: kd_params.StaticTrainParams) -> None:
    torch.manual_seed(params.seed)
    run_dir = pathlib.Path(params.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    action_latent_idx, value_latent_idx = compute_action_value_latent_idx()
    print(f"Action/value-only loss: action_latent_idx={action_latent_idx}, value_latent_idx={value_latent_idx}")
    latent_idx_by_name = {"action": action_latent_idx, "value": value_latent_idx}

    print(f"Loading student ({params.student_net_experiment_name}) on {params.student_device}...")
    student, _ = load_policy_model(
        to_device=params.student_device,
        experiment_name=params.student_net_experiment_name,
        checkpoint=params.student_init_path,
    )
    student.train()

    optimizer = torch.optim.AdamW(student.net.parameters(), lr=params.lr)

    start_iteration = 0
    resume_state = checkpoint_io.load_latest_versioned_checkpoint_for_resume(run_dir)
    if resume_state is not None:
        model_state, optimizer_state, last_iteration = resume_state
        student.load_state_dict(model_state, strict=True)
        optimizer.load_state_dict(optimizer_state)
        start_iteration = last_iteration + 1
        print(f"Resuming from {run_dir / 'checkpoints'} at iteration {start_iteration}")

    dataset = DistillShardDataset(params.distill_dataset_dir)
    print(f"Loaded {len(dataset)} precomputed individual examples from {params.distill_dataset_dir}")
    if params.batch_size > len(dataset):
        raise ValueError(
            f"batch_size ({params.batch_size}) is larger than the static dataset "
            f"({len(dataset)} examples) -- build a bigger dataset or lower batch_size."
        )

    synthetic_dataset = None
    if params.synthetic_dataset_dir:
        synthetic_dataset = DistillShardDataset(params.synthetic_dataset_dir)
        print(f"Loaded {len(synthetic_dataset)} precomputed synthetic examples from {params.synthetic_dataset_dir}")
        if params.batch_size > len(synthetic_dataset):
            raise ValueError(
                f"batch_size ({params.batch_size}) is larger than the synthetic dataset "
                f"({len(synthetic_dataset)} examples) -- build a bigger dataset or lower batch_size."
            )

    loss_csv_path = run_dir / "train_loss.csv"

    for iteration in itertools.count():
        if iteration < start_iteration:
            continue
        if iteration >= params.max_iter:
            break

        micro_batch = sample_micro_batch(dataset, params.batch_size, params.student_device)

        with policy_autocast(), torch.cuda.device(params.student_device):
            student_x0 = student.denoise(micro_batch["xt"], micro_batch["sigma"], micro_batch["condition"]).x0
        loss, distill_loss, ground_truth_loss = combined_kd_loss_action_value_only(
            student_x0.float(),
            micro_batch["teacher_x0"].float(),
            micro_batch["x0"].float(),
            params.ground_truth_loss_weight,
            action_latent_idx,
            value_latent_idx,
        )

        total_loss = loss
        synthetic_distill_loss = None
        synthetic_micro_batch = None
        synthetic_student_x0 = None
        if synthetic_dataset is not None:
            synthetic_micro_batch = sample_micro_batch(synthetic_dataset, params.batch_size, params.student_device)
            with policy_autocast(), torch.cuda.device(params.student_device):
                synthetic_student_x0 = student.denoise(
                    synthetic_micro_batch["xt"], synthetic_micro_batch["sigma"], synthetic_micro_batch["condition"]
                ).x0
            _, synthetic_distill_loss, _ = combined_kd_loss_action_value_only(
                synthetic_student_x0.float(),
                synthetic_micro_batch["teacher_x0"].float(),
                synthetic_micro_batch["x0"].float(),
                0.0,
                action_latent_idx,
                value_latent_idx,
            )
            total_loss = loss + synthetic_distill_loss

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

        if iteration % params.log_every == 0:
            log_msg = (
                f"iteration {iteration}: kd_loss={loss.item():.6f} "
                f"(distill={distill_loss.item():.6f}, ground_truth={ground_truth_loss.item():.6f})"
            )
            synthetic_action_given_loss = None
            synthetic_action_predicted_loss = None
            if synthetic_micro_batch is not None:
                synthetic_action_given_loss, synthetic_action_predicted_loss = split_synthetic_action_distill_loss(
                    synthetic_student_x0.float(),
                    synthetic_micro_batch["teacher_x0"].float(),
                    synthetic_micro_batch["condition"].condition_video_input_mask_B_C_T_H_W,
                    action_latent_idx,
                )
            if synthetic_distill_loss is not None:
                log_msg += f" synthetic_distill_loss={synthetic_distill_loss.item():.6f}"
            split_parts = []
            if synthetic_action_given_loss is not None:
                split_parts.append(f"given={synthetic_action_given_loss:.6f}")
            if synthetic_action_predicted_loss is not None:
                split_parts.append(f"predicted={synthetic_action_predicted_loss:.6f}")
            if split_parts:
                log_msg += f" ({', '.join(split_parts)})"
            print(log_msg)
            per_slot = per_slot_kd_losses(
                student_x0.float(),
                micro_batch["teacher_x0"].float(),
                micro_batch["x0"].float(),
                params.ground_truth_loss_weight,
                latent_idx_by_name,
            )
            per_slot_items = {name: tuple(x.item() for x in vals) for name, vals in per_slot.items()}
            sample_type_proportions = sample_type_proportions_from_condition(micro_batch["condition"], action_latent_idx)
            append_loss_csv_with_slots(
                loss_csv_path,
                iteration,
                loss.item(),
                distill_loss.item(),
                ground_truth_loss.item(),
                per_slot_items,
                sample_type_proportions,
                synthetic_distill_loss.item() if synthetic_distill_loss is not None else None,
                synthetic_action_given_loss,
                synthetic_action_predicted_loss,
            )

        if iteration % params.checkpoint_every == 0 or iteration == params.max_iter - 1:
            checkpoint_dir = checkpoint_io.save_versioned_checkpoint(student, optimizer, iteration, run_dir)
            print(f"Saved checkpoint at iteration {iteration} to {checkpoint_dir}")

    (run_dir / "TRAINING_DONE").write_text("")
    print(f"Wrote {run_dir / 'TRAINING_DONE'} -- periodic_libero_eval_static.py can stop once it catches up.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--static_params", required=True)
    parser.add_argument("--distill_dataset_dir", default=None)
    parser.add_argument("--synthetic_dataset_dir", default=None)
    parser.add_argument("--run_dir", default=None)
    args = parser.parse_args()

    overrides = {
        k: v
        for k, v in dict(
            distill_dataset_dir=args.distill_dataset_dir,
            synthetic_dataset_dir=args.synthetic_dataset_dir,
            run_dir=args.run_dir,
        ).items()
        if v is not None
    }
    params = kd_params.load_static_params(args.static_params, overrides=overrides or None)

    train(params)


if __name__ == "__main__":
    main()
