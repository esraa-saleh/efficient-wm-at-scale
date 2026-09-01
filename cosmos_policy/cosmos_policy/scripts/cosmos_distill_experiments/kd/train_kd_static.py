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
Trains a student against a precomputed distillation dataset (build_distill_dataset.py's output)
instead of querying the teacher live. Needs only ONE GPU (student_device): the teacher is never
loaded here at all -- its work was already done once, when the dataset was built. See
distill_dataset.py's module docstring for the on-disk format and build_distill_dataset.py's for why
this exists (reusing the same precomputed targets across several student runs instead of re-paying
the teacher's forward-pass cost every time).

The training loop itself mirrors train_kd.py's tail half almost exactly (same denoise()/
combined_kd_loss/checkpoint structure) -- the real difference is where (xt, sigma, condition,
teacher_x0, x0) comes from: `params.batch_size` individually, independently, uniformly sampled
examples from DistillShardDataset, assembled into one batch via `distill_dataset.sample_micro_batch`,
rather than a live `teacher.training_step(...)` call. Since DistillShardDataset holds individual
examples (not sealed groups -- see distill_dataset.py's module docstring), every training batch is
a fresh random mix, not a fixed regrouping of whatever the builder happened to batch together.

If `params.synthetic_dataset_dir` is set, a SECOND `sample_micro_batch` call (against a
build_synthetic_distill_dataset.py-built dataset -- stored in this exact same on-disk format, see
that script's module docstring) contributes a second, always-distill-only loss term: total loss =
loss(real batch) + loss(synthetic batch). Both terms flow into one `optimizer.step()` -- this is
NOT two separate optimizer updates.

No eval of any kind happens in this process -- deliberately. Every `params.checkpoint_every`
iterations this writes a versioned checkpoint (checkpoint_io.save_versioned_checkpoint) under
`run_dir/checkpoints/iter_NNNNNNNNN/` and KEEPS it (never overwritten, never deleted) -- unlike
train_kd.py's single overwritten model.pt/train_state.pt, since periodic_libero_eval_static.py (a
genuinely separate job -- see its own module docstring) needs every one of them still on disk to
sim-eval independently, on its own schedule, without racing this training loop. The only thing this
writes at the very end is `run_dir/TRAINING_DONE`, a plain marker (not an eval) so that job knows no
further checkpoints are coming and can stop polling once it's caught up.

Usage:
    python -m cosmos_policy.scripts.cosmos_distill_experiments.kd.train_kd_static \\
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
    combined_kd_loss,
    compute_named_latent_idx,
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

    latent_idx_by_name = compute_named_latent_idx()
    action_latent_idx = latent_idx_by_name["action"]

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

        # Every example is sampled independently and uniformly at random each iteration (no fixed
        # groups, no fixed order to exhaust/reshuffle) -- exactly what DistillShardDataset storing
        # individual examples, not sealed micro-batches, was for.
        micro_batch = sample_micro_batch(dataset, params.batch_size, params.student_device)

        with policy_autocast(), torch.cuda.device(params.student_device):
            student_x0 = student.denoise(micro_batch["xt"], micro_batch["sigma"], micro_batch["condition"]).x0
        loss, distill_loss, ground_truth_loss = combined_kd_loss(
            student_x0.float(),
            micro_batch["teacher_x0"].float(),
            micro_batch["x0"].float(),
            params.ground_truth_loss_weight,
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
            # ground_truth_loss_weight=0.0 -- always distill-only for the synthetic term, regardless
            # of the real term's own weight above (see this module's own docstring).
            _, synthetic_distill_loss, _ = combined_kd_loss(
                synthetic_student_x0.float(),
                synthetic_micro_batch["teacher_x0"].float(),
                synthetic_micro_batch["x0"].float(),
                0.0,
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
            sample_type_proportions = sample_type_proportions_from_condition(
                micro_batch["condition"], latent_idx_by_name["action"]
            )
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
