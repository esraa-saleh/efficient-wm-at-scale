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
Identical to train_kd.py / train_kd_av.py in every respect (same KDParams schema, same
teacher/student device placement, same LIBERODataset/DataLoader, same resume/checkpoint/logging
shape) except the loss: this uses batch_prep.combined_kd_loss_action_only instead of
combined_kd_loss/combined_kd_loss_action_value_only, so ONLY the action-chunk slot of the shared
video latent is supervised -- both the future-state slots AND the value slot are excluded from the
loss entirely (narrower than train_kd_av.py, which still supervises value).

Same cross-check as train_kd_av.py: asserts the real per-batch `action_latent_idx` (LIBERODataset's
own, set directly in `__getitem__`) matches batch_prep.compute_action_value_latent_idx()'s
derivation every iteration -- what makes train_kd_static_action.py's use of that same derived
constant trustworthy (see train_kd_av.py's own module docstring for the full reasoning; this only
needs the action half of it, since combined_kd_loss_action_only never reads value_latent_idx at
all).

teacher.training_step(data_batch, iteration) is called UNMODIFIED, under torch.no_grad() -- see
train_kd.py's own module docstring for what that call provides in one shot. The forward pass itself
is completely unchanged from train_kd.py: the student still denoises every slot of the video latent
every step, only the loss stops supervising every slot but the action one.

If `params.synthetic_dataset_dir` is set, a SECOND, always-distill-only action-only loss term is
added -- same two-term total loss as train_kd.py's own module docstring, just narrowed to
action-only like the real term here. No teacher call needed for it -- see train_kd.py's own
docstring for why.

Usage:
    python -m cosmos_policy.scripts.cosmos_distill_experiments.kd.train_kd_action --kd_params <path/to/kd_params.yaml>

Optional overrides (used by ../submit_sweep.py's KD branch instead of editing the yaml per run):
    --data_dir, --t5_text_embeddings_path, --rollout_data_dir, --run_dir
"""

import argparse
import itertools
import pathlib

import torch
from torch.utils.data import DataLoader

from cosmos_policy.datasets.libero_dataset import LIBERODataset
from cosmos_policy.scripts.cosmos_distill_experiments.kd import batch_prep, checkpoint_io
from cosmos_policy.scripts.cosmos_distill_experiments.kd import params as kd_params
from cosmos_policy.scripts.cosmos_distill_experiments.kd.batch_prep import DATASET_KWARGS as _DATASET_KWARGS
from cosmos_policy.scripts.cosmos_distill_experiments.kd.distill_dataset import DistillShardDataset, sample_micro_batch
from cosmos_policy.scripts.cosmos_distill_experiments.kd.teacher_loader import load_policy_model, load_teacher


def train(params: kd_params.KDParams) -> None:
    torch.manual_seed(params.seed)
    run_dir = pathlib.Path(params.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    action_latent_idx, _ = batch_prep.compute_action_value_latent_idx()
    print(f"Action-only loss: action_latent_idx={action_latent_idx}")
    latent_idx_by_name = {"action": action_latent_idx}

    print(f"Loading teacher ({params.teacher_checkpoint}) on {params.teacher_device}...")
    teacher, _ = load_teacher(
        to_device=params.teacher_device,
        checkpoint=params.teacher_checkpoint,
        experiment_name=params.teacher_experiment_name,
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    print(f"Loading student ({params.student_net_experiment_name}) on {params.student_device}...")
    student, _ = load_policy_model(
        to_device=params.student_device,
        experiment_name=params.student_net_experiment_name,
        checkpoint=params.student_init_path,
    )
    student.train()

    optimizer = torch.optim.AdamW(student.net.parameters(), lr=params.lr)  # student params ONLY

    start_iteration = 0
    train_state_path = run_dir / "train_state.pt"
    if train_state_path.exists():
        model_state, optimizer_state, last_iteration = checkpoint_io.load_student_for_resume(train_state_path)
        student.load_state_dict(model_state, strict=True)
        optimizer.load_state_dict(optimizer_state)
        start_iteration = last_iteration + 1
        print(f"Resuming from {train_state_path} at iteration {start_iteration}")

    dataset = LIBERODataset(
        data_dir=params.data_dir,
        t5_text_embeddings_path=params.t5_text_embeddings_path,
        rollout_data_dir=params.rollout_data_dir,
        **_DATASET_KWARGS,
    )
    loader = DataLoader(dataset, batch_size=params.batch_size, shuffle=True, num_workers=0)

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

    for iteration, data_batch in enumerate(itertools.cycle(loader), start=0):
        if iteration < start_iteration:
            continue
        if iteration >= params.max_iter:
            break

        # Cross-checks compute_action_value_latent_idx()'s derivation against this batch's real,
        # LIBERODataset-assigned index -- see this module's own docstring for why that matters.
        assert torch.all(data_batch["action_latent_idx"] == action_latent_idx), (
            f"data_batch['action_latent_idx'] does not match compute_action_value_latent_idx()'s "
            f"derived {action_latent_idx} -- DATASET_KWARGS and that derivation have drifted apart."
        )

        data_batch = batch_prep.move_batch_to_device(data_batch, params.teacher_device)

        with torch.no_grad(), batch_prep.policy_autocast():
            output_batch, _ = teacher.training_step(data_batch, iteration)
        xt, sigma, condition = output_batch["xt"], output_batch["sigma"], output_batch["condition"]
        teacher_x0 = output_batch["model_pred"].x0
        ground_truth_x0 = output_batch["x0"]

        xt_s = xt.to(params.student_device)
        sigma_s = sigma.to(params.student_device)
        condition_s = batch_prep.move_condition(condition, params.student_device)

        with batch_prep.policy_autocast(), torch.cuda.device(params.student_device):
            student_x0 = student.denoise(xt_s, sigma_s, condition_s).x0
        loss, distill_loss, ground_truth_loss = batch_prep.combined_kd_loss_action_only(
            student_x0.float(),
            teacher_x0.to(params.student_device).float(),
            ground_truth_x0.to(params.student_device).float(),
            params.ground_truth_loss_weight,
            action_latent_idx,
        )

        total_loss = loss
        synthetic_distill_loss = None
        synthetic_micro_batch = None
        synthetic_student_x0 = None
        if synthetic_dataset is not None:
            synthetic_micro_batch = sample_micro_batch(synthetic_dataset, params.batch_size, params.student_device)
            with batch_prep.policy_autocast(), torch.cuda.device(params.student_device):
                synthetic_student_x0 = student.denoise(
                    synthetic_micro_batch["xt"], synthetic_micro_batch["sigma"], synthetic_micro_batch["condition"]
                ).x0
            _, synthetic_distill_loss, _ = batch_prep.combined_kd_loss_action_only(
                synthetic_student_x0.float(),
                synthetic_micro_batch["teacher_x0"].float(),
                synthetic_micro_batch["x0"].float(),
                0.0,
                action_latent_idx,
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
                synthetic_action_given_loss, synthetic_action_predicted_loss = (
                    batch_prep.split_synthetic_action_distill_loss(
                        synthetic_student_x0.float(),
                        synthetic_micro_batch["teacher_x0"].float(),
                        synthetic_micro_batch["condition"].condition_video_input_mask_B_C_T_H_W,
                        action_latent_idx,
                    )
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
            per_slot = batch_prep.per_slot_kd_losses(
                student_x0.float(),
                teacher_x0.to(params.student_device).float(),
                ground_truth_x0.to(params.student_device).float(),
                params.ground_truth_loss_weight,
                latent_idx_by_name,
            )
            per_slot_items = {name: tuple(x.item() for x in vals) for name, vals in per_slot.items()}
            sample_type_proportions = batch_prep.sample_type_proportions_exact(
                data_batch["world_model_sample_mask"], data_batch["value_function_sample_mask"]
            )
            batch_prep.append_loss_csv_with_slots(
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
            checkpoint_io.save_student(student, optimizer, iteration, run_dir)
            print(f"Saved checkpoint at iteration {iteration} to {run_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kd_params", required=True)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--t5_text_embeddings_path", default=None)
    parser.add_argument("--rollout_data_dir", default=None)
    parser.add_argument("--synthetic_dataset_dir", default=None)
    parser.add_argument("--run_dir", default=None)
    args = parser.parse_args()

    overrides = {
        k: v
        for k, v in dict(
            data_dir=args.data_dir,
            t5_text_embeddings_path=args.t5_text_embeddings_path,
            rollout_data_dir=args.rollout_data_dir,
            synthetic_dataset_dir=args.synthetic_dataset_dir,
            run_dir=args.run_dir,
        ).items()
        if v is not None
    }
    params = kd_params.load_params(args.kd_params, overrides=overrides or None)

    train(params)


if __name__ == "__main__":
    main()
