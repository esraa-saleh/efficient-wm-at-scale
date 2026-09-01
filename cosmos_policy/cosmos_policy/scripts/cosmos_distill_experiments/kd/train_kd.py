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
The bespoke, 2-GPU KD training loop: frozen 2B teacher on `params.teacher_device` (cuda:0), the
depth-reduced trainable student on `params.student_device` (cuda:1), one process, plain `.to(device)`
placement -- no torchrun/FSDP/imaginaire Trainer. Deliberately not integrated into that framework's
Trainer/checkpointer, per the KD plan's locked decision #2.

For each batch, `teacher.training_step(data_batch, iteration)` -- called UNMODIFIED, under
`torch.no_grad()` -- gets the canonical (xt, sigma, condition) the real pipeline builds, the
teacher's own denoise() output (the original pure-distillation target), AND the real ground-truth
target (`output_batch["x0"]`, the same clean target the base non-KD job trains against) from one
call (see batch_prep.py's module docstring and the KD plan's finding #3). The student's loss is a
weighted blend of both targets (`batch_prep.combined_kd_loss`, `params.ground_truth_loss_weight`)
-- 0.0 reproduces this job's original KD-only design exactly.

If `params.synthetic_dataset_dir` is set, a SECOND loss term is added against a
build_synthetic_distill_dataset.py-built dataset (see its own module docstring) -- total loss =
loss(real batch) + loss(synthetic batch), the second term always distill-only regardless of
`ground_truth_loss_weight` above. Unlike the real batch, this needs NO teacher call: that dataset's
(xt, sigma, condition, teacher_x0) were already precomputed at build time (the whole point of
storing it in distill_dataset.py's existing ShardWriter format -- see that build script's module
docstring), so `distill_dataset.sample_micro_batch` + `student.denoise(...)` is all this needs,
exactly like train_kd_static.py's own real-data term.

Usage:
    python -m cosmos_policy.scripts.cosmos_distill_experiments.kd.train_kd --kd_params <path/to/kd_params.yaml>

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

    latent_idx_by_name = batch_prep.compute_named_latent_idx()
    action_latent_idx = latent_idx_by_name["action"]

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
    # Plain shuffling sampler, no DistributedSampler/parallel_state: there's no data-parallel rank
    # to shard by in a single-process 2-GPU KD job. cosmos_policy/scripts/train.py itself hand-builds
    # its DistributedSampler rather than instantiating it from config for the same underlying reason
    # ("difficult to set up ... without creating two duplicates of the dataset").
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

        data_batch = batch_prep.move_batch_to_device(data_batch, params.teacher_device)

        with torch.no_grad(), batch_prep.policy_autocast():
            output_batch, _ = teacher.training_step(data_batch, iteration)
        xt, sigma, condition = output_batch["xt"], output_batch["sigma"], output_batch["condition"]
        teacher_x0 = output_batch["model_pred"].x0
        ground_truth_x0 = output_batch["x0"]  # the same clean target the base (non-KD) job trains
        # against -- training_step already builds it via frame-replace injection on the real demo/
        # rollout data, so getting it costs nothing beyond reading one more key off the same call.

        xt_s = xt.to(params.student_device)
        sigma_s = sigma.to(params.student_device)
        condition_s = batch_prep.move_condition(condition, params.student_device)

        # denoise() internally does `net_state_in.to(device="cuda")` (bare, no index -- see
        # text2world_model.py's `self.tensor_kwargs`), which resolves via torch.cuda.current_device()
        # rather than the input tensor's own device, silently rerouting onto whatever GPU happens to
        # be "current" (process default: 0) instead of student_device. torch.cuda.device(...) makes
        # student_device the current device for this block so that resolves correctly.
        with batch_prep.policy_autocast(), torch.cuda.device(params.student_device):
            student_x0 = student.denoise(xt_s, sigma_s, condition_s).x0
        loss, distill_loss, ground_truth_loss = batch_prep.combined_kd_loss(
            student_x0.float(),
            teacher_x0.to(params.student_device).float(),
            ground_truth_x0.to(params.student_device).float(),
            params.ground_truth_loss_weight,
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
            _, synthetic_distill_loss, _ = batch_prep.combined_kd_loss(
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
