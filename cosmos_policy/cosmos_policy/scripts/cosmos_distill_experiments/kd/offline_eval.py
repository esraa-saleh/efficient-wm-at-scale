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
Offline fidelity diagnostics for a trained KD student, run against a fixed held-out data slice
(shuffle=False) -- a cheap sanity check to run before committing to a full LIBERO simulator
evaluation (run_libero_eval.py, which this script does not replace or wrap).

Reports, for each batch:
  - global denoiser MSE (student vs. teacher x0, same metric train_kd.py optimizes, but held-out)
  - action-region latent MSE (sliced to just the action latent frame)
  - decoded action L1 / max error (actions live directly in the latent -- no VAE decode needed,
    unlike image latents -- see extract_action_chunk_from_latent_sequence's own docstring)

Usage:
    python -m cosmos_policy.scripts.cosmos_distill_experiments.kd.offline_eval \\
      --student_experiment_name cosmos_kd_student_1b_libero \\
      --student_checkpoint /path/to/model.pt \\
      --data_dir /path/to/libero_object_regen \\
      --t5_text_embeddings_path /path/to/t5_embeddings.pkl \\
      --dataset_stats_path /path/to/dataset_statistics.json
"""

import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cosmos_policy.datasets.libero_dataset import LIBERODataset
from cosmos_policy.experiments.robot.cosmos_utils import (
    extract_action_chunk_from_latent_sequence,
    load_dataset_stats,
    unnormalize_actions,
)
from cosmos_policy.scripts.cosmos_distill_experiments.kd import batch_prep
from cosmos_policy.scripts.cosmos_distill_experiments.kd.teacher_loader import load_policy_model, load_teacher


def evaluate_batch(teacher, student, student_device: str, data_batch: dict, dataset_stats: dict) -> dict:
    data_batch = {k: (v.to("cuda:0") if torch.is_tensor(v) else v) for k, v in data_batch.items()}

    with torch.no_grad(), batch_prep.policy_autocast():
        output_batch, _ = teacher.training_step(data_batch, 0)
        xt, sigma, condition = output_batch["xt"], output_batch["sigma"], output_batch["condition"]
        teacher_x0 = output_batch["model_pred"].x0

        xt_s = xt.to(student_device)
        sigma_s = sigma.to(student_device)
        condition_s = batch_prep.move_condition(condition, student_device)
        student_x0 = student.denoise(xt_s, sigma_s, condition_s).x0

    teacher_x0_on_student = teacher_x0.to(student_device).float()
    student_x0 = student_x0.float()

    global_mse = F.mse_loss(student_x0, teacher_x0_on_student).item()

    action_indices = data_batch["action_latent_idx"].to(student_device)
    action_shape = tuple(data_batch["actions"].shape[1:])  # (chunk_size, action_dim)
    teacher_action_latent = extract_action_chunk_from_latent_sequence(teacher_x0_on_student, action_shape, action_indices)
    student_action_latent = extract_action_chunk_from_latent_sequence(student_x0, action_shape, action_indices)
    action_region_mse = F.mse_loss(student_action_latent, teacher_action_latent).item()

    teacher_actions = unnormalize_actions(teacher_action_latent.cpu().numpy(), dataset_stats)
    student_actions = unnormalize_actions(student_action_latent.cpu().numpy(), dataset_stats)
    action_error = teacher_actions - student_actions

    return {
        "global_denoiser_mse": global_mse,
        "action_region_latent_mse": action_region_mse,
        "decoded_action_l1": float(abs(action_error).mean()),
        "decoded_action_max_error": float(abs(action_error).max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student_experiment_name", required=True)
    parser.add_argument("--student_checkpoint", required=True)
    parser.add_argument("--teacher_experiment_name", default=None)
    parser.add_argument("--teacher_checkpoint", default=None)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--t5_text_embeddings_path", required=True)
    parser.add_argument("--dataset_stats_path", required=True)
    parser.add_argument("--num_batches", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--student_device", default="cuda:1")
    args = parser.parse_args()

    print("Loading teacher on cuda:0...")
    teacher_kwargs = {}
    if args.teacher_experiment_name:
        teacher_kwargs["experiment_name"] = args.teacher_experiment_name
    if args.teacher_checkpoint:
        teacher_kwargs["checkpoint"] = args.teacher_checkpoint
    teacher, _ = load_teacher(to_device="cuda:0", **teacher_kwargs)
    teacher.eval()

    print(f"Loading student ({args.student_experiment_name}) on {args.student_device}...")
    student, _ = load_policy_model(
        to_device=args.student_device, experiment_name=args.student_experiment_name, checkpoint=args.student_checkpoint
    )
    student.eval()

    dataset_stats = load_dataset_stats(args.dataset_stats_path)
    dataset = LIBERODataset(data_dir=args.data_dir, t5_text_embeddings_path=args.t5_text_embeddings_path, chunk_size=16)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    totals = {"global_denoiser_mse": 0.0, "action_region_latent_mse": 0.0, "decoded_action_l1": 0.0, "decoded_action_max_error": 0.0}
    n = 0
    for data_batch in loader:
        if n >= args.num_batches:
            break
        metrics = evaluate_batch(teacher, student, args.student_device, data_batch, dataset_stats)
        for key, value in metrics.items():
            totals[key] = totals[key] + value if key != "decoded_action_max_error" else max(totals[key], value)
        n += 1
        print(f"batch {n}: {metrics}")

    print("\n=== Offline fidelity summary (mean over batches, max for decoded_action_max_error) ===")
    for key, value in totals.items():
        summarized = value if key == "decoded_action_max_error" else value / n
        print(f"{key}: {summarized:.6f}")


if __name__ == "__main__":
    main()
