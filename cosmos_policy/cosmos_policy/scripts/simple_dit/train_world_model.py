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
Train `SimpleDiTWorldModel` (see model.py) -- a small world model distilled from the full Cosmos
Policy teacher, meant to score candidate action chunks inside a best-of-N/MPC loop (see
eval_mpc.py) at a fraction of the teacher's cost.

Unlike train.py's flow-matching BC objective, this is direct MSE regression: given a current
observation and a *candidate* action chunk (not necessarily the one actually taken), predict that
action's likely future-state embedding and value. There's no distributional-multimodality concern
here the way there is for actions, so no noise/timestep input is needed.

    loss = MSE(pred_future_agentview_feat, target_future_agentview_feat)
         + MSE(pred_future_wrist_feat, target_future_wrist_feat)
         + MSE(pred_value, target_value)

`target_future_*_feat` is computed at train time by running the model's own `SmallImageEncoder`
(under `torch.no_grad()`) over the target future frames -- see model.py's `return_pre_proj` option
-- rather than a value precomputed and frozen ahead of time, so the target always matches whatever
`--embed-dim` this run is using.

Usage:
    python -m cosmos_policy.scripts.simple_dit_bc.train_world_model \
        --data-dir /path/to/libero_object_regen \
        --t5-embeddings-path /path/to/t5_embeddings.pkl \
        --work-dir /tmp/simple_dit_world_model_demo \
        --max-iter 30000

`--synthetic-cache-path` (optional) points at precompute_teacher_targets.py's output -- without it,
training uses only "real" (recorded-action, real-future) samples, which is enough to sanity-check
the pipeline but won't give the model any exposure to off-trajectory candidate actions (see
world_model_dataset.py's module docstring for why that matters for MPC).
"""

import argparse
import csv
import json
import pathlib
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cosmos_policy.scripts.simple_dit_bc.dataset import find_hdf5_files
from cosmos_policy.scripts.simple_dit_bc.model import SimpleDiTWorldModel
from cosmos_policy.scripts.simple_dit_bc.train import make_lr_lambda, save_checkpoint
from cosmos_policy.scripts.simple_dit_bc.world_model_dataset import WorldModelDistillationDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--data-dir", required=True, help="Directory containing LIBERO demo *.hdf5 files (searched recursively)."
    )
    parser.add_argument("--t5-embeddings-path", required=True, help="Path to t5_embeddings.pkl.")
    parser.add_argument(
        "--work-dir", required=True, help="Directory to write checkpoints, loss CSV, and dataset stats to."
    )
    parser.add_argument(
        "--synthetic-cache-path",
        default=None,
        help="Output of precompute_teacher_targets.py -- teacher-imagined counterfactual samples. "
        "Omit to train on real (on-trajectory) samples only.",
    )
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument(
        "--k-future", type=int, default=16, help="How many steps ahead the 'future' frame/value target is."
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for the real-branch MC return.")
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable pad+crop/brightness/contrast image augmentation (on by default).",
    )
    parser.add_argument("--task-emb-dim", type=int, default=1024, help="T5 pooled embedding dim -- leave at default.")
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--num-blocks", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-iters", type=int, default=200)
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument(
        "--min-lr-ratio",
        type=float,
        default=0.05,
        help="Cosine-decay the LR from its peak down to this fraction of --lr by --max-iter.",
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    work_dir = pathlib.Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    config_path = work_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Saved training config to {config_path}")

    hdf5_paths = find_hdf5_files(args.data_dir)
    if not hdf5_paths:
        raise FileNotFoundError(f"No *.hdf5 files found under {args.data_dir}")
    print(f"Training on {len(hdf5_paths)} demo file(s): {hdf5_paths}")

    dataset = WorldModelDistillationDataset(
        hdf5_paths,
        args.t5_embeddings_path,
        synthetic_cache_path=args.synthetic_cache_path,
        chunk_size=args.chunk_size,
        k_future=args.k_future,
        gamma=args.gamma,
        image_size=args.image_size,
        augment=not args.no_augment,
    )
    dataset.save_stats(work_dir / "dataset_stats.json")
    dataset.save_task_instructions(work_dir / "task_instructions.json")
    print(
        f"Loaded {len(dataset.episodes)} demo(s), {len(dataset.real_index)} real sample(s), "
        f"{len(dataset.synthetic_samples)} synthetic sample(s), {len(dataset)} total, "
        f"{len(dataset.instructions)} unique task(s): {dataset.instructions}"
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleDiTWorldModel(
        action_dim=7,
        proprio_dim=9,
        task_emb_dim=args.task_emb_dim,
        chunk_size=args.chunk_size,
        embed_dim=args.embed_dim,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
    ).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {num_params:,} params ({num_params / 1e6:.2f}M) on device={device}.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, make_lr_lambda(args.warmup_iters, args.max_iter, args.min_lr_ratio)
    )

    loss_csv_path = work_dir / "train_loss.csv"
    with open(loss_csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["iteration", "loss", "agentview_loss", "wrist_loss", "value_loss", "timestamp"])

    checkpoint_dir = work_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    iteration = 0
    model.train()
    start_time = time.time()
    while iteration < args.max_iter:
        for batch in loader:
            if iteration >= args.max_iter:
                break

            agentview_img = batch["agentview_img"].to(device)
            wrist_img = batch["wrist_img"].to(device)
            proprio = batch["proprio"].to(device)
            task_emb = batch["task_emb"].to(device)
            action_chunk = batch["action_chunk"].to(device)
            future_agentview_img = batch["future_agentview_img"].to(device)
            future_wrist_img = batch["future_wrist_img"].to(device)
            target_value = batch["value"].to(device)

            pred_agentview_feat, pred_wrist_feat, pred_value = model(
                agentview_img, wrist_img, proprio, action_chunk, task_emb
            )
            with torch.no_grad():
                _, target_agentview_feat = model.image_encoder(future_agentview_img, return_pre_proj=True)
                _, target_wrist_feat = model.image_encoder(future_wrist_img, return_pre_proj=True)

            agentview_loss = F.mse_loss(pred_agentview_feat, target_agentview_feat)
            wrist_loss = F.mse_loss(pred_wrist_feat, target_wrist_feat)
            value_loss = F.mse_loss(pred_value, target_value)
            loss = agentview_loss + wrist_loss + value_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            if iteration % args.log_every == 0:
                elapsed = time.time() - start_time
                print(
                    f"Iteration: {iteration}, loss: {loss.item():.6f} "
                    f"(agentview: {agentview_loss.item():.6f}, wrist: {wrist_loss.item():.6f}, "
                    f"value: {value_loss.item():.6f}), elapsed: {elapsed:.1f}s"
                )
                with open(loss_csv_path, "a", newline="") as f:
                    csv.writer(f).writerow(
                        [
                            iteration,
                            loss.item(),
                            agentview_loss.item(),
                            wrist_loss.item(),
                            value_loss.item(),
                            time.strftime("%Y-%m-%dT%H:%M:%S"),
                        ]
                    )

            if iteration > 0 and iteration % args.save_every == 0:
                save_checkpoint(checkpoint_dir / f"iter_{iteration:09d}.pt", model, iteration, args)

            iteration += 1

    save_checkpoint(checkpoint_dir / f"iter_{iteration:09d}.pt", model, iteration, args)
    print("Done with training.")


if __name__ == "__main__":
    main()
