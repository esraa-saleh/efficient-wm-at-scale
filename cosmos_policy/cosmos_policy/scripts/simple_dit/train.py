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
Train the simple DiT BC policy (see model.py) on one or more LIBERO task demo hdf5 files.

Standalone: plain argparse + a single-process/single-GPU training loop, deliberately independent
of the Cosmos Policy Hydra config system, FSDP/torchrun, and DCP checkpointer -- this policy is
small enough (~10-15M params at the defaults) that none of that machinery is needed, and it has no
tokenizer/T5 conditioning to load, so it starts training immediately.

Objective: rectified-flow / flow-matching BC, the same family the existing Cosmos Policy loss
uses. For a ground-truth (normalized) action chunk x1 and noise x0 ~ N(0, I):
    x_t = (1 - t) * x0 + t * x1,   t ~ U(0, 1)
    target velocity v = x1 - x0
    loss = MSE(model(images, proprio, x_t, t), v)

Usage:
    python -m cosmos_policy.scripts.simple_dit_bc.train \
        --data-dir /path/to/single_task_data \
        --work-dir /tmp/simple_dit_bc_demo \
        --max-iter 5000

`--data-dir` can point straight at the same single-task data directory
`train_from_scratch_bc_demo.py` already produces (`<work_dir>/single_task_data`), or any directory
containing LIBERO demo *.hdf5 files (searched recursively) -- e.g. pointed at
`LIBERO-Cosmos-Policy/success_only/libero_object_regen/` this trains one policy across that whole
suite's ~10 tasks, disambiguated via `--t5-embeddings-path`'s per-task language embeddings (see
model.py's `TaskEmbedder`).
"""

import argparse
import csv
import json
import math
import pathlib
import time

import torch
from torch.utils.data import DataLoader

from cosmos_policy.scripts.simple_dit_bc.dataset import SimpleLiberoChunkDataset, find_hdf5_files
from cosmos_policy.scripts.simple_dit_bc.model import SimpleDiTBC


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--data-dir", required=True, help="Directory containing LIBERO demo *.hdf5 files (searched recursively)."
    )
    parser.add_argument(
        "--t5-embeddings-path",
        required=True,
        help="Path to t5_embeddings.pkl (e.g. LIBERO-Cosmos-Policy/success_only/t5_embeddings.pkl), "
        "used to condition the policy on which task it's doing.",
    )
    parser.add_argument(
        "--work-dir", required=True, help="Directory to write checkpoints, loss CSV, and dataset stats to."
    )
    parser.add_argument("--chunk-size", type=int, default=16)
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


def make_lr_lambda(warmup_iters: int, max_iter: int, min_lr_ratio: float):
    """Linear warmup, then cosine decay from peak LR down to `min_lr_ratio` * peak by `max_iter`.

    Without the decay, LR sits at its peak for the entire post-warmup run, which just makes the
    loss oscillate around a floor instead of settling -- e.g. simple_dit_bc's loss.csv flattening
    out at ~0.1 for thousands of iterations rather than continuing to drop.
    """

    def fn(step: int) -> float:
        if step < warmup_iters:
            return (step + 1) / max(1, warmup_iters)
        progress = (step - warmup_iters) / max(1, max_iter - warmup_iters)
        progress = min(1.0, progress)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr_ratio + (1 - min_lr_ratio) * cosine

    return fn


def save_checkpoint(path: pathlib.Path, model: torch.nn.Module, iteration: int, args: argparse.Namespace) -> None:
    torch.save({"model": model.state_dict(), "iteration": iteration, "args": vars(args)}, path)
    print(f"Saved checkpoint to {path}")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    work_dir = pathlib.Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # Every checkpoint already embeds `vars(args)` under its "args" key (see save_checkpoint), which
    # is what eval.py actually reads to reconstruct the model and default --t5-embeddings-path -- this
    # is purely a human-readable copy at the run root, so `data-dir`/`t5-embeddings-path`/etc. for a
    # given run can be checked (e.g. `cat config.json`) without loading a checkpoint in torch.
    config_path = work_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Saved training config to {config_path}")

    hdf5_paths = find_hdf5_files(args.data_dir)
    if not hdf5_paths:
        raise FileNotFoundError(f"No *.hdf5 files found under {args.data_dir}")
    print(f"Training on {len(hdf5_paths)} demo file(s): {hdf5_paths}")

    dataset = SimpleLiberoChunkDataset(
        hdf5_paths,
        args.t5_embeddings_path,
        chunk_size=args.chunk_size,
        image_size=args.image_size,
        augment=not args.no_augment,
    )
    dataset.save_stats(work_dir / "dataset_stats.json")
    dataset.save_task_instructions(work_dir / "task_instructions.json")
    print(
        f"Loaded {len(dataset.episodes)} demo(s), {len(dataset)} total timesteps, "
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
    model = SimpleDiTBC(
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
        csv.writer(f).writerow(["iteration", "loss", "timestamp"])

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
            x1 = batch["action_chunk"].to(device)

            x0 = torch.randn_like(x1)
            t = torch.rand(x1.shape[0], device=device)
            x_t = (1 - t.view(-1, 1, 1)) * x0 + t.view(-1, 1, 1) * x1
            target_v = x1 - x0

            pred_v = model(agentview_img, wrist_img, proprio, x_t, t, task_emb)
            loss = torch.nn.functional.mse_loss(pred_v, target_v)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            if iteration % args.log_every == 0:
                elapsed = time.time() - start_time
                print(f"Iteration: {iteration}, loss: {loss.item():.6f}, elapsed: {elapsed:.1f}s")
                with open(loss_csv_path, "a", newline="") as f:
                    csv.writer(f).writerow([iteration, loss.item(), time.strftime("%Y-%m-%dT%H:%M:%S")])

            if iteration > 0 and iteration % args.save_every == 0:
                save_checkpoint(checkpoint_dir / f"iter_{iteration:09d}.pt", model, iteration, args)

            iteration += 1

    save_checkpoint(checkpoint_dir / f"iter_{iteration:09d}.pt", model, iteration, args)
    print("Done with training.")


if __name__ == "__main__":
    main()
