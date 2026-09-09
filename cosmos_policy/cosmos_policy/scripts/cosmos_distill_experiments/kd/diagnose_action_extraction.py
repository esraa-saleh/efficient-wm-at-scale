# SPDX-License-Identifier: Apache-2.0
"""Throwaway diagnostic: is the teacher_on_demos ROTATION COLLAPSE caused by
extract_action_chunk_from_latent_sequence's reverse-tile-AND-AVERAGE, or does the teacher
genuinely generate ~0 rotation?

For a sample of ketchup demo states it runs the exact generate_teacher_native_targets generation
(num_steps configurable), then pulls the RAW action / future-proprio / value latent frames out of
`generated_sample` and, instead of averaging the ~112 reverse-tiled copies, keeps all of them so we
can look at:

  * spread ACROSS the 112 copies, per action dim (is rotation noisier *relative to its signal*?)
  * mean-over-copies (what the extractor uses) vs copy[0] vs median  -- does averaging shrink
    rotation more than translation/gripper?
  * both vs the demo action for the same state

Writes <out_dir>/action_extraction_diag.npz with the raw arrays for offline analysis and prints a
summary table.

    python -m cosmos_policy.scripts.cosmos_distill_experiments.kd.diagnose_action_extraction \
        --data_dir  <.../success_only/libero_object_regen> \
        --t5_text_embeddings_path <.../success_only/t5_embeddings.pkl> \
        --out_dir   <somewhere writable> \
        [--num_denoising_steps 5] [--n_states 48] [--batch_size 8] [--seed 0]
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from cosmos_policy.constants import ACTION_DIM, PROPRIO_DIM
from cosmos_policy.datasets.libero_dataset import LIBERODataset
from cosmos_policy.experiments.robot.cosmos_utils import (
    extract_action_chunk_from_latent_sequence,
    extract_value_from_latent_sequence,
)
from cosmos_policy.scripts.cosmos_distill_experiments.kd import batch_prep
from cosmos_policy.scripts.cosmos_distill_experiments.kd.synthetic_generation import _LATENT_IDX
from cosmos_policy.scripts.cosmos_distill_experiments.kd.teacher_loader import load_teacher

_DIMS = ["x", "y", "z", "roll", "pitch", "yaw", "grip"]

_DATASET_KWARGS = dict(
    chunk_size=16,
    use_wrist_images=True,
    use_third_person_images=True,
    use_proprio=True,
    normalize_proprio=True,
    normalize_actions=True,
    num_duplicates_per_image=4,
    return_value_function_returns=True,
    gamma=0.99,
    use_image_aug=False,
    use_stronger_image_aug=False,
    demonstration_sampling_prob=1.0,
    success_rollout_sampling_prob=0.5,
)


def _all_copies(latent_frame: torch.Tensor, unit_shape: tuple[int, int]) -> torch.Tensor:
    """latent_frame (B, C', H', W') -> (B, K, u0, u1) : the K whole reverse-tiled copies, NOT
    averaged. Mirrors extract_action_chunk_from_latent_sequence exactly up to the final mean."""
    b = latent_frame.shape[0]
    flat = latent_frame.reshape(b, -1)
    n = flat.shape[1]
    u = unit_shape[0] * unit_shape[1]
    k = n // u
    return flat[:, : k * u].reshape(b, k, unit_shape[0], unit_shape[1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--t5_text_embeddings_path", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--task_names", nargs="*", default=["ketchup"])
    ap.add_argument("--num_denoising_steps", type=int, default=5)
    ap.add_argument("--n_states", type=int, default=48)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--teacher_checkpoint", default="")
    ap.add_argument("--teacher_experiment_name", default="")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    ds = LIBERODataset(
        data_dir=args.data_dir,
        t5_text_embeddings_path=args.t5_text_embeddings_path,
        rollout_data_dir="",
        task_names=args.task_names or None,
        **_DATASET_KWARGS,
    )
    # spread the sampled states across the whole task (every episode, not just the first)
    idxs = np.linspace(0, ds.num_steps - 1, num=min(args.n_states, ds.num_steps), dtype=int).tolist()

    load_kw = dict(to_device=args.device)
    if args.teacher_checkpoint:
        load_kw["checkpoint"] = args.teacher_checkpoint
    if args.teacher_experiment_name:
        load_kw["experiment_name"] = args.teacher_experiment_name
    print(f"loading teacher on {args.device} ...", flush=True)
    teacher, _ = load_teacher(**load_kw)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    loader = DataLoader(ds, batch_size=args.batch_size, sampler=idxs, num_workers=4, collate_fn=default_collate)

    demo_act, mean_act, copy0_act, med_act = [], [], [], []
    copy_std_act, copy_mean_act_full = [], []
    demo_fp, mean_fp, copy0_fp = [], [], []
    val_all = []

    ai = int(_LATENT_IDX["action"])
    fpi = int(_LATENT_IDX["future_proprio"])
    vi = int(_LATENT_IDX["value"])

    for bn, data_batch in enumerate(loader):
        b = data_batch["actions"].shape[0]
        demo = data_batch["actions"].float().cpu().numpy()  # (b,16,7) normalized
        seed_b = args.seed + bn * 1000
        data_batch = batch_prep.move_batch_to_device(data_batch, args.device)
        with torch.no_grad(), batch_prep.policy_autocast():
            _raw, latent_state, condition = teacher.get_data_and_condition(dict(data_batch))
            nb = dict(data_batch)
            nb["num_conditional_frames"] = teacher.config.min_num_conditional_frames
            gen = teacher.generate_samples_from_batch(
                nb,
                n_sample=b,
                num_steps=args.num_denoising_steps,
                seed=seed_b,
                is_negative_prompt=False,
                skip_vae_encoding=True,
                previous_generated_latent=condition.gt_frames,
                return_orig_clean_latent_frames=False,
            )
            dev = gen.device
            bidx = torch.arange(b, device=dev)
            act_frame = gen[bidx, :, torch.full((b,), ai, device=dev), :, :]  # (b,C',H',W')
            fp_frame = gen[bidx, :, torch.full((b,), fpi, device=dev), :, :]

            act_copies = _all_copies(act_frame, (16, ACTION_DIM)).float().cpu().numpy()  # (b,K,16,7)
            fp_copies = _all_copies(fp_frame, (1, PROPRIO_DIM)).float().cpu().numpy()  # (b,K,1,9)

            mean_a = extract_action_chunk_from_latent_sequence(
                gen, (16, ACTION_DIM), torch.full((b,), ai, dtype=torch.int64, device=dev)
            ).float().cpu().numpy()
            mean_f = extract_action_chunk_from_latent_sequence(
                gen, (1, PROPRIO_DIM), torch.full((b,), fpi, dtype=torch.int64, device=dev)
            ).squeeze(1).float().cpu().numpy()
            val = extract_value_from_latent_sequence(
                gen, torch.full((b,), vi, dtype=torch.int64, device=dev)
            ).float().cpu().numpy()

        demo_act.append(demo)
        mean_act.append(mean_a)
        copy0_act.append(act_copies[:, 0])
        med_act.append(np.median(act_copies, axis=1))
        copy_std_act.append(act_copies.std(axis=1))  # (b,16,7) spread across K copies
        copy_mean_act_full.append(act_copies.mean(axis=1))
        demo_fp.append(data_batch["future_proprio"].float().cpu().numpy())
        mean_fp.append(mean_f)
        copy0_fp.append(fp_copies[:, 0, 0])
        val_all.append(val)
        print(f"  batch {bn+1}: {b} states", flush=True)

    demo_act = np.concatenate(demo_act).reshape(-1, 7)
    mean_act = np.concatenate(mean_act).reshape(-1, 7)
    copy0_act = np.concatenate(copy0_act).reshape(-1, 7)
    med_act = np.concatenate(med_act).reshape(-1, 7)
    copy_std_act = np.concatenate(copy_std_act).reshape(-1, 7)

    np.savez(
        os.path.join(args.out_dir, "action_extraction_diag.npz"),
        demo_act=demo_act, mean_act=mean_act, copy0_act=copy0_act, med_act=med_act,
        copy_std_act=copy_std_act,
        demo_fp=np.concatenate(demo_fp), mean_fp=np.concatenate(mean_fp),
        copy0_fp=np.concatenate(copy0_fp), val=np.concatenate(val_all),
        num_denoising_steps=args.num_denoising_steps,
    )

    def row(name, a):
        return name.ljust(12) + "  ".join(f"{a[i]:+7.3f}" for i in range(7))

    print("\n=== per-dim, pooled over", demo_act.shape[0], "chunk steps (normalized [-1,1]) ===")
    print("dim         " + "  ".join(f"{d:>7}" for d in _DIMS))
    print(row("demo |.|", np.abs(demo_act).mean(0)))
    print(row("mean |.|", np.abs(mean_act).mean(0)))
    print(row("copy0 |.|", np.abs(copy0_act).mean(0)))
    print(row("median|.|", np.abs(med_act).mean(0)))
    print()
    print(row("demo  mean", demo_act.mean(0)))
    print(row("gen   mean", mean_act.mean(0)))
    print(row("copy0 mean", copy0_act.mean(0)))
    print()
    print(row("copy-std", copy_std_act.mean(0)), "  <- spread across the ~112 reverse-tiled copies")
    print(row("std/|mean|", copy_std_act.mean(0) / (np.abs(mean_act).mean(0) + 1e-6)),
          "  <- relative spread (high => averaging kills this dim)")
    print()
    print(row("corr demo,gen", np.array([np.corrcoef(demo_act[:, i], mean_act[:, i])[0, 1] for i in range(7)])))
    print(row("corr demo,cp0", np.array([np.corrcoef(demo_act[:, i], copy0_act[:, i])[0, 1] for i in range(7)])))
    print(f"\nsaved: {os.path.join(args.out_dir, 'action_extraction_diag.npz')}")


if __name__ == "__main__":
    main()
