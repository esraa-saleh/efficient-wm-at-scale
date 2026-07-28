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
Offline precompute of "synthetic" (counterfactual) distillation targets for
`SimpleDiTWorldModel` (see model.py, world_model_dataset.py), using the full 2B-param Cosmos
Policy teacher (`nvidia/Cosmos-Policy-LIBERO-Predict2-2B`) to imagine the future state and value
of action chunks that were NEVER actually executed in any demo.

Why this needs the teacher (and why real demo data alone isn't enough): a value/dynamics model
trained only on the recorded (on-trajectory) actions in the demos never sees the off-trajectory
candidate actions a best-of-N MPC search proposes at inference time, and has no reason to
generalize to scoring them. The teacher, in contrast, can be queried on any candidate action --
including ones a real demonstrator never took -- so its predictions there are exactly the training
signal a small student world model needs to learn to imitate.

Requires a GPU (the teacher needs ~7-10GB VRAM for inference per its own README -- no training,
bf16, no optimizer state) and network access to download the HF checkpoint on first use. This
script was NOT executed/validated in the authoring session (no GPU available there) -- run it on a
handful of episodes first (small --num-episodes, --num-synthetic-per-episode 1) and sanity-check
the cached samples (e.g. via world_model_dataset.py's own sanity check, or by eyeballing a few
`future_agentview_img` tensors as images) before scaling up to the full dataset.

Counterfactual-action injection mechanism (the one genuinely new piece of glue code here, not a
simple call to an existing teacher entrypoint): `get_action()` always either samples an action
chunk from the diffusion model or reuses one already in a previous generated latent -- there is no
supported path to hand it an arbitrary "already decided" action chunk to condition future/value
prediction on. So we call `get_action(..., generate_future_state_and_value_in_parallel=False)` to
get back its `generated_latent`/`data_batch`/`latent_indices`, then directly overwrite the action
slot in that latent with our own candidate action via `replace_latent_with_action_chunk` (the same
tensor op `get_action` itself uses internally to inject the *real* action during training, and the
same pattern this repo's own multi-depth best-of-N search already uses at
cosmos_utils.py:2064-2069 to inject a newly-sampled action into a fresh latent), before passing the
modified latent into `get_future_state_prediction`/`get_value_prediction` as `previous_generated_latent`.

Usage:
    python -m cosmos_policy.scripts.simple_dit_bc.precompute_teacher_targets \
        --data-dir /path/to/libero_object_regen \
        --teacher-ckpt-path nvidia/Cosmos-Policy-LIBERO-Predict2-2B \
        --teacher-dataset-stats-path nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_dataset_statistics.json \
        --teacher-t5-embeddings-path nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_t5_embeddings.pkl \
        --student-t5-embeddings-path /path/to/t5_embeddings.pkl \
        --out-path /path/to/synthetic_cache.pt \
        --num-synthetic-per-episode 2
"""

import argparse
import pathlib
import random

import numpy as np
import torch

from cosmos_policy.constants import ACTION_DIM
from cosmos_policy.datasets.dataset_utils import calculate_dataset_statistics, decode_jpeg_bytes_dataset, resize_images
from cosmos_policy.experiments.robot.cosmos_utils import (
    get_action,
    get_future_state_prediction,
    get_model,
    get_value_prediction,
    init_t5_text_embeddings_cache,
    load_dataset_stats,
)
from cosmos_policy.experiments.robot.libero.run_libero_eval import PolicyEvalConfig
from cosmos_policy.models.policy_text2world_model import replace_latent_with_action_chunk
from cosmos_policy.scripts.simple_dit_bc.dataset import find_hdf5_files, instruction_from_filename, load_t5_embeddings

import h5py


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", required=True, help="Directory containing LIBERO demo *.hdf5 files.")
    parser.add_argument(
        "--teacher-ckpt-path", default="nvidia/Cosmos-Policy-LIBERO-Predict2-2B", help="Teacher HF repo or path."
    )
    parser.add_argument(
        "--teacher-config", default="cosmos_predict2_2b_480p_libero__inference_only", help="Teacher inference config."
    )
    parser.add_argument(
        "--teacher-config-file", default="cosmos_policy/config/config.py", help="Cosmos default config file path."
    )
    parser.add_argument(
        "--teacher-dataset-stats-path",
        default="nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_dataset_statistics.json",
        help="Dataset stats the TEACHER was trained with -- used only for talking to the teacher "
        "(action/proprio normalization on that boundary), NOT for the cache's saved action_chunk.",
    )
    parser.add_argument(
        "--teacher-t5-embeddings-path",
        default="nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_t5_embeddings.pkl",
        help="Full per-token T5 embeddings for the teacher's cross-attention (NOT simple_dit_bc's "
        "pooled single-vector embeddings -- the teacher needs the full 512-token sequence).",
    )
    parser.add_argument(
        "--student-t5-embeddings-path",
        required=True,
        help="Path to the pooled t5_embeddings.pkl used by simple_dit_bc (see dataset.py's "
        "load_t5_embeddings) -- stored in the cache for the student model to consume.",
    )
    parser.add_argument("--out-path", required=True, help="Where to write the synthetic sample cache (torch.save).")
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=96, help="Student model's image resolution.")
    parser.add_argument(
        "--num-synthetic-per-episode",
        type=int,
        default=2,
        help="Number of counterfactual (episode, timestep) samples to draw per demo episode.",
    )
    parser.add_argument(
        "--action-noise-std",
        type=float,
        default=0.3,
        help="Std of Gaussian noise added to the recorded action (in student-normalized [-1,1] "
        "space) to produce a counterfactual candidate action.",
    )
    parser.add_argument("--num-denoising-steps-action", type=int, default=5)
    parser.add_argument("--num-denoising-steps-future-state", type=int, default=1)
    parser.add_argument("--num-denoising-steps-value", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def build_teacher_cfg(args: argparse.Namespace) -> PolicyEvalConfig:
    """Reuse run_libero_eval.py's PolicyEvalConfig directly rather than hand-rolling a new config
    type -- it already has every field get_action/get_future_state_prediction/get_value_prediction
    read, with the same defaults used by the teacher's own documented LIBERO eval invocation."""
    return PolicyEvalConfig(
        suite="libero",
        config=args.teacher_config,
        ckpt_path=args.teacher_ckpt_path,
        config_file=args.teacher_config_file,
        use_wrist_image=True,
        use_proprio=True,
        normalize_proprio=True,
        unnormalize_actions=True,
        dataset_stats_path=args.teacher_dataset_stats_path,
        t5_text_embeddings_path=args.teacher_t5_embeddings_path,
        trained_with_image_aug=True,
        chunk_size=args.chunk_size,
        flip_images=False,  # demo HDF5 frames are already correctly oriented (no live-env flip)
        mask_current_state_action_for_value_prediction=False,
    )


def normalize_minmax(x: np.ndarray, x_min: np.ndarray, x_max: np.ndarray) -> np.ndarray:
    """Inverse of cosmos_utils.py's unnormalize_actions/rescale_proprio formula -- maps raw values
    to [-1, 1]. No `normalize_actions` counterpart exists in cosmos_utils.py, so this is new."""
    return 2 * (x - x_min) / (x_max - x_min) - 1


def sample_counterfactual_action(
    recorded_action_chunk_norm: np.ndarray, rng: np.random.Generator, noise_std: float
) -> np.ndarray:
    """Perturb a recorded (student-normalized, [-1,1]) action chunk with Gaussian noise, clipped
    back to [-1,1] -- a simple, broad-coverage way to sample "actions a demonstrator didn't take"
    without needing any model of what's dynamically plausible."""
    noise = rng.normal(scale=noise_std, size=recorded_action_chunk_norm.shape).astype(np.float32)
    return np.clip(recorded_action_chunk_norm + noise, -1.0, 1.0)


def inject_counterfactual_action_into_latent(
    generated_latent: torch.Tensor,
    candidate_action_chunk_teacher_norm: torch.Tensor,
    action_latent_idx: int,
) -> torch.Tensor:
    """Overwrite get_action's own diffusion-sampled action slot with a given candidate action, so
    get_future_state_prediction/get_value_prediction condition on OUR action instead of the
    teacher's. `action_latent_idx` is the scalar int from get_action's
    `return_dict["latent_indices"]["action_latent_idx"]` -- expand it to a (batch_size,) tensor
    exactly as the existing multi-depth search code does (cosmos_utils.py:2064-2069) before calling
    replace_latent_with_action_chunk (policy_text2world_model.py:45-109), which tiles the action
    chunk to fill the (C', H', W') latent-frame volume at that index.
    """
    batch_size = generated_latent.shape[0]
    action_indices = torch.full(
        (batch_size,), action_latent_idx, dtype=torch.int64, device=generated_latent.device
    )
    return replace_latent_with_action_chunk(generated_latent.clone(), candidate_action_chunk_teacher_norm, action_indices)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)

    print("Loading teacher model...")
    cfg = build_teacher_cfg(args)
    model, _ = get_model(cfg)
    teacher_stats = load_dataset_stats(cfg.dataset_stats_path)
    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)

    student_t5_embeddings = load_t5_embeddings(args.student_t5_embeddings_path)

    hdf5_paths = find_hdf5_files(args.data_dir)
    if not hdf5_paths:
        raise FileNotFoundError(f"No *.hdf5 files found under {args.data_dir}")
    print(f"Found {len(hdf5_paths)} demo file(s).")

    # Student-side normalization stats -- computed fresh over these same demos, matching exactly
    # what world_model_dataset.py's WorldModelDistillationDataset (and SimpleLiberoChunkDataset
    # before it) would compute, so the cache's action_chunk/proprio share the same scale the
    # student model is actually trained/run with. Deliberately NOT teacher_stats -- those are the
    # teacher's own training-time stats, only used for the get_action/get_future_state_prediction/
    # get_value_prediction calls below.
    raw_episodes = []
    for hdf5_path in hdf5_paths:
        instruction = instruction_from_filename(hdf5_path)
        with h5py.File(hdf5_path, "r") as f:
            demo_keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
            for demo_key in demo_keys:
                demo = f[f"data/{demo_key}"]
                raw_episodes.append(
                    {
                        "agentview_jpeg": demo["obs/agentview_rgb_jpeg"][:],
                        "wrist_jpeg": demo["obs/eye_in_hand_rgb_jpeg"][:],
                        "proprio": demo["robot_states"][:].astype(np.float32),
                        "actions": demo["actions"][:].astype(np.float32),
                        "instruction": instruction,
                    }
                )
    stats_input = {i: {"actions": ep["actions"], "proprio": ep["proprio"]} for i, ep in enumerate(raw_episodes)}
    student_stats = calculate_dataset_statistics(stats_input)

    synthetic_samples = []
    for ep_idx, ep in enumerate(raw_episodes):
        agentview_raw = decode_jpeg_bytes_dataset(ep["agentview_jpeg"])  # (T, H, W, 3) uint8
        wrist_raw = decode_jpeg_bytes_dataset(ep["wrist_jpeg"])
        num_steps = len(ep["actions"])
        student_task_emb = torch.from_numpy(student_t5_embeddings[ep["instruction"]]).float()

        candidate_ts = rng.choice(num_steps, size=min(args.num_synthetic_per_episode, num_steps), replace=False)
        for t in candidate_ts:
            t = int(t)
            remaining = num_steps - t
            if remaining >= args.chunk_size:
                recorded_action_chunk_raw = ep["actions"][t : t + args.chunk_size]
            else:
                pad = np.tile(ep["actions"][-1], (args.chunk_size - remaining, 1))
                recorded_action_chunk_raw = np.concatenate([ep["actions"][t:], pad], axis=0)

            recorded_action_chunk_student_norm = normalize_minmax(
                recorded_action_chunk_raw, student_stats["actions_min"], student_stats["actions_max"]
            )
            candidate_action_chunk_student_norm = sample_counterfactual_action(
                recorded_action_chunk_student_norm, rng, args.action_noise_std
            )
            # Unnormalize back to raw action units, then renormalize under the TEACHER's own stats
            # -- the two models were not necessarily trained with identical min/max ranges.
            candidate_action_chunk_raw = 0.5 * (candidate_action_chunk_student_norm + 1) * (
                student_stats["actions_max"] - student_stats["actions_min"]
            ) + student_stats["actions_min"]
            candidate_action_chunk_teacher_norm = normalize_minmax(
                candidate_action_chunk_raw, teacher_stats["actions_min"], teacher_stats["actions_max"]
            )

            obs = {
                "wrist_image": wrist_raw[t],
                "primary_image": agentview_raw[t],
                "proprio": ep["proprio"][t],
            }

            action_return_dict = get_action(
                cfg,
                model,
                teacher_stats,
                obs,
                ep["instruction"],
                seed=args.seed,
                num_denoising_steps_action=args.num_denoising_steps_action,
                generate_future_state_and_value_in_parallel=False,
            )

            candidate_tensor = torch.tensor(
                candidate_action_chunk_teacher_norm, dtype=action_return_dict["generated_latent"].dtype,
                device=action_return_dict["generated_latent"].device,
            ).unsqueeze(0)
            injected_latent = inject_counterfactual_action_into_latent(
                action_return_dict["generated_latent"],
                candidate_tensor,
                action_return_dict["latent_indices"]["action_latent_idx"],
            )

            future_state_return_dict = get_future_state_prediction(
                cfg,
                model,
                data_batch=action_return_dict["data_batch"],
                generated_latent_with_action=injected_latent,
                orig_clean_latent_frames=action_return_dict["orig_clean_latent_frames"],
                future_proprio_latent_idx=action_return_dict["latent_indices"]["future_proprio_latent_idx"],
                future_wrist_image_latent_idx=action_return_dict["latent_indices"]["future_wrist_image_latent_idx"],
                future_wrist_image2_latent_idx=action_return_dict["latent_indices"]["future_wrist_image2_latent_idx"],
                future_image_latent_idx=action_return_dict["latent_indices"]["future_image_latent_idx"],
                future_image2_latent_idx=action_return_dict["latent_indices"]["future_image2_latent_idx"],
                seed=args.seed,
                num_denoising_steps_future_state=args.num_denoising_steps_future_state,
            )
            value_return_dict = get_value_prediction(
                cfg,
                model,
                data_batch=action_return_dict["data_batch"],
                future_state_samples_list=future_state_return_dict["future_state_samples_list"],
                seed=args.seed,
                num_denoising_steps_value=args.num_denoising_steps_value,
            )

            future_agentview_224 = future_state_return_dict["future_image_predictions"]["future_image"]
            future_wrist_224 = future_state_return_dict["future_image_predictions"]["future_wrist_image"]
            future_agentview_96 = resize_images(future_agentview_224[None], args.image_size)[0]
            future_wrist_96 = resize_images(future_wrist_224[None], args.image_size)[0]

            agentview_96 = resize_images(agentview_raw[t][None], args.image_size)[0]
            wrist_96 = resize_images(wrist_raw[t][None], args.image_size)[0]
            proprio_student_norm = normalize_minmax(
                ep["proprio"][t], student_stats["proprio_min"], student_stats["proprio_max"]
            )

            synthetic_samples.append(
                {
                    "agentview_img": torch.from_numpy(agentview_96).permute(2, 0, 1).float() / 255.0,
                    "wrist_img": torch.from_numpy(wrist_96).permute(2, 0, 1).float() / 255.0,
                    "proprio": torch.from_numpy(proprio_student_norm).float(),
                    "action_chunk": torch.from_numpy(candidate_action_chunk_student_norm).float(),
                    "task_emb": student_task_emb,
                    "future_agentview_img": torch.from_numpy(future_agentview_96).permute(2, 0, 1).float() / 255.0,
                    "future_wrist_img": torch.from_numpy(future_wrist_96).permute(2, 0, 1).float() / 255.0,
                    "value": torch.tensor([value_return_dict["value_prediction"]], dtype=torch.float32),
                }
            )
            print(
                f"episode {ep_idx + 1}/{len(raw_episodes)}, t={t}: "
                f"value={value_return_dict['value_prediction']:.4f}, "
                f"{len(synthetic_samples)} synthetic sample(s) so far"
            )

    out_path = pathlib.Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(synthetic_samples, out_path)
    print(f"Saved {len(synthetic_samples)} synthetic sample(s) to {out_path}")


if __name__ == "__main__":
    main()
