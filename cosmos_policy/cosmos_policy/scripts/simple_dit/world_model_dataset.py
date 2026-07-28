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
Dataset for training `SimpleDiTWorldModel` (see model.py) from a mix of two kinds of samples:

  - "real" samples: the recorded action chunk actually executed in a demo, its real K-steps-ahead
    future frame, and its real Monte-Carlo return -- read directly from the same in-memory episode
    arrays `SimpleLiberoChunkDataset` already loads (via the shared `load_episodes` helper). No
    teacher model needed, since these are genuinely-executed on-trajectory transitions.
  - "synthetic" samples: perturbed/counterfactual action chunks that were never actually executed
    in any demo, with their future state and value imagined by the full 2B Cosmos Policy teacher --
    precomputed offline by `precompute_teacher_targets.py` (needs a GPU + the teacher checkpoint).

The synthetic half is what a value/dynamics model needs to generalize to the off-trajectory
candidate actions a best-of-N MPC search will propose at inference time: a model trained only on
real (state, recorded-action, real-future) transitions never sees such candidates during training
and has no reason to score them sensibly.
"""

import argparse
import json
import pathlib

import numpy as np
import torch
from torch.utils.data import Dataset

from cosmos_policy.datasets.dataset_common import compute_monte_carlo_returns, get_action_chunk_with_padding
from cosmos_policy.scripts.simple_dit_bc.dataset import augment_image, find_hdf5_files, load_episodes


class WorldModelDistillationDataset(Dataset):
    def __init__(
        self,
        hdf5_paths: list[str],
        t5_embeddings_path: str,
        synthetic_cache_path: str | None = None,
        chunk_size: int = 16,
        k_future: int = 16,
        gamma: float = 0.99,
        image_size: int = 96,
        stats: dict | None = None,
        augment: bool = True,
    ):
        self.chunk_size = chunk_size
        self.k_future = k_future
        self.gamma = gamma
        self.image_size = image_size
        self.augment = augment

        episodes, stats, instructions = load_episodes(hdf5_paths, t5_embeddings_path, image_size, stats)
        self.instructions = instructions
        self.stats = stats
        self.episodes = episodes
        # Flat (episode_idx, timestep) index across every timestep of every demo, same as
        # SimpleLiberoChunkDataset -- these are the "real" samples.
        self.real_index = [(ep_idx, t) for ep_idx, ep in enumerate(episodes) for t in range(len(ep["actions"]))]

        # Precomputed by precompute_teacher_targets.py using this SAME `stats` dict, so proprio/
        # action normalization is identical between the real and synthetic halves. Optional --
        # the dataset works real-only (e.g. for a quick CPU smoke test) if omitted.
        self.synthetic_samples = []
        if synthetic_cache_path is not None:
            self.synthetic_samples = torch.load(synthetic_cache_path, weights_only=False)
            print(f"Loaded {len(self.synthetic_samples)} synthetic (teacher-imagined) sample(s) from "
                  f"{synthetic_cache_path}")

    def __len__(self) -> int:
        return len(self.real_index) + len(self.synthetic_samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx < len(self.real_index):
            return self._get_real(idx)
        return self._get_synthetic(idx - len(self.real_index))

    def _get_real(self, idx: int) -> dict[str, torch.Tensor]:
        ep_idx, t = self.real_index[idx]
        ep = self.episodes[ep_idx]
        num_steps = len(ep["actions"])

        agentview_img = torch.from_numpy(ep["agentview"][t]).permute(2, 0, 1).float() / 255.0
        wrist_img = torch.from_numpy(ep["wrist"][t]).permute(2, 0, 1).float() / 255.0
        if self.augment:
            agentview_img = augment_image(agentview_img)
            wrist_img = augment_image(wrist_img)
        proprio = torch.from_numpy(ep["proprio"][t]).float()
        action_chunk = get_action_chunk_with_padding(ep["actions"], t, self.chunk_size, num_steps)
        action_chunk = torch.from_numpy(action_chunk).float()
        task_emb = torch.from_numpy(ep["task_emb"]).float()

        # Real future frame K steps ahead -- clipped to the episode's last frame, the same
        # repeat-last-step convention get_action_chunk_with_padding uses for actions past the end.
        future_t = min(t + self.k_future, num_steps - 1)
        future_agentview_img = torch.from_numpy(ep["agentview"][future_t]).permute(2, 0, 1).float() / 255.0
        future_wrist_img = torch.from_numpy(ep["wrist"][future_t]).permute(2, 0, 1).float() / 255.0

        # LIBERO demos under success_only/ (see dataset.py's module docstring) end in success by
        # construction, so terminal_reward=1.0 is valid for every episode here.
        returns = compute_monte_carlo_returns(num_steps, terminal_reward=1.0, gamma=self.gamma)
        value = (float(returns[t]) + 1.0) / 2.0  # [-1,1] -> [0,1], matching the teacher's value scale

        return {
            "agentview_img": agentview_img,
            "wrist_img": wrist_img,
            "proprio": proprio,
            "action_chunk": action_chunk,
            "task_emb": task_emb,
            "future_agentview_img": future_agentview_img,
            "future_wrist_img": future_wrist_img,
            "value": torch.tensor([value], dtype=torch.float32),
        }

    def _get_synthetic(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.synthetic_samples[idx]
        agentview_img = sample["agentview_img"].clone()
        wrist_img = sample["wrist_img"].clone()
        if self.augment:
            agentview_img = augment_image(agentview_img)
            wrist_img = augment_image(wrist_img)
        return {
            "agentview_img": agentview_img,
            "wrist_img": wrist_img,
            "proprio": sample["proprio"],
            "action_chunk": sample["action_chunk"],
            "task_emb": sample["task_emb"],
            "future_agentview_img": sample["future_agentview_img"],
            "future_wrist_img": sample["future_wrist_img"],
            "value": sample["value"],
        }

    def save_stats(self, path) -> None:
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({k: np.asarray(v).tolist() for k, v in self.stats.items()}, f, indent=2)

    def save_task_instructions(self, path) -> None:
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.instructions, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sanity-check WorldModelDistillationDataset shapes/dtypes/ranges.")
    parser.add_argument("data_dir", help="Directory containing one or more LIBERO demo *.hdf5 files.")
    parser.add_argument("t5_embeddings_path", help="Path to t5_embeddings.pkl.")
    parser.add_argument("--synthetic-cache-path", default=None, help="Output of precompute_teacher_targets.py.")
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--k-future", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=96)
    args = parser.parse_args()

    hdf5_paths = find_hdf5_files(args.data_dir)
    print(f"Found {len(hdf5_paths)} hdf5 file(s): {hdf5_paths}")

    dataset = WorldModelDistillationDataset(
        hdf5_paths,
        args.t5_embeddings_path,
        synthetic_cache_path=args.synthetic_cache_path,
        chunk_size=args.chunk_size,
        k_future=args.k_future,
        image_size=args.image_size,
    )
    print(f"Loaded {len(dataset.episodes)} demo(s), {len(dataset.real_index)} real sample(s), "
          f"{len(dataset.synthetic_samples)} synthetic sample(s), {len(dataset)} total, "
          f"{len(dataset.instructions)} unique task(s): {dataset.instructions}")

    sample = dataset[0]
    for key, value in sample.items():
        print(f"{key}: shape={tuple(value.shape)} dtype={value.dtype} min={value.min():.3f} max={value.max():.3f}")

    # Also check a sample from near the end of an episode, where future_t clipping kicks in.
    sample = dataset[len(dataset.episodes[0]["actions"]) - 1]
    print(f"last-timestep-of-episode-0 value={sample['value'].item():.3f} "
          f"(should be close to 1.0, since it's ~the terminal step)")
