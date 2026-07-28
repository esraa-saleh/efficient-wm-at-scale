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
Standalone LIBERO BC dataset: raw camera JPEG frames + `robot_states` proprio + action chunks.

Deliberately independent of `cosmos_policy.datasets.libero_dataset.LIBERODataset`, which is
tightly coupled to the video2world latent-sequence pipeline (return computation, future-proprio,
tokenizer chunk layout) that `simple_dit_bc` doesn't need. Reuses only the plain numpy/PIL/torch
helper functions LIBERODataset itself is built on, confirmed to have no CUDA/Hydra/model
dependencies:
    - `decode_jpeg_bytes_dataset` / `resize_images` / `calculate_dataset_statistics` /
      `rescale_episode_data` (cosmos_policy/datasets/dataset_utils.py)
    - `get_action_chunk_with_padding` (cosmos_policy/datasets/dataset_common.py)

Expected hdf5 layout per demo (confirmed against a real LIBERO-Cosmos-Policy file):
    data/demo_N/obs/agentview_rgb_jpeg   (T,) object array of per-timestep JPEG bytes
    data/demo_N/obs/eye_in_hand_rgb_jpeg (T,) object array of per-timestep JPEG bytes
    data/demo_N/robot_states             (T, 9) float64 -- gripper_qpos(2)+eef_pos(3)+eef_quat(4)
    data/demo_N/actions                  (T, 7) float64
"""

import argparse
import json
import pathlib
import pickle

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from cosmos_policy.datasets.dataset_common import get_action_chunk_with_padding
from cosmos_policy.datasets.dataset_utils import (
    calculate_dataset_statistics,
    decode_jpeg_bytes_dataset,
    resize_images,
    rescale_episode_data,
)


def find_hdf5_files(data_dir) -> list[str]:
    """Every *.hdf5 under `data_dir` (recursive) -- mirrors LIBERODataset's file discovery."""
    return sorted(str(p) for p in pathlib.Path(data_dir).rglob("*.hdf5"))


def instruction_from_filename(hdf5_path) -> str:
    """The same instruction string `LIBERODataset` derives from a demo filename
    (`cosmos_policy/datasets/libero_dataset.py:204-212`), so it matches `t5_embeddings.pkl`'s keys.

    E.g. "pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5" ->
    "pick up the alphabet soup and place it in the basket". Any token containing "SCENE" resets
    the accumulated instruction (drops e.g. a "KITCHEN_SCENE4_" filename prefix).
    """
    raw = pathlib.Path(hdf5_path).name
    words = raw[:-10].split("_")  # strip trailing "_demo.hdf5" (10 chars)
    command = ""
    for w in words:
        if "SCENE" in w:
            command = ""
            continue
        command = command + w + " "
    return command[:-1]


def augment_image(img: torch.Tensor, pad: int = 8) -> torch.Tensor:
    """Small random translation (pad + random-crop) and brightness/contrast jitter.

    A from-scratch CNN trained on ~45 demos/task with no augmentation overfits to the exact pixel
    statistics of the training frames -- statistics that never exactly recur during closed-loop
    rollout, where the object's on-screen position and lighting drift slightly from any single
    training frame. `img` is (C, H, W) float in [0, 1].
    """
    c, h, w = img.shape
    padded = torch.nn.functional.pad(img.unsqueeze(0), (pad, pad, pad, pad), mode="replicate").squeeze(0)
    top = int(torch.randint(0, 2 * pad + 1, (1,)))
    left = int(torch.randint(0, 2 * pad + 1, (1,)))
    img = padded[:, top : top + h, left : left + w]

    brightness = 1.0 + (torch.rand(1).item() - 0.5) * 0.4  # U(0.8, 1.2)
    contrast = 1.0 + (torch.rand(1).item() - 0.5) * 0.4  # U(0.8, 1.2)
    mean = img.mean()
    img = (img - mean) * contrast + mean
    img = img * brightness
    return img.clamp(0.0, 1.0)


def load_t5_embeddings(t5_embeddings_path) -> dict[str, np.ndarray]:
    """Load `t5_embeddings.pkl` (dict: instruction -> (1,512,1024) T5 tensor) and mean-pool each
    over the token dimension to a single (1024,) float32 vector -- simple_dit_bc conditions
    purely through AdaLN on one pooled vector, no cross-attention over the full token sequence
    like the full Cosmos Policy model.

    Sequences are zero-padded to 512 tokens, but a typical LIBERO instruction ("pick up the X and
    place it in the basket") is only ~10-15 real tokens -- confirmed directly (e.g. only 12/512
    rows nonzero for one instruction). A naive mean over all 512 positions dilutes the real,
    per-task-distinguishing signal by ~40x against a constant (same for every task) zero-padding
    contribution, which in practice showed up as several tasks whose objects most needed language
    to disambiguate from similarly-shaped distractors (e.g. ketchup/milk among other bottles)
    failing outright while visually-distinctive ones still worked. Masking out the zero rows
    before averaging fixes this.
    """
    with open(t5_embeddings_path, "rb") as f:
        raw = pickle.load(f)
    pooled = {}
    for instruction, emb in raw.items():
        emb = emb.float().squeeze(0)  # (512, 1024)
        real_token_mask = emb.abs().sum(dim=-1) != 0  # True for non-padded rows
        pooled[instruction] = emb[real_token_mask].mean(dim=0).numpy()
    return pooled


def load_episodes(
    hdf5_paths: list[str],
    t5_embeddings_path: str,
    image_size: int = 96,
    stats: dict | None = None,
) -> tuple[list[dict], dict, list[str]]:
    """Decode every demo into memory and min-max normalize proprio/actions to [-1, 1].

    Factored out of `SimpleLiberoChunkDataset.__init__` so `world_model_dataset.py`'s
    `WorldModelDistillationDataset` can reuse the exact same loading/normalization path (needed to
    read a real future frame + compute a real Monte-Carlo return per episode) without duplicating
    it or coupling the two dataset classes together.
    """
    t5_embeddings = load_t5_embeddings(t5_embeddings_path)

    episodes = []
    instructions = set()
    for hdf5_path in hdf5_paths:
        instruction = instruction_from_filename(hdf5_path)
        if instruction not in t5_embeddings:
            raise KeyError(
                f"No T5 embedding for instruction {instruction!r} (derived from {hdf5_path!r}) in "
                f"{t5_embeddings_path!r}. Available instructions: {sorted(t5_embeddings.keys())}"
            )
        instructions.add(instruction)
        task_emb = t5_embeddings[instruction]

        with h5py.File(hdf5_path, "r") as f:
            demo_keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
            for demo_key in demo_keys:
                demo = f[f"data/{demo_key}"]
                agentview = resize_images(decode_jpeg_bytes_dataset(demo["obs/agentview_rgb_jpeg"]), image_size)
                wrist = resize_images(decode_jpeg_bytes_dataset(demo["obs/eye_in_hand_rgb_jpeg"]), image_size)
                proprio = demo["robot_states"][:].astype(np.float32)
                actions = demo["actions"][:].astype(np.float32)
                episodes.append(
                    {
                        "agentview": agentview,
                        "wrist": wrist,
                        "proprio": proprio,
                        "actions": actions,
                        "task_emb": task_emb,
                        "instruction": instruction,
                    }
                )

    if not episodes:
        raise ValueError(f"No demos found in hdf5 files: {hdf5_paths}")
    instructions = sorted(instructions)

    # Compute stats fresh over exactly the demos passed in, rather than reusing any
    # pre-existing dataset_statistics.json elsewhere -- e.g. LIBERO-Cosmos-Policy's own
    # success_only/dataset_statistics.json was cached over the combined 4-suite set, not this
    # (likely single-task) subset, and its sibling dataset_statistics_post_norm.json is stale.
    if stats is None:
        stats_input = {i: {"actions": ep["actions"], "proprio": ep["proprio"]} for i, ep in enumerate(episodes)}
        stats = calculate_dataset_statistics(stats_input)

    # Normalize once up front (min-max to [-1, 1]) rather than per-__getitem__.
    for ep in episodes:
        ep["proprio"] = rescale_episode_data(ep, stats, "proprio")
        ep["actions"] = rescale_episode_data(ep, stats, "actions")

    return episodes, stats, instructions


class SimpleLiberoChunkDataset(Dataset):
    """Flattened per-timestep (agentview_img, wrist_img, proprio, action_chunk) samples.

    Loads and decodes every demo into memory in `__init__` -- fine at single-task scale (tens of
    demos, a few hundred timesteps each), and avoids repeated JPEG-decode + hdf5-file-handle
    overhead on every `__getitem__` call that a lazy-loading dataset would pay every epoch.
    """

    def __init__(
        self,
        hdf5_paths: list[str],
        t5_embeddings_path: str,
        chunk_size: int = 16,
        image_size: int = 96,
        stats: dict | None = None,
        augment: bool = True,
    ):
        self.chunk_size = chunk_size
        self.image_size = image_size
        self.augment = augment

        episodes, stats, instructions = load_episodes(hdf5_paths, t5_embeddings_path, image_size, stats)
        self.instructions = instructions
        self.stats = stats
        self.episodes = episodes
        # Flat (episode_idx, timestep) index across every timestep of every demo -- dense
        # indexing, same idea as LIBERODataset's _step_to_episode_map, simplified.
        self.index = [(ep_idx, t) for ep_idx, ep in enumerate(episodes) for t in range(len(ep["actions"]))]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ep_idx, t = self.index[idx]
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

        return {
            "agentview_img": agentview_img,
            "wrist_img": wrist_img,
            "proprio": proprio,
            "action_chunk": action_chunk,
            "task_emb": task_emb,
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
    parser = argparse.ArgumentParser(description="Sanity-check SimpleLiberoChunkDataset shapes/dtypes/ranges.")
    parser.add_argument("data_dir", help="Directory containing one or more LIBERO demo *.hdf5 files.")
    parser.add_argument("t5_embeddings_path", help="Path to t5_embeddings.pkl.")
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=96)
    args = parser.parse_args()

    hdf5_paths = find_hdf5_files(args.data_dir)
    print(f"Found {len(hdf5_paths)} hdf5 file(s): {hdf5_paths}")

    dataset = SimpleLiberoChunkDataset(
        hdf5_paths, args.t5_embeddings_path, chunk_size=args.chunk_size, image_size=args.image_size
    )
    print(f"Loaded {len(dataset.episodes)} demo(s), {len(dataset)} total timesteps, "
          f"{len(dataset.instructions)} unique task(s): {dataset.instructions}")

    sample = dataset[0]
    for key, value in sample.items():
        print(f"{key}: shape={tuple(value.shape)} dtype={value.dtype} min={value.min():.3f} max={value.max():.3f}")
