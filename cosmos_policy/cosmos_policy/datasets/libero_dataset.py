# -----------------------------------------------------------------------------
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This codebase constitutes NVIDIA proprietary technology and is strictly
# confidential. Any unauthorized reproduction, distribution, or disclosure
# of this code, in whole or in part, outside NVIDIA is strictly prohibited
# without prior written consent.
#
# For inquiries regarding the use of this code in other NVIDIA proprietary
# projects, please contact the Deep Imagination Research Team at
# dir@exchange.nvidia.com.
# -----------------------------------------------------------------------------

"""
LIBERO simulation benchmark task suites dataloader.

Run this command to print a few samples from the LIBERO dataset:
    python -m cosmos_policy.datasets.libero_dataset
"""

import json
import os
import pickle
import random
from collections import OrderedDict, defaultdict

import h5py
import imageio
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from cosmos_policy.datasets.dataset_common import (
    build_rollout_step_index_mapping,
    calculate_epoch_structure,
    compute_monte_carlo_returns,
    determine_sample_type,
    get_action_chunk_with_padding,
    load_or_compute_dataset_statistics,
    load_or_compute_post_normalization_statistics,
)
from cosmos_policy.datasets.dataset_utils import (
    calculate_dataset_statistics,
    decode_jpeg_bytes_dataset,
    decode_single_jpeg_frame,
    filter_hdf5_files_by_task_names,
    get_hdf5_files,
    preprocess_image,
    rescale_data,
    rescale_episode_data,
    resize_images,
)
from cosmos_policy.utils.utils import duplicate_array

# Set floating point precision to 3 decimal places and disable line wrapping
np.set_printoptions(precision=3, linewidth=np.inf)


class LIBERODataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        chunk_size: int = 8,
        final_image_size: int = 224,
        t5_text_embeddings_path: str = "",
        normalize_images=False,
        normalize_actions=True,
        normalize_proprio=True,
        use_image_aug: bool = True,
        use_stronger_image_aug: bool = True,
        use_wrist_images: bool = True,
        use_third_person_images: bool = True,
        use_proprio: bool = True,
        num_duplicates_per_image: int = 4,
        rollout_data_dir: str = "",
        demonstration_sampling_prob: float = 0.5,
        success_rollout_sampling_prob: float = 0.5,
        treat_success_rollouts_as_demos: bool = False,
        return_value_function_returns: bool = True,
        gamma: float = 0.99,
        task_names: list | None = None,
        teacher_on_demos_dir: str = "",
        dataset_stats_override_path: str = "",
        teacher_on_demos_num_samples: int = 1,
    ):
        """
        Initialize LIBERO dataset for training.

        Args:
            data_dir (str): Path to directory containing LIBERO task suite HDF5 files
            chunk_size (int): Action chunk size
            final_image_size (int): Target size for resized images (square), defaults to 224
            t5_text_embeddings_path (str): Path to precomputed T5 text embeddings dictionary (key: instruction, val: embedding)
            num_images_per_sample (int): Number of images to return per sample
            normalize_images (bool): Whether to normalize the images and return as torch.float32
            normalize_actions (bool): Whether to normalize the actions
            normalize_proprio (bool): Whether to normalize the proprioceptive state
            use_image_aug (bool): Whether to apply image augmentations
            use_stronger_image_aug (bool): Whether to apply stronger image augmentations
            use_wrist_images (bool): If True, loads wrist-mounted camera images
            use_third_person_images (bool): If True, loads third-person images
            use_proprio (bool): If True, adds proprio to image observations
            num_duplicates_per_image (int): Number of times to duplicate each image (so that each type of image fills 1 latent frame when encoded with the tokenizer)
            rollout_data_dir (str): Path to directory containing rollout data (if provided, will load rollout data in addition to base dataset)
            demonstration_sampling_prob (float): Probability of sampling from demonstration data instead of rollout data
            success_rollout_sampling_prob (float): Probability of sampling from success rollout data instead of failure rollout data
            treat_success_rollouts_as_demos (bool): If True, copy successful rollout episodes into demonstration dataset (self.data)
            return_value_function_returns (bool): If True, returns value function returns for rollout episodes
            gamma (float): Discount factor for value function returns
            task_names (list[str] | None): If given, restricts both `data_dir` and `rollout_data_dir`
                to files whose name contains one of these (case-insensitive substring match, e.g.
                ["ketchup"]) -- lets you select a subset of a suite's tasks (down to just one)
                without copying/symlinking a separate directory: point data_dir at the real suite
                dir as usual (dataset_statistics.json still resolves from there, so normalization
                stays consistent with the full suite) and this filters which files it actually
                reads. None (default) loads every task in data_dir, unchanged from before.
            teacher_on_demos_dir (str): If non-empty, a sidecar dataset root (built by
                kd/build_teacher_on_demos_dataset.py) mirroring `data_dir`'s layout. For every demo
                sample, the action chunk, future proprio, value_function_return, and future
                primary/wrist images are overridden with the teacher's predictions for that
                (episode, timestep) instead of the human demo's -- everything else in __getitem__
                (current obs, all blanks, latent indices, masks, t5, augmentation) is unchanged.
                "" (default) is a complete no-op: byte-identical to the pre-existing behavior. Only
                consulted for `sample_type == "demo"` (this path is demo-only when set). See
                kd/TEACHER_ON_DEMOS_PLAN.md.
            dataset_stats_override_path (str): If non-empty, a local path to a
                dataset_statistics.json to use INSTEAD OF computing/loading one from `data_dir` --
                e.g. the teacher checkpoint's own stats file, when this dataset's action/proprio
                values are meant to match a normalization convention other than the local suite's
                (teacher_on_demos: the teacher generates in its own convention, so training the
                student against it end-to-end -- current proprio, action, future proprio all in the
                teacher's own scale -- avoids the per-field renormalization that a mismatched
                convention otherwise requires; see kd/build_teacher_on_demos_dataset.py and
                experiment_journal.txt 2026-09-04). Never writes back to `data_dir`; the
                post-normalization diagnostic stats (unused elsewhere) are skipped in this mode so
                a mismatched file never gets cached at data_dir/dataset_statistics_post_norm.json.
                "" (default) is a complete no-op: byte-identical to the pre-existing behavior.
            teacher_on_demos_num_samples (int): How many independent teacher samples K the
                teacher_on_demos_dir sidecar stores per (episode, timestep) (see
                kd/build_teacher_on_demos_dataset.py's "MULTIPLE TEACHER SAMPLES" docstring
                section). 1 (default): every state is visited once per epoch (as always) and, if
                the sidecar happens to store more than one sample anyway, __getitem__ picks one at
                random each visit -- byte-identical to the pre-existing behavior when the sidecar
                itself has K=1. >1: FULL-COVERAGE mode -- the epoch is expanded from `num_steps` to
                `num_steps * K` entries so every one of the K generated samples for every state is
                visited exactly once per epoch (deterministically, via floor-division on the global
                index), instead of a random subset determined by how many epochs happen to run.
                Must match the sidecar's actual K (asserted at __getitem__ time). Only meaningful
                together with teacher_on_demos_dir; ignored otherwise.
        """
        self.data_dir = data_dir
        self.chunk_size = chunk_size
        self.final_image_size = final_image_size
        self.t5_text_embeddings_path = t5_text_embeddings_path
        self.normalize_images = normalize_images
        self.normalize_actions = normalize_actions
        self.normalize_proprio = normalize_proprio
        self.use_image_aug = use_image_aug
        self.use_stronger_image_aug = use_stronger_image_aug
        self.use_wrist_images = use_wrist_images
        self.use_third_person_images = use_third_person_images
        self.use_proprio = use_proprio
        self.num_duplicates_per_image = num_duplicates_per_image
        self.rollout_data_dir = rollout_data_dir
        self.demonstration_sampling_prob = demonstration_sampling_prob
        self.success_rollout_sampling_prob = success_rollout_sampling_prob
        self.treat_success_rollouts_as_demos = treat_success_rollouts_as_demos
        self.return_value_function_returns = return_value_function_returns
        self.gamma = gamma
        self.task_names = task_names
        self.teacher_on_demos_dir = teacher_on_demos_dir
        self.dataset_stats_override_path = dataset_stats_override_path
        self.teacher_on_demos_num_samples = teacher_on_demos_num_samples
        # episode_idx -> dict of this episode's teacher sidecar arrays (small LRU, images dominate).
        self._teacher_on_demos_cache = OrderedDict()
        self._teacher_on_demos_cache_size = 8

        assert self.use_wrist_images or self.use_third_person_images, (
            "Must use at least one of wrist images or third-person images!"
        )

        # Get all HDF5 files in data directory
        hdf5_files = get_hdf5_files(data_dir)
        hdf5_files = filter_hdf5_files_by_task_names(hdf5_files, self.task_names)

        # In debug mode, only load the first demo
        if os.environ.get("DEBUGGING", "False").lower() == "true":
            hdf5_files = hdf5_files[:1]

        # Placeholder list for rollout files (may be empty)
        rollout_hdf5_files = []
        if self.rollout_data_dir:
            assert os.path.exists(self.rollout_data_dir), (
                f"Error: Rollout data directory '{self.rollout_data_dir}' does not exist."
            )
            rollout_hdf5_files = get_hdf5_files(self.rollout_data_dir)
            rollout_hdf5_files = filter_hdf5_files_by_task_names(rollout_hdf5_files, self.task_names)

        # Load all episodes' metadata (+ small per-step arrays: actions/proprio/returns) into RAM.
        # Images/wrist_images are NOT eagerly loaded here -- they dominate memory (T x H x W x 3
        # uint8 per episode) and are instead lazily read from the source HDF5 file per-episode in
        # `_load_demo_episode_images`, called from `__getitem__`. This mirrors the lazy-loading
        # already used for rollout data below (see `_load_rollout_episode_data`) and is what keeps
        # multi-suite dataset construction from scaling host RAM with total dataset size.
        # Save dataset in this structure:
        # data = {
        #   episode index: {
        #      file_path=source HDF5 file, demo_key=key of this episode within it,
        #      image_key/image_is_jpeg, wrist_key/wrist_is_jpeg=how to lazily re-read images,
        #      proprio=proprio states,
        #      actions=actions,
        #      command=language instruction,
        #      num_steps=number of steps in episode,
        #      suite=task suite name,
        #      returns=observed returns,
        #   }
        # }
        self.data = {}
        self.rollout_episode_metadata = {}  # For lazy loading: episode_idx -> metadata dict
        self.num_episodes = 0
        self.num_steps = 0
        self.rollout_num_episodes = 0
        self.rollout_num_steps = 0
        self.unique_commands = set()

        # Global step mapping from task suite name to list[global_step_idx]
        # Populated later in `_build_step_index_mapping()`
        self._suite_to_step_indices = {}
        if self.demonstration_sampling_prob > 0:  # Only load demos if they are used
            for file in tqdm(hdf5_files):
                with h5py.File(file, "r") as f:
                    # Get demo keys and sort them numerically ("demo_0", "demo_1", ...)
                    demo_keys_list = list(f["data"].keys())
                    sorted_demo_keys = sorted(demo_keys_list, key=lambda x: int(x.split("_")[1]))

                    for demo_key in tqdm(sorted_demo_keys):
                        # Determine whether the dataset stores raw RGB frames or JPEG bytes, and
                        # record enough to lazily re-read the pixel data later -- read `.shape`
                        # only (cheap, metadata-only in h5py) rather than slicing `[:]`, which
                        # would pull the full (T, H, W, 3) uint8 array into RAM.
                        obs_group = f[f"data/{demo_key}/obs"]
                        # Agent-view (third-person) images
                        if "agentview_rgb" in obs_group:
                            image_key, image_is_jpeg = "agentview_rgb", False
                        elif "agentview_rgb_jpeg" in obs_group:
                            image_key, image_is_jpeg = "agentview_rgb_jpeg", True
                        else:
                            raise KeyError("Neither 'agentview_rgb' nor 'agentview_rgb_jpeg' found in HDF5 file.")
                        # Wrist-mounted camera images
                        if "eye_in_hand_rgb" in obs_group:
                            wrist_key, wrist_is_jpeg = "eye_in_hand_rgb", False
                        elif "eye_in_hand_rgb_jpeg" in obs_group:
                            wrist_key, wrist_is_jpeg = "eye_in_hand_rgb_jpeg", True
                        else:
                            raise KeyError("Neither 'eye_in_hand_rgb' nor 'eye_in_hand_rgb_jpeg' found in HDF5 file.")
                        num_steps = obs_group[image_key].shape[0]
                        # Actions
                        actions = f[f"data/{demo_key}/actions"][:].astype(
                            np.float32
                        )  # (episode_len, action_dim=7), float32
                        # Proprio states
                        proprio = f[f"data/{demo_key}/robot_states"][:].astype(
                            np.float32
                        )  # (episode_len, proprio_dim=9), float32
                        # Compute language instruction
                        raw_file_string = os.path.basename(file).split("/")[-1]
                        words = raw_file_string[:-10].split("_")
                        command = ""
                        for w in words:
                            if "SCENE" in w:
                                command = ""
                                continue
                            command = command + w + " "
                        command = command[:-1]
                        self.unique_commands.add(command)
                        # Add value function returns if applicable
                        if self.return_value_function_returns:
                            returns = compute_monte_carlo_returns(num_steps, terminal_reward=1.0, gamma=self.gamma)
                        # Add entry to dataset dict (images/wrist_images loaded lazily, see above)
                        self.data[self.num_episodes] = dict(
                            file_path=file,
                            demo_key=demo_key,
                            image_key=image_key,
                            image_is_jpeg=image_is_jpeg,
                            wrist_key=wrist_key,
                            wrist_is_jpeg=wrist_is_jpeg,
                            proprio=proprio,
                            actions=actions,
                            command=command,
                            num_steps=num_steps,
                            suite=os.path.relpath(file, self.data_dir).split(os.sep)[
                                0
                            ],  # Task suite folder name (e.g. libero_spatial_no_noops_rerendered)
                            returns=returns.copy() if self.return_value_function_returns else None,
                        )
                        # Update number of episodes
                        self.num_episodes += 1
                        # Update number of steps
                        self.num_steps += num_steps

        # Build mapping from global step index to episode step
        self._build_step_index_mapping()

        self.chunk_size = chunk_size

        # If applicable, load precomputed T5 text embeddings
        if t5_text_embeddings_path != "":
            with open(t5_text_embeddings_path, "rb") as file:
                self.t5_text_embeddings = pickle.load(file)

        # Calculate dataset statistics if the stats file doesn't exist -- unless a different
        # convention entirely was requested (dataset_stats_override_path), e.g. the teacher
        # checkpoint's own, in which case load that instead and never touch data_dir.
        if self.dataset_stats_override_path:
            with open(self.dataset_stats_override_path, "r") as f:
                self.dataset_stats = {k: np.array(v) for k, v in json.load(f).items()}
        else:
            self.dataset_stats = load_or_compute_dataset_statistics(
                data_dir=self.data_dir,
                data=self.data,
                calculate_dataset_statistics_func=calculate_dataset_statistics,
            )

        # Normalize actions and/or proprio
        if self.normalize_actions or self.normalize_proprio:
            if self.normalize_actions:
                self.data = rescale_data(self.data, self.dataset_stats, "actions")
            if self.normalize_proprio:
                self.data = rescale_data(self.data, self.dataset_stats, "proprio")

            if not self.dataset_stats_override_path:
                # Calculate post-normalization action statistics (diagnostic only, unused
                # elsewhere -- skipped under an override so a mismatched-convention file never
                # gets cached at data_dir/dataset_statistics_post_norm.json).
                self.dataset_stats_post_norm = load_or_compute_post_normalization_statistics(
                    data_dir=self.data_dir,
                    data=self.data,
                    calculate_dataset_statistics_func=calculate_dataset_statistics,
                )

        # ====================================================================
        # If applicable, load rollout dataset metadata (lazy loading)
        # ====================================================================
        if len(rollout_hdf5_files) > 0:
            for file in tqdm(rollout_hdf5_files, desc="Loading rollout metadata"):
                with h5py.File(file, "r") as f:
                    # Determine storage format of images (raw vs. JPEG)
                    if "primary_images" in f:
                        is_jpeg = False
                        num_steps = len(f["primary_images"])
                    elif "primary_images_jpeg" in f:
                        is_jpeg = True
                        num_steps = len(f["primary_images_jpeg"])
                    else:
                        raise KeyError(f"No primary/wrist images found in rollout file: {file}")

                    # Get task description
                    command = f.attrs.get("task_description", "")
                    self.unique_commands.add(command)
                    # Get success flag
                    success = bool(f.attrs.get("success", False))

                    # Store metadata for lazy loading
                    self.rollout_episode_metadata[self.rollout_num_episodes] = dict(
                        file_path=file,
                        command=command,
                        num_steps=num_steps,
                        success=success,
                        is_jpeg=is_jpeg,  # Flag to indicate JPEG compression
                    )
                    # Add value function returns if applicable
                    if self.return_value_function_returns:
                        # Get success label
                        success = bool(f.attrs.get("success"))
                        terminal_reward = 1.0 if success else 0.0
                        returns = compute_monte_carlo_returns(
                            num_steps, terminal_reward=terminal_reward, gamma=self.gamma
                        )
                        self.rollout_episode_metadata[self.rollout_num_episodes]["returns"] = returns.copy()

                    self.rollout_num_episodes += 1
                    self.rollout_num_steps += num_steps

            # If applicable, copy successful rollout episodes into demonstration dataset
            if self.treat_success_rollouts_as_demos:
                for ep_idx, ep_meta in self.rollout_episode_metadata.items():
                    if not ep_meta.get("success", False):
                        continue

                    # Lazy load rollout episode data
                    episode_data = self._load_rollout_episode_data(ep_meta)

                    # Decode to raw uint8 arrays for demos if source was JPEG bytes
                    if episode_data.get("is_jpeg", False):
                        images = np.stack([decode_single_jpeg_frame(b) for b in episode_data["images"]], axis=0).astype(
                            np.uint8
                        )
                        wrist_images = np.stack(
                            [decode_single_jpeg_frame(b) for b in episode_data["wrist_images"]], axis=0
                        ).astype(np.uint8)
                    else:
                        images = episode_data["images"]
                        wrist_images = episode_data["wrist_images"]

                    actions = episode_data["actions"]
                    proprio = episode_data["proprio"]

                    # Determine suite name from rollout directory (for balanced sampling bookkeeping)
                    if "suite=libero_spatial" in ep_meta["file_path"]:
                        suite_name = "libero_spatial"
                    elif "suite=libero_object" in ep_meta["file_path"]:
                        suite_name = "libero_object"
                    elif "suite=libero_goal" in ep_meta["file_path"]:
                        suite_name = "libero_goal"
                    elif "suite=libero_10" in ep_meta["file_path"]:
                        suite_name = "libero_10"
                    else:
                        raise ValueError(
                            f"Could not determine suite name from rollout file path: {ep_meta['file_path']}"
                        )

                    # Use precomputed returns
                    returns = ep_meta.get("returns")
                    if returns is not None:
                        returns = returns.copy()

                    # Insert into demonstration dataset
                    self.data[self.num_episodes] = dict(
                        images=images,
                        wrist_images=wrist_images,
                        proprio=proprio,
                        actions=actions,
                        command=ep_meta.get("command"),
                        num_steps=ep_meta.get("num_steps"),
                        suite=suite_name,
                        returns=returns,
                    )
                    self.unique_commands.add(ep_meta.get("command"))
                    self.num_episodes += 1
                    self.num_steps += ep_meta.get("num_steps")

                # Rebuild step index mapping to include newly added demos
                self._build_step_index_mapping()

            # Build mapping from global rollout step → (episode, rel_idx)
            self._build_rollout_step_index_mapping()

        # Calculate epoch structure and counts
        self._calculate_epoch_structure()

    def _calculate_epoch_structure(self):
        """Calculate epoch layout with proper scaling: demos, success rollouts, failure rollouts."""
        # Initialize rollout step counts if not available
        if not hasattr(self, "_rollout_success_total_steps"):
            self._rollout_success_total_steps = 0
        if not hasattr(self, "_rollout_failure_total_steps"):
            self._rollout_failure_total_steps = 0
        if not hasattr(self, "_rollout_total_steps"):
            self._rollout_total_steps = self._rollout_success_total_steps + self._rollout_failure_total_steps

        demo_base_count = self.num_steps
        if self.teacher_on_demos_dir and self.teacher_on_demos_num_samples > 1:
            # Full-coverage mode: expand the epoch so every one of the K generated samples for
            # every state is visited exactly once per epoch (see teacher_on_demos_num_samples'
            # own docstring). self.num_steps itself stays the raw per-state count -- __getitem__
            # recovers it via floor-division on the (now K-times-larger) global index.
            demo_base_count = self.num_steps * self.teacher_on_demos_num_samples

        result = calculate_epoch_structure(
            num_steps=demo_base_count,
            rollout_success_total_steps=self._rollout_success_total_steps,
            rollout_failure_total_steps=self._rollout_failure_total_steps,
            demonstration_sampling_prob=self.demonstration_sampling_prob,
            success_rollout_sampling_prob=self.success_rollout_sampling_prob,
        )
        self.adjusted_demo_count = result["adjusted_demo_count"]
        self.adjusted_success_rollout_count = result["adjusted_success_rollout_count"]
        self.adjusted_failure_rollout_count = result["adjusted_failure_rollout_count"]
        self.epoch_length = result["epoch_length"]

    def _build_step_index_mapping(self):
        """Build a mapping from global step index to (episode index, relative index within episode)."""
        self._step_to_episode_map = {}
        self._total_steps = 0

        # Reset suite mapping if it already exists
        self._suite_to_step_indices = defaultdict(list)

        for episode_idx, episode_data in self.data.items():
            num_steps = episode_data["num_steps"]
            for i in range(num_steps):
                self._step_to_episode_map[self._total_steps] = (episode_idx, i)
                self._suite_to_step_indices[episode_data["suite"]].append(self._total_steps)
                self._total_steps += 1

        # Additional bookkeeping for balanced sampling
        self._suites = list(self._suite_to_step_indices.keys())
        if len(self._suites) > 0:
            self._max_suite_len = max(len(v) for v in self._suite_to_step_indices.values())

    def _build_rollout_step_index_mapping(self):
        """Build mapping for rollout dataset with separate tracking for successful/failure episodes."""
        result = build_rollout_step_index_mapping({}, self.rollout_episode_metadata)
        self._rollout_success_step_to_episode_map = result["_rollout_success_step_to_episode_map"]
        self._rollout_failure_step_to_episode_map = result["_rollout_failure_step_to_episode_map"]
        self._rollout_success_total_steps = result["_rollout_success_total_steps"]
        self._rollout_failure_total_steps = result["_rollout_failure_total_steps"]
        self._rollout_total_steps = result["_rollout_total_steps"]

    def _load_rollout_episode_data(self, episode_metadata):
        """
        Load rollout episode data from HDF5 file using metadata.

        Args:
            episode_metadata (dict): Episode metadata containing file_path, success, etc.

        Returns:
            dict: Episode data dictionary with loaded arrays
        """
        file_path = episode_metadata["file_path"]

        with h5py.File(file_path, "r") as f:
            # Load images based on storage format
            if episode_metadata["is_jpeg"]:
                # Store raw JPEG bytes
                images = f["primary_images_jpeg"][:]
                wrist_images = f["wrist_images_jpeg"][:]
            else:
                images = f["primary_images"][:]
                wrist_images = f["wrist_images"][:]

            # Load actions and proprio
            actions = f["actions"][:].astype(np.float32)
            proprio = f["proprio"][:].astype(np.float32)

            # Apply normalization if needed
            if self.normalize_actions:
                actions = rescale_episode_data({"actions": actions}, self.dataset_stats, "actions")
            if self.normalize_proprio:
                proprio = rescale_episode_data(
                    {"proprio": proprio},
                    self.dataset_stats,
                    "proprio",
                )

            # Create episode data dictionary
            episode_data = dict(
                images=images,
                wrist_images=wrist_images,
                proprio=proprio,
                actions=actions,
                command=episode_metadata["command"],
                num_steps=episode_metadata["num_steps"],
                success=episode_metadata["success"],
                is_jpeg=episode_metadata["is_jpeg"],
            )

            return episode_data

    def _load_demo_episode_images(self, episode_metadata):
        """
        Lazily load a demo episode's images/wrist_images from its source HDF5 file, using the
        file_path/demo_key/image_key recorded in `self.data` at construction time. Mirrors
        `_load_rollout_episode_data`'s lazy loading, kept separate because demo episodes already
        have their (cheap) actions/proprio/returns loaded eagerly in `self.data`.

        Args:
            episode_metadata (dict): Entry from `self.data` (has file_path, demo_key, image_key, etc.)

        Returns:
            tuple[np.ndarray, np.ndarray]: (images, wrist_images), each (T, H, W, 3) uint8
        """
        with h5py.File(episode_metadata["file_path"], "r") as f:
            obs_group = f[f"data/{episode_metadata['demo_key']}/obs"]
            if episode_metadata["image_is_jpeg"]:
                images = decode_jpeg_bytes_dataset(obs_group[episode_metadata["image_key"]])
            else:
                images = obs_group[episode_metadata["image_key"]][:]
            if episode_metadata["wrist_is_jpeg"]:
                wrist_images = decode_jpeg_bytes_dataset(obs_group[episode_metadata["wrist_key"]])
            else:
                wrist_images = obs_group[episode_metadata["wrist_key"]][:]

        return images, wrist_images

    def _teacher_on_demos_sidecar_path(self, episode_metadata):
        """Mirror of kd/build_teacher_on_demos_dataset.sidecar_path_for: the source file's path
        relative to `data_dir`, under `teacher_on_demos_dir`, with `.teacher_on_demos` inserted
        before the extension. Keyed on (file_path, data_dir) so it is unaffected by `task_names`
        filtering (which only re-numbers episode_idx)."""
        rel = os.path.relpath(episode_metadata["file_path"], self.data_dir)
        base, ext = os.path.splitext(rel)
        return os.path.join(self.teacher_on_demos_dir, base + ".teacher_on_demos" + ext)

    def _load_teacher_on_demos_sidecar(self, episode_idx):
        """Lazily load (and LRU-cache) one demo episode's teacher-target arrays from its sidecar
        HDF5. Returns a dict with a leading (T, K) shape on every field -- K independent teacher
        samples per timestep (different initial diffusion noise, same conditioning; see
        kd/build_teacher_on_demos_dataset.py's module docstring), kept SEPARATE rather than
        averaged so training sees the teacher's genuine multimodality: action_chunks (T,K,chunk,7)
        f32, future_proprio (T,K,9) f32, value (T,K) f32 in [-1,1], future_image_jpeg /
        future_wrist_image_jpeg (T,K) object arrays of raw JPEG bytes. __getitem__ picks one
        k uniformly at random per call (see `teacher_on_demos_k` below) -- the SAME k across all
        four fields for a given call, since they're one coherent generation. Raises
        FileNotFoundError naming the missing sidecar if the build didn't cover this episode (e.g.
        a per-task build without a matching task_names on this run)."""
        cached = self._teacher_on_demos_cache.get(episode_idx)
        if cached is not None:
            self._teacher_on_demos_cache.move_to_end(episode_idx)
            return cached

        episode_metadata = self.data[episode_idx]
        if "file_path" not in episode_metadata:
            # treat_success_rollouts_as_demos entries carry no source file -- they never occur on
            # the teacher_on_demos path (rollout_data_dir is unset there), so this is unreachable,
            # but fail loudly rather than silently serve demo targets if it ever changes.
            raise RuntimeError("teacher_on_demos_dir set but episode has no source file_path to key a sidecar on")
        sidecar_path = self._teacher_on_demos_sidecar_path(episode_metadata)
        if not os.path.exists(sidecar_path):
            raise FileNotFoundError(
                f"teacher_on_demos sidecar not found: {sidecar_path} (for episode {episode_metadata['demo_key']} "
                f"of {episode_metadata['file_path']}). Build it with "
                f"kd/build_teacher_on_demos_dataset.py, or restrict this run's task_names to the tasks "
                f"that were built."
            )
        demo_key = episode_metadata["demo_key"]
        with h5py.File(sidecar_path, "r") as f:
            if demo_key not in f:
                raise KeyError(f"teacher_on_demos sidecar {sidecar_path} has no group '{demo_key}'")
            g = f[demo_key]
            action_chunks = g["action_chunks"][:].astype(np.float32)
            future_proprio = g["future_proprio"][:].astype(np.float32)
            value = g["value"][:].astype(np.float32)
            future_image_jpeg = g["future_image_jpeg"][:]
            future_wrist_image_jpeg = g["future_wrist_image_jpeg"][:]
        # Pre-K sidecars have no sample axis (action_chunks (T,chunk,7), value (T,), jpeg fields
        # (T,)). Insert a size-1 K axis so every consumer below is uniform -- __getitem__'s
        # random/enum k then always resolves to 0 for these.
        if action_chunks.ndim == 3:
            action_chunks = action_chunks[:, None]  # (T,1,chunk,7)
            future_proprio = future_proprio[:, None]  # (T,1,9)
            value = value[:, None]  # (T,1)
            future_image_jpeg = future_image_jpeg[:, None]  # (T,1) object
            future_wrist_image_jpeg = future_wrist_image_jpeg[:, None]
        sidecar = dict(
            action_chunks=action_chunks,  # (T,K,chunk,7)
            future_proprio=future_proprio,  # (T,K,9)
            value=value,  # (T,K)
            # (T,K) object arrays -- h5py hands back each vlen cell as its own uint8 ndarray.
            future_image_jpeg=future_image_jpeg,
            future_wrist_image_jpeg=future_wrist_image_jpeg,
        )

        self._teacher_on_demos_cache[episode_idx] = sidecar
        self._teacher_on_demos_cache.move_to_end(episode_idx)
        while len(self._teacher_on_demos_cache) > self._teacher_on_demos_cache_size:
            self._teacher_on_demos_cache.popitem(last=False)
        return sidecar

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        # Return pre-calculated epoch length (which already accounts for suite balancing if enabled)
        return self.epoch_length

    def __getitem__(self, idx):
        """
        Fetches images and action chunk sample by index.
        Returns action chunk rather than just single-step action.
        If the action chunk retrieval would go out of bounds, the last action is repeated however
        many times needed to fill up the chunk.

        Args:
            idx: Integer index to retrieve sample

        Returns:
            dict: Data sample: {
                video=images,
                actions=action chunk,
                t5_text_embeddings=text embedding,
                t5_text_mask=text embedding mask,
                fps=frames per second,
                padding_mask=padding mask,
                num_frames=number of frames per sequence,
                image_size=image size,
                proprio=proprio state,
                __key__=unique sample identifier,
            }
        """

        # Determine which dataset to sample from based on index ranges
        # Layout of indices within dataset: [demos] [success rollouts] [failure rollouts]
        sample_type = determine_sample_type(idx, self.adjusted_demo_count, self.adjusted_success_rollout_count)

        rollout_data_mask = 1 if sample_type != "demo" else 0
        rollout_data_success_mask = 1 if sample_type == "success_rollout" else 0

        if sample_type == "demo":
            # Get demonstration sample
            global_step_idx = idx % self.num_steps
            # Full-coverage teacher_on_demos mode (see teacher_on_demos_num_samples' docstring):
            # idx ranges over [0, num_steps * K), so floor-division recovers which of the K
            # samples this particular idx enumerates -- deterministic, not the random draw below.
            teacher_on_demos_k_enum = (
                idx // self.num_steps
                if (self.teacher_on_demos_dir and self.teacher_on_demos_num_samples > 1)
                else None
            )
            # Using global step index, get episode index and relative step index within that episode
            episode_idx, relative_step_idx = self._step_to_episode_map[global_step_idx]
            episode_metadata = None
            episode_data = self.data[episode_idx]
            if "images" not in episode_data:
                # Lazily load this episode's images/wrist_images from disk (not kept in
                # self.data). Entries added via treat_success_rollouts_as_demos already carry
                # real images/wrist_images arrays and skip this.
                images, wrist_images = self._load_demo_episode_images(episode_data)
                episode_data = {**episode_data, "images": images, "wrist_images": wrist_images}
            global_rollout_idx = -1  # Not applicable for demonstration data
        elif sample_type == "success_rollout":
            # Success rollout sample
            success_idx = idx - self.adjusted_demo_count  # Index within success rollouts section
            global_rollout_idx = success_idx % self._rollout_success_total_steps
            episode_idx, relative_step_idx = self._rollout_success_step_to_episode_map[global_rollout_idx]
            # Lazy load from HDF5 file
            episode_metadata = self.rollout_episode_metadata[episode_idx]
            episode_data = self._load_rollout_episode_data(episode_metadata)
        else:
            # Failure rollout sample
            failure_idx = (
                idx - self.adjusted_demo_count - self.adjusted_success_rollout_count
            )  # Index within failure rollouts section
            global_rollout_idx = failure_idx % self._rollout_failure_total_steps
            episode_idx, relative_step_idx = self._rollout_failure_step_to_episode_map[global_rollout_idx]
            # Lazy load from HDF5 file
            episode_metadata = self.rollout_episode_metadata[episode_idx]
            episode_data = self._load_rollout_episode_data(episode_metadata)

        # If returning value function samples, randomly choose whether this sample is for
        # world model training or value function training
        is_world_model_sample = False
        is_value_function_sample = False
        if sample_type != "demo":
            if self.return_value_function_returns:
                p_world_model = 0.5
                if random.random() < p_world_model:
                    is_world_model_sample = True
                    is_value_function_sample = False
                else:
                    is_world_model_sample = False
                    is_value_function_sample = True
            else:
                is_world_model_sample = True
                is_value_function_sample = False

        # Calculate future frame index if needed
        future_frame_idx = relative_step_idx + self.chunk_size
        max_possible_idx = episode_data["num_steps"] - 1
        if future_frame_idx > max_possible_idx:
            future_frame_idx = max_possible_idx

        # Handle JPEG decompression for rollout data if needed
        decompressed_images = {}
        decompressed_wrist_images = {}
        frames_needed = {relative_step_idx, future_frame_idx}
        for frame_idx in frames_needed:
            if sample_type != "demo" and episode_data["is_jpeg"]:
                # Decompress JPEG frames
                decompressed_images[frame_idx] = decode_single_jpeg_frame(episode_data["images"][frame_idx])
                decompressed_wrist_images[frame_idx] = decode_single_jpeg_frame(episode_data["wrist_images"][frame_idx])
            else:
                # Use images as-is
                decompressed_images[frame_idx] = episode_data["images"][frame_idx]
                decompressed_wrist_images[frame_idx] = episode_data["wrist_images"][frame_idx]

        # teacher_on_demos: swap the four joint denoising targets (action chunk, future proprio,
        # value, future primary/wrist images) for the teacher's prediction for this (episode,
        # timestep). Everything else -- current obs, blanks, latent indices, masks, t5 -- is
        # untouched. Only applies to demo samples (this path is demo-only when the kwarg is set).
        use_teacher_on_demos = bool(self.teacher_on_demos_dir) and sample_type == "demo"
        teacher_on_demos = self._load_teacher_on_demos_sidecar(episode_idx) if use_teacher_on_demos else None
        # One of the K independent teacher samples for this state -- the SAME k across every field
        # below, since one k is one coherent teacher generation. Full-coverage mode
        # (teacher_on_demos_num_samples > 1): teacher_on_demos_k_enum, set above, deterministically
        # enumerates every k exactly once per epoch. Otherwise: picked fresh at random each call,
        # so different epochs (or DataLoader workers) see different draws over training.
        if teacher_on_demos is not None:
            K = teacher_on_demos["action_chunks"].shape[1]
            if teacher_on_demos_k_enum is not None:
                assert teacher_on_demos_k_enum < K, (
                    f"teacher_on_demos_num_samples={self.teacher_on_demos_num_samples} but sidecar for "
                    f"episode {episode_idx} only has K={K} samples per state"
                )
                teacher_on_demos_k = teacher_on_demos_k_enum
            else:
                teacher_on_demos_k = random.randrange(K)
        else:
            teacher_on_demos_k = None
        if teacher_on_demos is not None and future_frame_idx != relative_step_idx:
            # future_frame_idx == relative_step_idx only at the last step of an episode: keep the
            # real current frame as the "future" image there (substituting would corrupt the
            # current-obs slot, which shares this dict key), but still apply the non-image targets.
            decompressed_images[future_frame_idx] = decode_single_jpeg_frame(
                teacher_on_demos["future_image_jpeg"][relative_step_idx, teacher_on_demos_k]
            )
            decompressed_wrist_images[future_frame_idx] = decode_single_jpeg_frame(
                teacher_on_demos["future_wrist_image_jpeg"][relative_step_idx, teacher_on_demos_k]
            )

        # Initialize list to store all images
        image_list = []
        current_sequence_idx = 0  # Used to track which sequence of images we are on

        # Get blank array for the first input frame (needed for the tokenizer)
        # Do not duplicate this image
        first_input_image = np.expand_dims(np.zeros_like(decompressed_images[relative_step_idx]), axis=0)
        image_list.append(first_input_image)
        current_sequence_idx += 1

        # Add proprio state if using proprio
        if self.use_proprio:
            proprio = episode_data["proprio"][relative_step_idx]
            image = decompressed_images[relative_step_idx]
            # Proprio values will be injected into latent diffusion sequence later
            # For now just add blank image
            blank_image = np.zeros_like(decompressed_images[relative_step_idx])
            blank_image = duplicate_array(blank_image, total_num_copies=self.num_duplicates_per_image)
            image_list.append(blank_image)
            current_proprio_latent_idx = current_sequence_idx
            current_sequence_idx += 1

        # Add wrist image if using wrist images
        if self.use_wrist_images:
            wrist_image = decompressed_wrist_images[relative_step_idx]
            # Duplicate wrist image
            wrist_image = duplicate_array(wrist_image, total_num_copies=self.num_duplicates_per_image)
            image_list.append(wrist_image)
            current_wrist_image_latent_idx = current_sequence_idx
            current_sequence_idx += 1

        # Add current third-person image
        if self.use_third_person_images:
            current_image = decompressed_images[relative_step_idx]
            current_image = duplicate_array(current_image, total_num_copies=self.num_duplicates_per_image)
            image_list.append(current_image)
            current_image_latent_idx = current_sequence_idx
            current_sequence_idx += 1

        # Add blank image for action chunk
        blank_image = np.zeros_like(decompressed_images[relative_step_idx])
        # Duplicate blank image
        blank_image = duplicate_array(blank_image, total_num_copies=self.num_duplicates_per_image)
        image_list.append(blank_image)
        action_latent_idx = current_sequence_idx
        current_sequence_idx += 1

        # Add future proprio
        if self.use_proprio:
            future_proprio = episode_data["proprio"][future_frame_idx]
            if teacher_on_demos is not None:
                future_proprio = teacher_on_demos["future_proprio"][relative_step_idx, teacher_on_demos_k].astype(
                    np.float32
                )
            # Not using proprio image; proprio values will be injected into latent diffusion sequence later
            # For now just add blank image
            blank_image = np.zeros_like(decompressed_images[relative_step_idx])
            blank_image = duplicate_array(blank_image, total_num_copies=self.num_duplicates_per_image)
            image_list.append(blank_image)
            future_proprio_latent_idx = current_sequence_idx
            current_sequence_idx += 1

        # Add future wrist image
        if self.use_wrist_images:
            future_wrist_image = decompressed_wrist_images[future_frame_idx]
            future_wrist_image = duplicate_array(future_wrist_image, total_num_copies=self.num_duplicates_per_image)
            image_list.append(future_wrist_image)
            future_wrist_image_latent_idx = current_sequence_idx
            current_sequence_idx += 1

        # Add future primary image
        if self.use_third_person_images:
            future_image = decompressed_images[future_frame_idx]
            future_image = duplicate_array(future_image, total_num_copies=self.num_duplicates_per_image)
            image_list.append(future_image)
            future_image_latent_idx = current_sequence_idx
            current_sequence_idx += 1

        # Add blank value image
        if self.return_value_function_returns:
            value_image = np.zeros_like(decompressed_images[relative_step_idx])
            value_image = duplicate_array(value_image, total_num_copies=self.num_duplicates_per_image)
            image_list.append(value_image)
            value_latent_idx = current_sequence_idx
            current_sequence_idx += 1

        # Stack images and preprocess. teacher_on_demos future frames come from the teacher's VAE
        # decode at model resolution (final_image_size), while the real demo frames are at their
        # on-disk resolution -- resize every slot to final_image_size before concatenating so they
        # stack. Real frames get exactly the 256->224 they'd otherwise get inside preprocess_image
        # (resize_images is deterministic and no-ops the second call); teacher frames are already
        # at final_image_size and pass through untouched. No-op for the baseline path.
        if teacher_on_demos is not None:
            image_list = [resize_images(im, self.final_image_size) for im in image_list]

        # Stack images and preprocess
        images = np.concatenate(image_list, axis=0)
        images = preprocess_image(
            images,
            final_image_size=self.final_image_size,
            normalize_images=self.normalize_images,
            use_image_aug=self.use_image_aug,
            stronger_image_aug=self.use_stronger_image_aug,
        )

        # Calculate how many actions we can get from the current index
        action_chunk = get_action_chunk_with_padding(
            actions=episode_data["actions"],
            relative_step_idx=relative_step_idx,
            chunk_size=self.chunk_size,
            num_steps=episode_data["num_steps"],
        )
        if teacher_on_demos is not None:
            # The teacher always emits a full (chunk_size, action_dim) chunk -- no tail padding.
            action_chunk = teacher_on_demos["action_chunks"][relative_step_idx, teacher_on_demos_k].astype(
                np.float32
            )

        # Return the next action chunk as well (base loss never consumes next_action_chunk /
        # next_value_function_return -- left as the real demo values on the teacher_on_demos path).
        # Calculate how many actions we can get from the current index
        next_relative_step_idx = min(relative_step_idx + self.chunk_size, episode_data["num_steps"] - 1)
        next_action_chunk = get_action_chunk_with_padding(
            actions=episode_data["actions"],
            relative_step_idx=next_relative_step_idx,
            chunk_size=self.chunk_size,
            num_steps=episode_data["num_steps"],
        )

        # Get return for value function prediction
        if self.return_value_function_returns:
            return_timestep = future_frame_idx
            if episode_metadata is not None:
                value_function_return = episode_metadata["returns"][return_timestep]
            else:
                value_function_return = episode_data["returns"][return_timestep]
            if teacher_on_demos is not None:
                value_function_return = float(teacher_on_demos["value"][relative_step_idx, teacher_on_demos_k])
        else:
            value_function_return = float("-100")  # Just a placeholder

        # Calculate next future frame index if needed
        next_future_frame_idx = next_relative_step_idx + self.chunk_size
        max_possible_idx = episode_data["num_steps"] - 1
        if next_future_frame_idx > max_possible_idx:
            next_future_frame_idx = max_possible_idx

        # Return the next value function return as well
        if self.return_value_function_returns:
            return_timestep = next_future_frame_idx
            if episode_metadata is not None:
                next_value_function_return = episode_metadata["returns"][return_timestep]
            else:
                next_value_function_return = episode_data["returns"][return_timestep]
        else:
            next_value_function_return = float("-100")  # Just a placeholder

        sample_dict = {
            "video": images,
            "actions": action_chunk,
            "t5_text_embeddings": torch.squeeze(self.t5_text_embeddings[episode_data["command"]]),
            "t5_text_mask": torch.ones(512, dtype=torch.int64),  # Just copying what others have done in this codebase
            "fps": 16,  # Just set to some fixed value since we aren't generating videos anyway
            "padding_mask": torch.zeros(
                1, self.final_image_size, self.final_image_size
            ),  # Just copying what others have done in this codebase
            "image_size": self.final_image_size
            * torch.ones(
                4
            ),  # Just copying what others have done in this codebase; important because it shows up as model input
            "proprio": proprio if self.use_proprio else np.zeros_like(episode_data["proprio"][relative_step_idx]),
            "future_proprio": (
                future_proprio if self.use_proprio else np.zeros_like(episode_data["proprio"][future_frame_idx])
            ),
            "__key__": idx,  # Unique sample identifier (required for callbacks)
            "rollout_data_mask": rollout_data_mask,
            "rollout_data_success_mask": rollout_data_success_mask,
            "world_model_sample_mask": 1 if is_world_model_sample else 0,
            "value_function_sample_mask": 1 if is_value_function_sample else 0,
            "global_rollout_idx": global_rollout_idx,
            "action_latent_idx": action_latent_idx,
            "value_latent_idx": value_latent_idx if self.return_value_function_returns else -1,
            "current_proprio_latent_idx": current_proprio_latent_idx if self.use_proprio else -1,
            "current_wrist_image_latent_idx": current_wrist_image_latent_idx if self.use_wrist_images else -1,
            "current_image_latent_idx": current_image_latent_idx if self.use_third_person_images else -1,
            "future_proprio_latent_idx": future_proprio_latent_idx if self.use_proprio else -1,
            "future_wrist_image_latent_idx": future_wrist_image_latent_idx if self.use_wrist_images else -1,
            "future_image_latent_idx": future_image_latent_idx if self.use_third_person_images else -1,
            "value_function_return": value_function_return,
            "next_action_chunk": next_action_chunk,
            "next_value_function_return": next_value_function_return,
        }

        return sample_dict


def create_augmentation_visualization(
    data_dir: str,
    t5_text_embeddings_path: str,
    fixed_idx: int = 100,
    num_augmentations: int = 50,
    output_dir: str = "./temp",
):
    """
    Create MP4 videos visualizing the distribution of augmentations for a fixed data point.

    Args:
        data_dir (str): Path to the dataset directory
        t5_text_embeddings_path (str): Path to T5 embeddings file
        fixed_idx (int): Index of the data point to apply augmentations to
        num_augmentations (int): Number of different augmentations to sample
        output_dir (str): Directory to save the visualization videos
    """
    print(f"\nCreating augmentation visualization with {num_augmentations} samples...")

    # Create a dataset instance with augmentations enabled
    aug_dataset = LIBERODataset(
        data_dir=data_dir,
        chunk_size=16,
        t5_text_embeddings_path=t5_text_embeddings_path,
        normalize_images=False,
        normalize_actions=True,
        use_image_aug=True,  # Enable augmentations
        use_wrist_images=True,
        use_proprio=True,
        normalize_proprio=True,
        num_duplicates_per_image=1,
        use_stronger_image_aug=True,
    )

    # Collect different augmentations of the same data point
    augmented_samples = []

    print(f"Generating {num_augmentations} augmentations for data point {fixed_idx}...")
    for aug_idx in tqdm(range(num_augmentations)):
        sample = aug_dataset[fixed_idx]
        augmented_samples.append(sample)

    # Extract images from all augmented samples and organize them
    # Each sample has shape (C, T, H, W) where T is number of frames
    all_augmented_videos = []
    for sample in augmented_samples:
        video = sample["video"].permute(1, 2, 3, 0).numpy()  # (T, H, W, C)
        all_augmented_videos.append(video)

    # Stack all augmented videos: (num_augmentations, T, H, W, C)
    all_augmented_videos = np.stack(all_augmented_videos, axis=0)

    # Get dimensions
    num_augs, num_frames, height, width, channels = all_augmented_videos.shape
    print(f"Augmented video array shape: {all_augmented_videos.shape}")

    # Create video frames for each frame type
    frame_names = [
        "blank_input",
        "proprio",
        "wrist",
        "current_view",
        "future_proprio",
        "future_wrist",
        "future_view",
        "blank_action",
    ]
    if not aug_dataset.use_proprio:
        frame_names.remove("proprio")
    if not aug_dataset.use_wrist_images:
        frame_names.remove("wrist")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Create MP4 videos for each frame type
    for frame_idx in range(min(num_frames, len(frame_names))):
        frame_name = frame_names[frame_idx] if frame_idx < len(frame_names) else f"frame_{frame_idx}"

        # Skip blank frames for visualization
        if "blank" in frame_name:
            continue

        print(f"Creating video for {frame_name}...")

        # Extract frames for this frame type across all augmentations
        frames_for_video = []
        for aug_idx in range(num_augs):
            frame = all_augmented_videos[aug_idx, frame_idx]  # (H, W, C)
            frames_for_video.append(frame)

        # Save as MP4 video
        video_path = os.path.join(output_dir, f"augmentation_visualization_{frame_name}.mp4")

        # Convert to uint8 if needed
        frames_array = np.stack(frames_for_video, axis=0)  # (num_augs, H, W, C)
        if frames_array.dtype != np.uint8:
            frames_array = frames_array.astype(np.uint8)

        # Save video with slower frame rate to better see the augmentations
        imageio.mimsave(video_path, frames_array, fps=5, macro_block_size=None)
        print(f"Saved augmentation visualization video: {video_path}")

    # Also create a combined video showing all frame types side by side
    print("Creating combined video with all frame types...")

    # Only use non-blank frames
    valid_frame_indices = []
    valid_frame_names = []
    for frame_idx in range(min(num_frames, len(frame_names))):
        frame_name = frame_names[frame_idx] if frame_idx < len(frame_names) else f"frame_{frame_idx}"
        if "blank" not in frame_name:
            valid_frame_indices.append(frame_idx)
            valid_frame_names.append(frame_name)

    if len(valid_frame_indices) > 0:
        combined_frames = []
        for aug_idx in range(num_augs):
            # Extract valid frames for this augmentation
            frames_to_combine = []
            for frame_idx in valid_frame_indices:
                frame = all_augmented_videos[aug_idx, frame_idx]  # (H, W, C)
                frames_to_combine.append(frame)

            # Concatenate frames horizontally
            combined_frame = np.concatenate(frames_to_combine, axis=1)  # (H, W*num_frames, C)
            combined_frames.append(combined_frame)

        # Save combined video
        combined_frames_array = np.stack(combined_frames, axis=0)  # (num_augs, H, W*num_frames, C)
        if combined_frames_array.dtype != np.uint8:
            combined_frames_array = combined_frames_array.astype(np.uint8)

        combined_video_path = os.path.join(output_dir, "augmentation_visualization_combined.mp4")
        imageio.mimsave(combined_video_path, combined_frames_array, fps=5, macro_block_size=None)
        print(f"Saved combined augmentation visualization video: {combined_video_path}")
        print(f"Combined video shows frames in order: {' | '.join(valid_frame_names)}")

    print("Augmentation visualization complete!")


if __name__ == "__main__":
    dataset = LIBERODataset(
        data_dir="users/user/data/libero_regen",  # Successful demos
        t5_text_embeddings_path="users/user/data/libero_regen/t5_embeddings.pkl",
        chunk_size=16,
        use_image_aug=True,
        use_wrist_images=True,
        use_proprio=True,
        normalize_proprio=True,
        normalize_actions=True,
        num_duplicates_per_image=4,  # WAN 2.1 tokenizer: 4 images per latent frame
        use_stronger_image_aug=True,
        rollout_data_dir="users/user/data/libero_regen_rollout_data/",  # All demo rollouts (successes + failures)
        demonstration_sampling_prob=0.5,
        success_rollout_sampling_prob=0.5,
        return_value_function_returns=True,
        gamma=0.99,
    )

    # Fetch a sample
    np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})
    idx = 50
    sample = dataset[idx]
    print(f"\nImages shape, dtype: {sample['video'].shape, sample['video'].dtype}")
    print(f"Actions shape, dtype: {sample['actions'].shape, sample['actions'].dtype}")
    print(f"Actions:\n{sample['actions']}")
    print(f"T5 text embeddings shape, dtype: {sample['t5_text_embeddings'].shape, sample['t5_text_embeddings'].dtype}")
    print(f"T5 text embeddings:\n{sample['t5_text_embeddings']}")
    print(f"Unique commands: {dataset.unique_commands}")

    # Fetch more samples and save sample images
    os.makedirs("./temp", exist_ok=True)
    for _ in range(50):
        global_step_index = random.randint(0, len(dataset) - 1)
        sample = dataset[global_step_index]
        images = sample["video"].permute(1, 2, 3, 0).numpy()
        for i in range(images.shape[0]):
            img_np = images[i]
            image_path = f"./temp/video__global_step_index_{global_step_index}__is_rollout={sample['rollout_data_mask']}__global_rollout_idx={sample['global_rollout_idx']}__is_success={sample['rollout_data_success_mask']}__value_function_return={sample['value_function_return']:.4f}__frame_idx={i}.png"
            Image.fromarray(img_np).save(image_path)
            print(f"Saved image at path: {image_path}")
