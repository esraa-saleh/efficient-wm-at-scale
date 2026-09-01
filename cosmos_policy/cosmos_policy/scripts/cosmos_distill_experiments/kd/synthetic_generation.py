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
Generates synthetic (state, task, action, predicted_future_state, predicted_value) examples by
querying the teacher via the model's own real multi-step generation process
(`generate_samples_from_batch`) -- NOT the training-time single-`denoise()`-call shortcut. That
shortcut only works because training has real ground truth to noise at a random level and recover;
there is no ground truth for a state/action pair nobody actually executed, so the only valid way to
get an answer is genuine iterative sampling from noise, exactly like real inference does it.

Two independent example-generation functions live here, differing only in what's fixed vs. sampled
at the ACTION slot (everything downstream -- video splicing, mask semantics, ShardWriter format --
follows from that one choice, see each `splice_*_into_batch` function's own docstring for the mask
difference specifically):

  - `generate_synthetic_targets`/`splice_synthetic_targets_into_batch` ("counterfactual"): the
    recorded action perturbed with noise, held FIXED as conditioning -- future state and value are
    the only two things generated, for that off-trajectory candidate. Used by
    `build_synthetic_distill_dataset.py`.
  - `generate_teacher_native_targets`/`splice_teacher_native_targets_into_batch` ("teacher-native"):
    the action itself is ALSO generated, freely, by the teacher -- action, future state, AND value
    are all three sampled jointly, with nothing held fixed. Used by
    `build_teacher_native_distill_dataset.py`. Mirrors cosmos_utils.py's own real
    `get_action(..., generate_future_state_and_value_in_parallel=True)` path -- the same "let the
    model decide the action and score it in one pass" pattern real inference already uses.

IMPORTANT, and easy to get wrong (an earlier draft of this module got it wrong): the
`world_model_sample_mask`/`value_function_sample_mask` masking that `get_data_and_condition` (the
TRAINING-time entrypoint) applies has NO EFFECT on generation. `generate_samples_from_batch` calls
`get_x0_fn_from_batch` instead, which builds its own fresh condition from scratch via
`data_batch["num_conditional_frames"]` (a plain "first N frames are given" cutoff over the fixed
slot order [blank, current_proprio, current_wrist_image, current_image, action, future_proprio,
future_wrist_image, future_image, value] -- see `compute_named_latent_idx`) and never even looks at
those two masks. This mirrors this project's own real inference code exactly: `get_qvalue_prediction`/
`get_future_state_prediction` (cosmos_utils.py) set `data_batch["num_conditional_frames"]` directly,
never the training masks.
`get_x0_fn_from_batch` also has NO general mechanism for injecting `data_batch["actions"]` into the
generation input (unlike training's `get_data_and_condition`, which injects it unconditionally into
`condition.gt_frames`) -- the only way an action reaches a `generate_samples_from_batch` call is via
`skip_vae_encoding=True` + `previous_generated_latent=<an already action-injected latent>`. Real
inference gets that latent for free by chaining off a PRIOR `generate_samples_from_batch` call (the
action-sampling stage) whose output already has the sampled action baked in. We have no such prior
call -- our action is manually perturbed, not model-sampled -- so we build the equivalent ourselves
by calling `teacher.get_data_and_condition(...)` directly (the same method training uses) and
reading `condition.gt_frames` back out, which already has our perturbed action (and real current
proprio) injected by that method (see policy_video2world_model.py's `get_data_and_condition`,
"Additionally, add the action chunk to the gt_frames" and the current-proprio injection right
above it) -- one call gives us both that injected latent AND (via the same call's `latent_state`
return value, untouched by the injection, which only mutates `condition.gt_frames`) the true blank
pre-injection latent `undo_latent_injection` needs at [[0, 1, action_idx, future_proprio_idx]] before
VAE-decoding -- see `undo_latent_injection`'s own docstring for why decoding un-cleaned injected
slots corrupts the neighboring real image frames.

Future state and value are sampled TOGETHER in the one call (matching the training masking's own
`world_model` mode semantics conceptually, just realized here via `num_conditional_frames` instead)
rather than the real eval pipeline's two staged calls -- staging exists there because the ACTION
itself is also being generated in a first stage; here the action is already fixed (perturbed, not
sampled), so there is nothing left to stage.

Reuses cosmos_utils.py's own extract_value_from_latent_sequence /
extract_action_chunk_from_latent_sequence / get_future_images_from_generated_samples directly rather
than reimplementing VAE-decode or value/proprio-extraction logic.
`extract_action_chunk_from_latent_sequence` is reused for future_proprio too, not just actions:
proprio was injected into its own latent slot via the identical tile-to-fill-the-volume scheme
`replace_latent_with_action_chunk` uses for actions (see `replace_latent_with_proprio`'s own
docstring), so the same reverse-tile-and-average extraction applies -- just called with
`action_shape=(1, PROPRIO_DIM)` instead of `(chunk_size, ACTION_DIM)`.
"""

import dataclasses

import torch

from cosmos_policy.constants import ACTION_DIM, PROPRIO_DIM
from cosmos_policy.experiments.robot.cosmos_utils import (
    extract_action_chunk_from_latent_sequence,
    extract_value_from_latent_sequence,
    get_future_images_from_generated_samples,
)
from cosmos_policy.scripts.cosmos_distill_experiments.kd.batch_prep import (
    DATASET_KWARGS,
    compute_named_latent_idx,
    policy_autocast,
)

# Mirrors cosmos_utils.py's get_future_state_prediction's own LIBERO-specific INDICES_TO_REPLACE:
# non-image slots (blank, current proprio, action, future proprio) must be replaced with their
# original pre-injection (blank) latent before VAE-decoding the whole video tensor, or those
# non-image slots decode into visual garbage -- see undo_latent_injection's own docstring (called
# internally by get_future_images_from_generated_samples).
_LATENT_IDX = compute_named_latent_idx()
_INDICES_TO_REPLACE = [0, 1, _LATENT_IDX["action"], _LATENT_IDX["future_proprio"]]


@dataclasses.dataclass(frozen=True)
class _ImageDecodeConfig:
    """The 4 fields `get_future_images_from_generated_samples` actually reads off its `cfg`
    argument -- a minimal stand-in for the real (much larger) `PolicyEvalConfig`, since every
    value here is fixed for this whole project's LIBERO setup (`DATASET_KWARGS` never varies
    these)."""

    use_wrist_image: bool = True
    use_third_person_image: bool = True
    num_wrist_images: int = 1
    num_third_person_images: int = 1


def perturb_action(action_chunk: torch.Tensor, std: float) -> torch.Tensor:
    """Clipped-Gaussian action perturbation -- the same formula the old inline
    `perturb_demo_actions_for_world_model_mode` (batch_prep.py, since removed -- see this module's
    own docstring for why that mechanism was replaced entirely by this standalone pipeline) used to
    apply."""
    return torch.clamp(action_chunk + torch.randn_like(action_chunk) * std, -1.0, 1.0)


def generate_synthetic_targets(
    teacher: torch.nn.Module,
    data_batch: dict,
    action_perturbation_std: float,
    num_steps: int,
    seed: int = 1,
) -> dict[str, torch.Tensor]:
    """Given a real batch (from LIBERODataset -- demo or rollout, doesn't matter which, both are
    valid source pools), perturbs each example's action and queries the teacher with that action
    held fixed as conditioning -- future state AND value jointly SAMPLED together in one
    `generate_samples_from_batch` call.

    Does NOT mutate `data_batch` -- builds its own shallow copy internally. The original batch's
    real action/future-state/value are irrelevant here and never touched; this function only ever
    reads the *current*-state fields off `data_batch` (images, proprio, task/text embedding).

    Returns:
        perturbed_action:    torch.Tensor (B, chunk_size, action_dim) float, on teacher's device --
                              the synthetic action actually used
        future_proprio:      torch.Tensor (B, proprio_dim) float, on teacher's device
        future_wrist_image:  np.ndarray (B, H, W, C) uint8, range [0, 255] -- already .cpu().numpy()
                              (get_future_images_from_generated_samples' own VAE-decode does this)
        future_image:        np.ndarray (B, H, W, C) uint8, range [0, 255], same as above
        value:                torch.Tensor (B,) float in [0, 1], on teacher's device
    """
    batch_size = data_batch["actions"].shape[0]
    perturbed_action = perturb_action(data_batch["actions"], action_perturbation_std)

    synthetic_batch = dict(data_batch)
    synthetic_batch["actions"] = perturbed_action

    with torch.no_grad(), policy_autocast():
        # Gives us both things we need from one call: `condition.gt_frames` (our perturbed action
        # -- and real current proprio -- injected, per this module's own docstring) to seed
        # generation, and `latent_state` (untouched by that injection) as the true blank
        # pre-injection reference `undo_latent_injection` needs post-generation.
        _raw_state, latent_state, condition = teacher.get_data_and_condition(synthetic_batch)
        action_injected_latent = condition.gt_frames

        # First `min_num_conditional_frames + 1` frames (in the fixed slot order
        # `compute_named_latent_idx()` derives) are given/clean throughout sampling -- the "+1"
        # is the action slot, exactly mirroring cosmos_utils.py's own
        # `get_future_state_prediction` ("1 more conditional frame for the action chunk"). Every
        # slot after that (future_proprio, future_wrist_image, future_image, value) is generated,
        # JOINTLY, in this one call.
        synthetic_batch["num_conditional_frames"] = teacher.config.min_num_conditional_frames + 1

        generated_sample = teacher.generate_samples_from_batch(
            synthetic_batch,
            n_sample=batch_size,
            num_steps=num_steps,
            seed=seed,
            is_negative_prompt=False,
            skip_vae_encoding=True,  # reuse action_injected_latent -- no fresh VAE encode here
            previous_generated_latent=action_injected_latent,
            return_orig_clean_latent_frames=False,  # we already have the correct one: latent_state
        )  # (B, C', T', H', W')

        device = generated_sample.device
        batch_indices_value = torch.full((batch_size,), _LATENT_IDX["value"], dtype=torch.int64, device=device)
        value = extract_value_from_latent_sequence(generated_sample, batch_indices_value)
        value = torch.clamp((value + 1.0) / 2.0, min=0.0, max=1.0)  # [-1,1] -> [0,1], as get_qvalue_prediction does

        batch_indices_proprio = torch.full(
            (batch_size,), _LATENT_IDX["future_proprio"], dtype=torch.int64, device=device
        )
        future_proprio = extract_action_chunk_from_latent_sequence(
            generated_sample, (1, PROPRIO_DIM), batch_indices_proprio
        ).squeeze(1)  # (B, 1, PROPRIO_DIM) -> (B, PROPRIO_DIM)

        future_images = get_future_images_from_generated_samples(
            teacher,
            generated_sample.clone(),
            _ImageDecodeConfig(),
            latent_state,  # true blank pre-injection latent -- NOT generated_sample's own condition
            _INDICES_TO_REPLACE,
            future_wrist_image_latent_idx=_LATENT_IDX["future_wrist_image"],
            future_wrist_image2_latent_idx=-1,
            future_image_latent_idx=_LATENT_IDX["future_image"],
            future_image2_latent_idx=-1,
        )  # {"future_wrist_image": (B,H,W,C) uint8, "future_image": (B,H,W,C) uint8}

    return dict(
        perturbed_action=perturbed_action,
        future_proprio=future_proprio,
        future_wrist_image=future_images["future_wrist_image"],
        future_image=future_images["future_image"],
        value=value,
    )


def generate_teacher_native_targets(
    teacher: torch.nn.Module,
    data_batch: dict,
    num_steps: int,
    seed: int = 1,
) -> dict[str, torch.Tensor]:
    """Given a real batch (from LIBERODataset -- demo or rollout, doesn't matter which), queries the
    teacher for its OWN freely-chosen action at this state -- together with the future state AND
    value for that self-chosen action -- ALL THREE jointly SAMPLED together in one
    `generate_samples_from_batch` call. Unlike `generate_synthetic_targets` above, nothing is
    injected or held fixed as conditioning: the action slot is generated exactly like every other
    non-current-state slot, simply by excluding it from `num_conditional_frames` (one less than
    `generate_synthetic_targets` uses). This mirrors cosmos_utils.py's own real
    `get_action(..., generate_future_state_and_value_in_parallel=True)` path -- the same "let the
    model decide the action and score it in one pass" pattern real inference already uses, just
    reusing this module's own `get_data_and_condition`-based idiom instead of that file's.

    Does NOT mutate `data_batch` -- builds its own shallow copy internally, same as
    `generate_synthetic_targets`. Only the CURRENT-state fields (images, proprio, task/text
    embedding) off `data_batch` are actually used to condition generation. `data_batch["actions"]`
    is still read once, by `teacher.get_data_and_condition` internally (which unconditionally
    injects SOME action value into `condition.gt_frames`'s action slot -- see
    policy_video2world_model.py's "Additionally, add the action chunk to the gt_frames") -- but that
    slot lies beyond `num_conditional_frames` here, so `generate_samples_from_batch` overwrites it
    with noise and denoises its own fresh action regardless of what was injected there. Whatever the
    real recorded action happens to be is therefore irrelevant to this function's output.

    Returns:
        native_action:        torch.Tensor (B, chunk_size, action_dim) float, on teacher's device --
                               the teacher's own generated action
        future_proprio:       torch.Tensor (B, proprio_dim) float, on teacher's device
        future_wrist_image:   np.ndarray (B, H, W, C) uint8, range [0, 255] -- already .cpu().numpy()
                               (get_future_images_from_generated_samples' own VAE-decode does this)
        future_image:         np.ndarray (B, H, W, C) uint8, range [0, 255], same as above
        value:                 torch.Tensor (B,) float in [0, 1], on teacher's device
    """
    batch_size = data_batch["actions"].shape[0]
    chunk_size = data_batch["actions"].shape[1]

    native_batch = dict(data_batch)

    with torch.no_grad(), policy_autocast():
        # Same two things `generate_synthetic_targets` needs from one call (see its own docstring):
        # `condition.gt_frames` to seed generation (real current proprio/wrist/primary image content
        # baked in, whatever action value happened to be injected here is about to become
        # irrelevant), and `latent_state` (untouched by that injection) as the true blank
        # pre-injection reference `undo_latent_injection` needs post-generation.
        _raw_state, latent_state, condition = teacher.get_data_and_condition(native_batch)
        seed_latent = condition.gt_frames

        # First `min_num_conditional_frames` frames (in the fixed slot order
        # `compute_named_latent_idx()` derives) are given/clean throughout sampling -- NO "+1" here,
        # unlike `generate_synthetic_targets`, since the action slot is not held fixed: it is
        # generated JOINTLY with future_proprio/future_wrist_image/future_image/value in this one
        # call, exactly like every other non-current-state slot.
        native_batch["num_conditional_frames"] = teacher.config.min_num_conditional_frames

        generated_sample = teacher.generate_samples_from_batch(
            native_batch,
            n_sample=batch_size,
            num_steps=num_steps,
            seed=seed,
            is_negative_prompt=False,
            skip_vae_encoding=True,  # reuse seed_latent -- no fresh VAE encode here
            previous_generated_latent=seed_latent,
            return_orig_clean_latent_frames=False,  # we already have the correct one: latent_state
        )  # (B, C', T', H', W')

        device = generated_sample.device
        batch_indices_action = torch.full((batch_size,), _LATENT_IDX["action"], dtype=torch.int64, device=device)
        native_action = extract_action_chunk_from_latent_sequence(
            generated_sample, (chunk_size, ACTION_DIM), batch_indices_action
        )  # (B, chunk_size, ACTION_DIM)

        batch_indices_value = torch.full((batch_size,), _LATENT_IDX["value"], dtype=torch.int64, device=device)
        value = extract_value_from_latent_sequence(generated_sample, batch_indices_value)
        value = torch.clamp((value + 1.0) / 2.0, min=0.0, max=1.0)  # [-1,1] -> [0,1], as get_qvalue_prediction does

        batch_indices_proprio = torch.full(
            (batch_size,), _LATENT_IDX["future_proprio"], dtype=torch.int64, device=device
        )
        future_proprio = extract_action_chunk_from_latent_sequence(
            generated_sample, (1, PROPRIO_DIM), batch_indices_proprio
        ).squeeze(1)  # (B, 1, PROPRIO_DIM) -> (B, PROPRIO_DIM)

        future_images = get_future_images_from_generated_samples(
            teacher,
            generated_sample.clone(),
            _ImageDecodeConfig(),
            latent_state,  # true blank pre-injection latent -- NOT generated_sample's own condition
            _INDICES_TO_REPLACE,
            future_wrist_image_latent_idx=_LATENT_IDX["future_wrist_image"],
            future_wrist_image2_latent_idx=-1,
            future_image_latent_idx=_LATENT_IDX["future_image"],
            future_image2_latent_idx=-1,
        )  # {"future_wrist_image": (B,H,W,C) uint8, "future_image": (B,H,W,C) uint8}

    return dict(
        native_action=native_action,
        future_proprio=future_proprio,
        future_wrist_image=future_images["future_wrist_image"],
        future_image=future_images["future_image"],
        value=value,
    )


def splice_synthetic_targets_into_batch(data_batch: dict, targets: dict) -> dict:
    """Builds a new LIBERODataset-shaped batch by overwriting `data_batch`'s action/future-state/
    value fields with `generate_synthetic_targets`'s output -- everything else (current images/
    proprio, text embedding, every `*_latent_idx`) stays exactly as drawn from the real batch that
    seeded generation. Lets `build_synthetic_distill_dataset.py` feed the result straight into
    `teacher.training_step(...)`, the SAME call `build_distill_dataset.py` already uses on real
    batches -- reusing distill_dataset.py's existing `ShardWriter`/`DistillShardDataset` format
    (and therefore every training script's existing consumption code) for synthetic examples too,
    rather than inventing a second on-disk format or a second training-time code path.

    `video`'s future_wrist_image/future_image slots (each `num_duplicates_per_image` identical
    raw-pixel copies -- see `compute_named_latent_idx`'s slot ordering and
    `LIBERODataset.__getitem__`'s own per-slot `duplicate_array` construction) are replaced with the
    teacher's predicted frames, tiled the same way; every other slot (blank, current_proprio,
    current_wrist_image, current_image, action, value -- none of them real per-pixel image content
    to begin with) is left untouched, since `actions`/`future_proprio`/`value_function_return` are
    what actually get latent-injected into those non-image slots downstream, not `video` itself.

    Also forces `world_model_sample_mask=1`/`value_function_sample_mask=0` for every example: the
    perturbed action is a GIVEN input here, not something to ask the model to "predict" (there is no
    right answer for what action should have been taken -- it was an arbitrary perturbation), while
    future state AND value are exactly the two things this example has a valid (teacher-sampled)
    answer for. `get_data_and_condition` (the method `teacher.training_step` calls internally) is
    what actually reads these two masks -- see policy_video2world_model.py.
    """
    synthetic_batch = dict(data_batch)
    synthetic_batch["actions"] = targets["perturbed_action"]
    synthetic_batch["future_proprio"] = targets["future_proprio"]
    synthetic_batch["value_function_return"] = targets["value"]

    video = data_batch["video"].clone()
    device = video.device
    num_dup = DATASET_KWARGS["num_duplicates_per_image"]
    for slot_name, image in (
        ("future_wrist_image", targets["future_wrist_image"]),
        ("future_image", targets["future_image"]),
    ):
        frame_start = _LATENT_IDX[slot_name] * num_dup
        frame = torch.from_numpy(image).to(device=device, dtype=video.dtype).permute(0, 3, 1, 2)  # (B,C,H,W)
        video[:, :, frame_start : frame_start + num_dup, :, :] = frame.unsqueeze(2).expand(-1, -1, num_dup, -1, -1)
    synthetic_batch["video"] = video

    batch_size = video.shape[0]
    synthetic_batch["world_model_sample_mask"] = torch.ones(batch_size, dtype=torch.int64, device=device)
    synthetic_batch["value_function_sample_mask"] = torch.zeros(batch_size, dtype=torch.int64, device=device)
    return synthetic_batch


def splice_teacher_native_targets_into_batch(data_batch: dict, targets: dict) -> dict:
    """Builds a new LIBERODataset-shaped batch by overwriting `data_batch`'s action/future-state/
    value fields with `generate_teacher_native_targets`'s output -- everything else (current
    images/proprio, text embedding, every `*_latent_idx`) stays exactly as drawn from the real batch
    that seeded generation. Same downstream reuse as `splice_synthetic_targets_into_batch` (feeds
    `teacher.training_step(...)`, writes via distill_dataset.py's existing `ShardWriter` format) --
    see that function's own docstring for the video-splicing mechanics, unchanged here.

    UNLIKE `splice_synthetic_targets_into_batch`, this forces `world_model_sample_mask=0`/
    `value_function_sample_mask=0` (the "bc" category -- see batch_prep.py's
    `sample_type_proportions_exact` docstring) for every example, not `world_model_sample_mask=1`:
    here the action is ALSO the teacher's own output, not a given/fixed input, so it belongs among
    this example's genuine denoising targets right alongside future state and value -- exactly the
    same "everything is a target" semantics real demo examples already get (LIBERODataset always
    sets both masks 0 for `sample_type == "demo"`, since a real demo has ground truth for
    literally everything). The only difference from a real demo example is that the "ground truth"
    for action/future-state/value here is the teacher's own self-consistent prediction, not what a
    human demonstrator actually did.
    """
    native_batch = dict(data_batch)
    native_batch["actions"] = targets["native_action"]
    native_batch["future_proprio"] = targets["future_proprio"]
    native_batch["value_function_return"] = targets["value"]

    video = data_batch["video"].clone()
    device = video.device
    num_dup = DATASET_KWARGS["num_duplicates_per_image"]
    for slot_name, image in (
        ("future_wrist_image", targets["future_wrist_image"]),
        ("future_image", targets["future_image"]),
    ):
        frame_start = _LATENT_IDX[slot_name] * num_dup
        frame = torch.from_numpy(image).to(device=device, dtype=video.dtype).permute(0, 3, 1, 2)  # (B,C,H,W)
        video[:, :, frame_start : frame_start + num_dup, :, :] = frame.unsqueeze(2).expand(-1, -1, num_dup, -1, -1)
    native_batch["video"] = video

    batch_size = video.shape[0]
    native_batch["world_model_sample_mask"] = torch.zeros(batch_size, dtype=torch.int64, device=device)
    native_batch["value_function_sample_mask"] = torch.zeros(batch_size, dtype=torch.int64, device=device)
    return native_batch
