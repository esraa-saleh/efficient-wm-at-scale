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
The canonical (xt, sigma, condition) triple the KD loop needs never gets built by code in this
package: CosmosPolicyDiffusionModel.training_step's own return value already contains it.
training_step (cosmos_policy/models/policy_text2world_model.py:240-308) builds
`output_batch = {"xt": ..., "sigma": ..., "condition": ..., "model_pred": ..., ...}`
(policy_text2world_model.py:666-699) where `model_pred` is the real `self.denoise(xt, sigma,
condition)` result (line 410) -- so calling `teacher.training_step(data_batch, iteration)` under
`torch.no_grad()` gets both the canonical batch prep AND the teacher's KD-target output
(`output_batch["model_pred"].x0`) from one unmodified upstream call. See train_kd.py for that call
site. This module only has to solve the one remaining problem: moving the resulting `condition`
object (a mutable dataclass, see cosmos_policy/conditioner.py's Text2WorldCondition) onto the
student's device without aliasing the teacher's copy.
"""

import csv
import pathlib
import time
from dataclasses import fields
from typing import TypeVar

import torch
import torch.nn.functional as F

ConditionT = TypeVar("ConditionT")

# Matches libero_all_4_suites_dataset's construction
# (cosmos_policy/config/experiment/cosmos_policy_experiment_configs.py:41-61) -- the same dataset
# settings the released teacher was itself trained with, so the KD student (live or from a static
# dataset) sees the same distribution of inputs the teacher's own training did. Shared by
# train_kd.py and build_distill_dataset.py so both construct LIBERODataset identically.
DATASET_KWARGS = dict(
    chunk_size=16,
    use_image_aug=True,
    use_wrist_images=True,
    use_proprio=True,
    normalize_proprio=True,
    normalize_actions=True,
    num_duplicates_per_image=4,
    use_stronger_image_aug=True,
    demonstration_sampling_prob=0.5,
    success_rollout_sampling_prob=0.5,
    return_value_function_returns=True,
    gamma=0.99,
)


def endless_batches(loader):
    """Yields batches from `loader` forever, re-iterating from the start once exhausted.

    NOT `itertools.cycle(loader)`: `cycle` internally saves every item it ever yields (so it can
    replay them once the source is exhausted), which means it pins every batch a build script has
    ever pulled -- full `video` tensors included -- in RAM for the rest of the run. That's an
    unbounded leak, not a hypothetical one: it's what OOM-killed the real 2000-batch
    build_synthetic_distill_dataset.py / build_teacher_native_distill_dataset.py runs after ~1300+
    batches, while the smaller 500-batch build_distill_dataset.py runs (same pattern, fewer batches)
    stayed under the memory limit. A plain re-`iter()` loop instead re-reads from `loader` each
    time around, so a completed pass is freed like any other exhausted iterator.
    """
    while True:
        yield from loader


def move_batch_to_device(data_batch: dict, device: str) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in data_batch.items()}


def combined_kd_loss(
    student_x0: torch.Tensor,
    teacher_x0: torch.Tensor,
    ground_truth_x0: torch.Tensor,
    ground_truth_loss_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The student's loss against BOTH targets available for a training example: the teacher's own
    prediction (pure distillation, this KD job's original design) and the real demonstration/
    rollout ground truth (`output_batch["x0"]` -- the same clean, frame-replace-injected target the
    base non-KD job trains against, and already computed by the same `teacher.training_step(...)`
    call that produces the distillation target, so nothing extra needs querying to get it).

    `ground_truth_loss_weight` (0 to 1) linearly interpolates between the two:
    `ground_truth_loss_weight=0.0` reproduces this job's original pure-distillation behavior
    exactly (the KD-only loss this job shipped with); `1.0` would train on ground truth alone,
    ignoring the teacher entirely. Returned separately (not just the combined scalar) so both
    components can be logged -- useful for telling "the student is drifting from the teacher" apart
    from "the student is drifting from the real answer," which the combined number alone can't.
    """
    distill_loss = F.mse_loss(student_x0, teacher_x0)
    ground_truth_loss = F.mse_loss(student_x0, ground_truth_x0)
    loss = ground_truth_loss_weight * ground_truth_loss + (1.0 - ground_truth_loss_weight) * distill_loss
    return loss, distill_loss, ground_truth_loss


# Canonical slot order every per-slot breakdown (compute_named_latent_idx, per_slot_kd_losses,
# append_loss_csv_with_slots) iterates in -- fixes the CSV column order across every training
# script/run, not just within one process's lifetime, so train_loss.csv files from different runs
# stay diffable/comparable column-for-column.
SLOT_NAMES: tuple[str, ...] = (
    "current_proprio",
    "current_wrist_image",
    "current_image",
    "action",
    "future_proprio",
    "future_wrist_image",
    "future_image",
    "value",
)


def compute_named_latent_idx() -> dict[str, int]:
    """Mirrors `LIBERODataset.__getitem__`'s sequence-assembly order (datasets/libero_dataset.py)
    for this module's own `DATASET_KWARGS` -- every slot's latent index is a deterministic constant
    for this whole distillation project (DATASET_KWARGS never varies at runtime), not per-example
    data. `build_distill_dataset.py`'s precomputed shards don't store them (see
    distill_dataset.py's module docstring), so `train_kd_static_av.py`/`train_kd_static_action.py`
    can't read them off a real batch the way `train_kd_av.py`/`train_kd_action.py` still can
    (LIBERODataset's own `__getitem__` sets them directly). Those `_av`/`_action` live-path scripts
    cross-check this derivation against the real per-batch `action_latent_idx`/`value_latent_idx`
    every iteration (a cheap assert) precisely so any future drift between the two is caught
    immediately instead of silently training the static-path scripts against the wrong token
    position.

    Sequence order (only entries whose DATASET_KWARGS flag is on occupy a slot -- disabled ones map
    to -1, matching the `-1 indicates ... not used` convention `policy_video2world_model.py`/
    `cosmos_utils.py` already use):
        [blank, current_proprio?, current_wrist_image?, current_image, action,
         future_proprio?, future_wrist_image?, future_image, value?]
    `current_image`/`future_image` are unconditional here because DATASET_KWARGS doesn't override
    `use_third_person_images`, so LIBERODataset's own default (True) applies.
    """
    idx = 1  # slot 0 is always the blank first frame
    result: dict[str, int] = {}

    result["current_proprio"] = idx if DATASET_KWARGS["use_proprio"] else -1
    idx += 1 if DATASET_KWARGS["use_proprio"] else 0
    result["current_wrist_image"] = idx if DATASET_KWARGS["use_wrist_images"] else -1
    idx += 1 if DATASET_KWARGS["use_wrist_images"] else 0
    result["current_image"] = idx
    idx += 1
    result["action"] = idx
    idx += 1
    result["future_proprio"] = idx if DATASET_KWARGS["use_proprio"] else -1
    idx += 1 if DATASET_KWARGS["use_proprio"] else 0
    result["future_wrist_image"] = idx if DATASET_KWARGS["use_wrist_images"] else -1
    idx += 1 if DATASET_KWARGS["use_wrist_images"] else 0
    result["future_image"] = idx
    idx += 1
    result["value"] = idx if DATASET_KWARGS["return_value_function_returns"] else -1

    return result


def compute_action_value_latent_idx() -> tuple[int, int]:
    """`(action_latent_idx, value_latent_idx)` -- the two slots `combined_kd_loss_action_value_only`/
    `combined_kd_loss_action_only` need. A thin convenience wrapper over
    `compute_named_latent_idx()`'s `"action"`/`"value"` entries; see that function's own docstring
    for the full derivation and why it exists."""
    named = compute_named_latent_idx()
    return named["action"], named["value"]


def per_slot_kd_losses(
    student_x0: torch.Tensor,
    teacher_x0: torch.Tensor,
    ground_truth_x0: torch.Tensor,
    ground_truth_loss_weight: float,
    latent_idx_by_name: dict[str, int],
) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Breaks `combined_kd_loss`'s (or `combined_kd_loss_action_value_only`'s/
    `combined_kd_loss_action_only`'s) blended MSE down per named slot -- e.g. so a training script
    can tell "the action loss is fine but future_wrist_image is what's driving the total up" apart
    from one opaque scalar. Purely a logging aid: nothing here is meant to be backpropagated (call
    this in addition to, not instead of, whichever `combined_kd_loss*` function actually produces
    the optimized loss) -- pass `latent_idx_by_name` skipped down to just the slots that function
    itself supervises (e.g. `{"action": ..., "value": ...}` for the `_av` scripts) so this reports
    exactly the same scope the optimizer sees, not slots that were never part of the loss to begin
    with.

    Entries whose index is -1 (disabled by DATASET_KWARGS -- see `compute_named_latent_idx()`) are
    skipped, not included with a placeholder value.
    """
    result: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for name, idx in latent_idx_by_name.items():
        if idx == -1:
            continue
        student_slot = student_x0[:, :, idx, :, :]
        teacher_slot = teacher_x0[:, :, idx, :, :]
        ground_truth_slot = ground_truth_x0[:, :, idx, :, :]
        distill_loss = F.mse_loss(student_slot, teacher_slot)
        ground_truth_loss = F.mse_loss(student_slot, ground_truth_slot)
        loss = ground_truth_loss_weight * ground_truth_loss + (1.0 - ground_truth_loss_weight) * distill_loss
        result[name] = (loss, distill_loss, ground_truth_loss)
    return result


def combined_kd_loss_action_value_only(
    student_x0: torch.Tensor,
    teacher_x0: torch.Tensor,
    ground_truth_x0: torch.Tensor,
    ground_truth_loss_weight: float,
    action_latent_idx: int,
    value_latent_idx: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Same blend as `combined_kd_loss` (see its own docstring) -- teacher-distillation vs.
    ground-truth, interpolated by `ground_truth_loss_weight` -- but computed ONLY over the action
    and value slots of the shared (B, C, T, H, W) video latent. The future-state slots (current/
    future proprio, wrist image, third-person image) are excluded entirely from the loss, not just
    down-weighted: nothing here tells the model what those slots should denoise to, mirroring
    `joint_ve_model.py`'s "value equivalence" design in the simple_dit/ folder, just applied to the
    real teacher/student's loss instead of a toy model's. The forward pass itself is untouched --
    those slots still get denoised every step (there's nothing else to occupy those token
    positions in the real teacher/student's fixed video-latent grid), only the loss ignores them.

    `action_latent_idx`/`value_latent_idx` select which index along dim=2 (T) each slot occupies --
    see `compute_action_value_latent_idx()` for how these are derived.
    """
    def _select(x0: torch.Tensor) -> torch.Tensor:
        return torch.cat([x0[:, :, action_latent_idx, :, :], x0[:, :, value_latent_idx, :, :]], dim=0)

    student_av = _select(student_x0)
    teacher_av = _select(teacher_x0)
    ground_truth_av = _select(ground_truth_x0)

    distill_loss = F.mse_loss(student_av, teacher_av)
    ground_truth_loss = F.mse_loss(student_av, ground_truth_av)
    loss = ground_truth_loss_weight * ground_truth_loss + (1.0 - ground_truth_loss_weight) * distill_loss
    return loss, distill_loss, ground_truth_loss


def combined_kd_loss_action_only(
    student_x0: torch.Tensor,
    teacher_x0: torch.Tensor,
    ground_truth_x0: torch.Tensor,
    ground_truth_loss_weight: float,
    action_latent_idx: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Same blend as `combined_kd_loss`/`combined_kd_loss_action_value_only` (see their own
    docstrings), narrower still: computed ONLY over the action slot of the shared (B, C, T, H, W)
    video latent -- value AND every future-state slot are excluded from the loss, not just
    down-weighted. The forward pass itself is untouched (same reasoning as
    `combined_kd_loss_action_value_only`'s own docstring) -- only the loss ignores every slot but
    the action one.

    `action_latent_idx` selects which index along dim=2 (T) the action slot occupies -- see
    `compute_action_value_latent_idx()` for how this is derived (its `value_latent_idx` return is
    simply unused here).
    """
    def _select(x0: torch.Tensor) -> torch.Tensor:
        return x0[:, :, action_latent_idx, :, :]

    student_action = _select(student_x0)
    teacher_action = _select(teacher_x0)
    ground_truth_action = _select(ground_truth_x0)

    distill_loss = F.mse_loss(student_action, teacher_action)
    ground_truth_loss = F.mse_loss(student_action, ground_truth_action)
    loss = ground_truth_loss_weight * ground_truth_loss + (1.0 - ground_truth_loss_weight) * distill_loss
    return loss, distill_loss, ground_truth_loss


def split_synthetic_action_distill_loss(
    student_x0: torch.Tensor,
    teacher_x0: torch.Tensor,
    condition_video_input_mask_B_C_T_H_W: torch.Tensor,
    action_latent_idx: int,
) -> tuple[float | None, float | None]:
    """Splits a synthetic-batch action-slot distill MSE into its two constituent sample types,
    which `params.synthetic_dataset_dir` silently merges together whenever it points at
    kd_synthetic_distill_dataset_merged_std_0_05-style directory (symlinks combining
    build_synthetic_distill_dataset.py's perturbed-action shards with
    build_teacher_native_distill_dataset.py's teacher-native shards -- see those two scripts' own
    module docstrings): action `given` (`condition_video_input_mask_B_C_T_H_W==1` at
    `action_latent_idx` -- the perturbed-action data, where `denoise_replace_gt_frames` clamps the
    action slot to `condition.gt_frames` regardless of network weights, so this half's distill loss
    is ALWAYS exactly 0 -- confirmed empirically against a real shard with an untrained student) vs
    action `predicted` (mask==0 -- the teacher-native data, where the action slot is a genuine,
    heavily-noised denoising target the student actually has to learn).

    The training loop's own `total_loss` keeps summing the combined (unsplit) synthetic distill
    loss -- this is a logging-only split, so averaging the two halves back together with the
    combined batch's per-example counts reproduces the exact number the unsplit loss function
    already returns. Splitting matters because that combined number is otherwise a blend of a
    guaranteed 0 and whatever the real (teacher-native) number is, which hides the very thing
    you'd want to monitor -- whether the teacher-native half is actually learning -- behind the
    other half's free zeros.

    Returns `(given_loss, predicted_loss)` as plain floats, `None` for either half absent from this
    particular micro-batch (the mask is per-example, so a random sample could -- rarely, at typical
    batch sizes -- draw all-one-type)."""
    action_given_B = (
        condition_video_input_mask_B_C_T_H_W[:, :, action_latent_idx, :, :].reshape(student_x0.shape[0], -1).mean(dim=1)
        > 0.5
    )
    per_example_mse = F.mse_loss(
        student_x0[:, :, action_latent_idx, :, :], teacher_x0[:, :, action_latent_idx, :, :], reduction="none"
    ).mean(dim=[1, 2, 3])
    given_loss = per_example_mse[action_given_B].mean().item() if action_given_B.any() else None
    predicted_loss = per_example_mse[~action_given_B].mean().item() if (~action_given_B).any() else None
    return given_loss, predicted_loss


# Canonical column order for sample-type-proportion logging -- covers both the exact 3-way split
# (`sample_type_proportions_exact`, live path only) and the approximate 2-way split
# (`sample_type_proportions_from_condition`, both paths) in one fixed order, same reasoning as
# `SLOT_NAMES`: keeps `train_loss.csv`'s header stable/diffable across runs regardless of which
# split a given script happens to log.
SAMPLE_TYPE_NAMES: tuple[str, ...] = ("bc", "world_model", "value_function", "action_given", "action_predicted")


def sample_type_proportions_exact(
    world_model_sample_mask: torch.Tensor, value_function_sample_mask: torch.Tensor
) -> dict[str, float]:
    """The EXACT per-batch composition of `LIBERODataset`'s own three mutually-exclusive sample
    types (see libero_dataset.py's `is_world_model_sample`/`is_value_function_sample` -- an
    if/else, so every example is exactly one of the three): `"bc"` (both masks 0 -- action is a
    genuine denoising target, the normal case), `"world_model"` (`world_model_sample_mask==1` --
    action given, future state predicted; includes both real off-trajectory rollout examples AND
    this project's synthetic `perturb_demo_actions_for_world_model_mode` augmentation -- pass the
    POST-perturbation masks, not the raw dataset output, so this reflects what the model actually
    trained on this iteration), `"value_function"` (`value_function_sample_mask==1` -- nearly
    everything given, only value predicted).

    Only available on the live path (`train_kd.py`/`train_kd_av.py`/`train_kd_action.py`): these
    masks are real `LIBERODataset.__getitem__` output, not something `train_kd_static*.py` has
    access to (`build_distill_dataset.py`'s shards don't persist them -- see
    `sample_type_proportions_from_condition`'s own docstring for that path's approximation
    instead)."""
    is_world_model = world_model_sample_mask == 1
    is_value_function = value_function_sample_mask == 1
    is_bc = ~is_world_model & ~is_value_function
    n = world_model_sample_mask.shape[0]
    return {
        "bc": is_bc.float().sum().item() / n,
        "world_model": is_world_model.float().sum().item() / n,
        "value_function": is_value_function.float().sum().item() / n,
    }


def sample_type_proportions_from_condition(condition, action_latent_idx: int) -> dict[str, float]:
    """An APPROXIMATION of `sample_type_proportions_exact`'s 3-way split, usable on the static path
    (`train_kd_static*.py`) where the original `world_model_sample_mask`/`value_function_sample_mask`
    labels no longer exist (`build_distill_dataset.py`'s shards only persist `(xt, sigma, condition,
    teacher_x0, x0)` -- see `distill_dataset.py`'s module docstring). Rather than guess at the
    original 3-way label, this reads back the one thing that's still actually true for each
    example: whether its action slot's `condition_video_input_mask_B_C_T_H_W` was 1 (given/clean,
    not noised -- covers both `"world_model"` and `"value_function"` samples, which both force
    action given) or 0 (a genuine denoising target -- `"bc"` samples only). This collapses
    `sample_type_proportions_exact`'s 3 categories into 2 (`"action_given"` merges `world_model` +
    `value_function`), which is what's actually recoverable post-hoc, not a design choice to hide
    the finer distinction -- name the columns accordingly in comparisons against the live path's
    exact 3-way numbers.

    `condition.condition_video_input_mask_B_C_T_H_W` is (B, C, T, H, W); the mask is broadcast
    uniformly across C/H/W per example (see `policy_video2world_model.py`'s own assignment), so any
    single channel/pixel reflects the whole example -- `.mean()` over C/H/W is just a
    robust-to-that-assumption way to reduce to one scalar per example rather than relying on
    exact indexing into an arbitrary channel.
    """
    mask_B_C_H_W = condition.condition_video_input_mask_B_C_T_H_W[:, :, action_latent_idx, :, :]
    action_given_B = mask_B_C_H_W.reshape(mask_B_C_H_W.shape[0], -1).mean(dim=1) > 0.5
    frac_action_given = action_given_B.float().mean().item()
    return {"action_given": frac_action_given, "action_predicted": 1.0 - frac_action_given}


def append_loss_csv(
    csv_path: pathlib.Path, iteration: int, loss: float, distill_loss: float, ground_truth_loss: float
) -> None:
    is_new = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["iteration", "loss", "distill_loss", "ground_truth_loss", "timestamp"])
        writer.writerow([iteration, loss, distill_loss, ground_truth_loss, time.time()])


def append_loss_csv_with_slots(
    csv_path: pathlib.Path,
    iteration: int,
    loss: float,
    distill_loss: float,
    ground_truth_loss: float,
    per_slot: dict[str, tuple[float, float, float]],
    sample_type_proportions: dict[str, float] | None = None,
    synthetic_distill_loss: float | None = None,
    synthetic_action_given_distill_loss: float | None = None,
    synthetic_action_predicted_distill_loss: float | None = None,
) -> None:
    """Same as `append_loss_csv`, plus three extra columns per entry in `per_slot` (the output of
    `per_slot_kd_losses`, already `.item()`'d): `<name>_loss`, `<name>_distill_loss`,
    `<name>_ground_truth_loss`. Column order follows `SLOT_NAMES`, filtered down to whichever names
    `per_slot` actually contains (so e.g. the `_av` scripts' 2-slot breakdown and the full
    8-slot breakdown both produce a stable, deterministic header) -- NOT `per_slot`'s own dict
    iteration order, which callers shouldn't be relied on to keep consistent across process
    restarts.

    `sample_type_proportions` (output of `sample_type_proportions_exact` or
    `sample_type_proportions_from_condition`, optional) adds one `frac_<name>` column per entry,
    in `SAMPLE_TYPE_NAMES` order -- e.g. `frac_bc`/`frac_world_model`/`frac_value_function` on the
    live path, `frac_action_given`/`frac_action_predicted` on the static path. Omit (default None)
    to keep the CSV exactly as it was before this was added -- existing callers/tests that don't
    pass it are unaffected.

    `synthetic_distill_loss` (optional): the second loss term computed against a
    build_synthetic_distill_dataset.py-built dataset, when a training script has one configured --
    see e.g. train_kd_static.py's own module docstring for the two-term total loss this belongs to
    (`loss` above stays exactly the real-data term's own value, unchanged in meaning; the actual
    backward pass sums this column and `loss` together, but each is logged separately so either can
    be inspected/plotted on its own). Omit (default None) when no synthetic dataset is configured --
    existing callers/tests that don't pass it are unaffected, same as `sample_type_proportions`.

    `synthetic_action_given_distill_loss`/`synthetic_action_predicted_distill_loss` (optional): the
    two-way split of the synthetic term's action-slot component -- see
    `split_synthetic_action_distill_loss`'s own docstring for why `synthetic_distill_loss` alone
    can't distinguish "the perturbed-action half, which is always exactly 0" from "the
    teacher-native half, which is real signal." Pass both or neither (both None when either half was
    empty for this particular batch, or when the training script doesn't compute this split).
    """
    slot_order = [name for name in SLOT_NAMES if name in per_slot]
    sample_type_order = [name for name in SAMPLE_TYPE_NAMES if sample_type_proportions and name in sample_type_proportions]
    is_new = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            header = ["iteration", "loss", "distill_loss", "ground_truth_loss"]
            for name in slot_order:
                header += [f"{name}_loss", f"{name}_distill_loss", f"{name}_ground_truth_loss"]
            for name in sample_type_order:
                header.append(f"frac_{name}")
            if synthetic_distill_loss is not None:
                header.append("synthetic_distill_loss")
            if synthetic_action_given_distill_loss is not None:
                header.append("synthetic_action_given_distill_loss")
            if synthetic_action_predicted_distill_loss is not None:
                header.append("synthetic_action_predicted_distill_loss")
            header.append("timestamp")
            writer.writerow(header)
        row = [iteration, loss, distill_loss, ground_truth_loss]
        for name in slot_order:
            row += list(per_slot[name])
        for name in sample_type_order:
            row.append(sample_type_proportions[name])
        if synthetic_distill_loss is not None:
            row.append(synthetic_distill_loss)
        if synthetic_action_given_distill_loss is not None:
            row.append(synthetic_action_given_distill_loss)
        if synthetic_action_predicted_distill_loss is not None:
            row.append(synthetic_action_predicted_distill_loss)
        row.append(time.time())
        writer.writerow(row)


def policy_autocast():
    """`torch.autocast(device_type="cuda", dtype=torch.bfloat16)` -- required around every
    `model.training_step(...)`/`model.denoise(...)` call in this package. The real Trainer/FSDP
    path gets this for free: FSDP's mixed-precision wrapping auto-casts forward-pass inputs to the
    configured param dtype. `load_model_from_checkpoint(..., enable_fsdp=False)` (what every KD
    script uses, per the locked "no FSDP" design decision) skips that wrapping entirely, so
    metadata tensors the model itself never casts to bf16 (e.g. `condition.padding_mask`, built
    fresh per batch, not part of any bf16-converted model state) reach a bf16-weighted layer
    unconverted -- concretely, `prepare_embedded_sequence`'s `torch.cat([x (bf16), padding_mask
    (float32)], dim=1)` (minimal_v4_dit.py:1684-1687) upcasts the result to float32, which then
    fails at the next bf16 Linear layer with `RuntimeError: expected mat1 and mat2 to have the
    same dtype`. This exact failure was reproduced against the real released teacher checkpoint
    (batch_prep_equivalence_test.py, without this fix); autocast reproduces what FSDP's
    mixed-precision cast would have done, without requiring FSDP itself."""
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def move_condition(condition: ConditionT, device: torch.device | str) -> ConditionT:
    """Reconstructs `condition` on `device`, moving every Tensor-valued field and leaving
    everything else untouched. Uses the same idiom this codebase's own
    Text2WorldCondition.edit_data_type (cosmos_policy/conditioner.py:90-99) uses for building a
    modified copy of a condition object: `to_dict(skip_underscore=False)` + `type(condition)(**kwargs)`.
    `skip_underscore=False` is required here (not the `to_dict()` default) so private fields like
    `_is_broadcasted` are preserved across the move -- dropping it would silently reset broadcast
    state on the moved copy. Reconstructing rather than mutating the tensors in place (Text2WorldCondition
    is `@dataclass(frozen=False)`, so in-place mutation is possible) is deliberate: it keeps the
    teacher's and student's condition objects from aliasing the same underlying tensors.

    Fields set dynamically outside the dataclass's declared fields (e.g. `orig_x0_B_C_T_H_W`,
    assigned in compute_loss_with_epsilon_and_sigma for debugging/visualization only,
    policy_text2world_model.py:376) are NOT carried over -- `denoise()`'s own `condition.to_dict()`
    call (policy_video2world_model.py:452) only ever reads declared fields too, so this is exactly
    what `denoise()` needs and nothing more.
    """
    kwargs = condition.to_dict(skip_underscore=False)
    moved = {key: (value.to(device) if isinstance(value, torch.Tensor) else value) for key, value in kwargs.items()}
    return type(condition)(**moved)


def _condition_field_names(condition) -> list[str]:
    """Declared dataclass field names on `condition` -- used by tests to assert `move_condition`
    covers every field `to_dict`/`denoise()` would see, not just the ones a particular fixture happens
    to populate."""
    return [f.name for f in fields(condition)]
