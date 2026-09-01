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
Schema + loader for train_kd.py's hyperparameters. Mirrors ../params.py's own established pattern
(plain `OmegaConf.load()` + frozen dataclass schema) for consistency with this folder's idiom, even
though the reason that file gives for avoiding Hydra `defaults` composition (job 18781181, struct-
locking a partial dataclass) doesn't actually apply here -- train_kd.py is a bespoke script, not a
Hydra entrypoint, so this config is never composed into anything via `defaults` at all. It's loaded
via train_kd.py's own `--kd_params <path>` CLI flag, not `sweep.py`'s RunConfig/SweepConfig (that
schema describes how to launch via sbatch/torchrun -- a genuinely different concern from KD's own
hyperparameters, which train_kd.py never receives through the Hydra CLI overrides that
RunConfig.extra_overrides is for).
"""

import pathlib
from dataclasses import dataclass, field
from typing import List, Optional

from omegaconf import MISSING, OmegaConf

from cosmos_policy.scripts.cosmos_distill_experiments.kd.teacher_loader import (
    DEFAULT_TEACHER_CHECKPOINT,
    TEACHER_EXPERIMENT_NAME,
)


@dataclass(frozen=True)
class KDParams:
    teacher_experiment_name: str = TEACHER_EXPERIMENT_NAME
    # HF repo id (or local path) for the teacher's own checkpoint -- the base experiment's own
    # checkpoint.load_path is the pretrain-only model, NOT this released LIBERO policy, so this
    # must not default to "" (see teacher_loader.py's module docstring).
    teacher_checkpoint: str = DEFAULT_TEACHER_CHECKPOINT

    student_net_experiment_name: str = MISSING  # e.g. "cosmos_kd_student_1b_libero" (net_experiments.py)
    student_init_path: str = MISSING  # output of init_student_from_teacher.py's model.pt

    lr: float = 1e-4
    max_iter: int = MISSING
    batch_size: int = MISSING
    log_every: int = 10
    checkpoint_every: int = 500

    teacher_device: str = "cuda:0"
    student_device: str = "cuda:1"

    # Already-resolved absolute path (single suite, or a dataset_combos/ symlink-union dir for
    # multiple suites) -- resolution itself (RunConfig.suites -> data_dir, including the
    # multi-suite symlink-combo logic) stays in ../submit_sweep.py's resolve_dataset_dir, the same
    # single place the existing torchrun/Trainer path already resolves it, rather than
    # duplicating that logic here. submit_sweep.py's KD branch passes the resolved value via
    # train_kd.py's --data_dir CLI flag, which overrides this field -- set this directly only when
    # running train_kd.py by hand (not via submit_sweep.py).
    data_dir: str = MISSING
    # Always {run.data_root}/t5_embeddings.pkl regardless of single- vs. multi-suite data_dir
    # (run.data_root is always the shared "success_only" root -- a plain RunConfig field, see
    # ../conf/runs/train.yaml) -- submit_sweep.py's KD branch passes this explicitly for the same reason it resolves data_dir
    # itself, rather than this file trying to derive it from data_dir (which breaks for the
    # multi-suite dataset_combos/ case, where data_dir's parent isn't data_root).
    t5_text_embeddings_path: str = MISSING
    rollout_data_dir: str = ""

    # See batch_prep.py's combined_kd_loss: linearly interpolates the student's loss between the
    # teacher's own prediction (0.0, this job's original pure-distillation design) and the real
    # ground truth (1.0). 0.0 (default) reproduces the original KD-only behavior exactly --
    # existing runs are unaffected unless a run's own kd_params.yaml sets this explicitly.
    ground_truth_loss_weight: float = 0.0

    # Optional second loss term against a build_synthetic_distill_dataset.py-built dataset (see its
    # own module docstring) -- total loss = loss(real batch) + loss(synthetic batch), the synthetic
    # term always distill-only regardless of ground_truth_loss_weight above (that dataset's own
    # `x0`/`teacher_x0` are stored identical -- see build_synthetic_distill_dataset.py for why).
    # "" (default) disables this term entirely -- existing runs are unaffected unless a run's own
    # kd_params.yaml sets this explicitly.
    synthetic_dataset_dir: str = ""

    seed: int = 0

    run_dir: str = MISSING  # where model.pt/train_state.pt/train_loss.csv get written


def load_params(path: str, overrides: Optional[dict] = None) -> KDParams:
    """`overrides` (e.g. submit_sweep.py's resolved --data_dir/--t5_text_embeddings_path/--run_dir)
    must be merged in BEFORE `OmegaConf.to_object()`, not after via `dataclasses.replace` on the
    returned object: `to_object()` eagerly validates every MISSING field against the frozen
    dataclass schema, so a field left MISSING in the yaml specifically because it's meant to be
    filled by an override (data_dir/t5_text_embeddings_path/run_dir all are, by design -- see
    KDParams's field docstrings) would already have raised `MissingMandatoryValue` by the time a
    post-hoc `dataclasses.replace` could fill it in. Confirmed by the very first real
    `kd_smoketest` launch (not exercised by any test here, which all construct `KDParams` directly
    with every field already filled)."""
    schema = OmegaConf.structured(KDParams)
    values = OmegaConf.load(pathlib.Path(path))
    merged = OmegaConf.merge(schema, values)
    if overrides:
        merged = OmegaConf.merge(merged, overrides)
    return OmegaConf.to_object(merged)


@dataclass(frozen=True)
class StaticTrainParams:
    """train_kd_static.py's own hyperparameters -- deliberately a separate schema from KDParams,
    not a superset/subset of it: this path never loads a teacher at all (no teacher_checkpoint/
    teacher_experiment_name/teacher_device/data_dir/t5_text_embeddings_path/rollout_data_dir/
    action_perturbation_* -- all of that was already consumed once, by build_distill_dataset.py,
    when the static dataset was built), so KDParams's teacher/data fields would just be dead
    weight here."""

    student_net_experiment_name: str = MISSING
    student_init_path: str = MISSING
    student_device: str = "cuda:0"  # only one accelerator needed -- no teacher to co-locate with

    distill_dataset_dir: str = MISSING  # build_distill_dataset.py's --out_dir
    # DistillShardDataset yields individual examples (see distill_dataset.py) -- this is a real,
    # independent training-time knob now, NOT inherited from whatever batch_size the dataset was
    # built with. train_kd_static.py randomly samples this many individual examples per iteration
    # and assembles them into one batch via join_examples_into_micro_batch.
    batch_size: int = MISSING

    lr: float = 1e-4
    max_iter: int = MISSING
    log_every: int = 10
    # Every checkpoint at this cadence is KEPT (checkpoint_io.save_versioned_checkpoint), not
    # overwritten -- deliberately, so periodic_libero_eval_static.py (a separate job, not run
    # in-process during training) can sim-eval every single one independently. No training-side
    # eval, no in-loop stopping criterion -- see train_kd_static.py's module docstring.
    checkpoint_every: int = 500

    # See batch_prep.py's combined_kd_loss -- same meaning as KDParams.ground_truth_loss_weight.
    # The static dataset must have been built with x0 (ground truth) stored (see
    # build_distill_dataset.py) for any value other than 0.0 to make sense.
    ground_truth_loss_weight: float = 0.0

    # Optional second loss term, same meaning as KDParams.synthetic_dataset_dir -- see its own
    # comment for the two-term total loss this belongs to. "" (default) disables this term
    # entirely. Sampled the exact same way as `distill_dataset_dir` above
    # (distill_dataset.sample_micro_batch), `batch_size` examples per iteration.
    synthetic_dataset_dir: str = ""

    seed: int = 0

    run_dir: str = MISSING  # where checkpoints/, train_loss.csv, TRAINING_DONE get written


def load_static_params(path: str, overrides: Optional[dict] = None) -> StaticTrainParams:
    """Same MISSING-before-`to_object()` ordering concern as `load_params` above -- see its
    docstring."""
    schema = OmegaConf.structured(StaticTrainParams)
    values = OmegaConf.load(pathlib.Path(path))
    merged = OmegaConf.merge(schema, values)
    if overrides:
        merged = OmegaConf.merge(merged, overrides)
    return OmegaConf.to_object(merged)


@dataclass(frozen=True)
class BuildDistillDatasetParams:
    """build_distill_dataset.py's own hyperparameters -- same config-file pattern as KDParams/
    StaticTrainParams (plain OmegaConf.load() + frozen dataclass schema), loaded via that script's
    own `--build_params <path>` CLI flag, instead of one long list of individual CLI flags. Keeps a
    build's exact parameters reviewable/diffable/version-controllable the same way every other
    job's hyperparameters already are, rather than buried inline in a generated .sbatch file."""

    teacher_experiment_name: str = TEACHER_EXPERIMENT_NAME
    teacher_checkpoint: str = DEFAULT_TEACHER_CHECKPOINT

    data_dir: str = MISSING
    t5_text_embeddings_path: str = MISSING
    rollout_data_dir: str = ""
    # Restricts data_dir/rollout_data_dir to files whose name contains one of these (case-
    # insensitive substring match, e.g. ["ketchup"]) -- see LIBERODataset's own task_names
    # docstring (datasets/libero_dataset.py). Empty (default) uses every task, unchanged from
    # before.
    task_names: List[str] = field(default_factory=list)
    out_dir: str = MISSING  # train_kd_static.py's --distill_dataset_dir points here

    num_batches: int = MISSING  # raw batches drawn from LIBERODataset
    noise_draws_per_batch: int = 8  # teacher.training_step(...) calls per raw batch -- see
    # build_distill_dataset.py's module docstring for why repeated calls on the same raw batch give
    # distinct noise levels for free.
    batch_size: int = 8  # per raw batch, at query time only -- see distill_dataset.py's module
    # docstring for why every stored example is still independent and individually sample-able
    # regardless of this value (splitting happens before anything is written to disk).
    examples_per_shard: int = 256

    device: str = "cuda:0"
    seed: int = 0


def load_build_params(path: str, overrides: Optional[dict] = None) -> BuildDistillDatasetParams:
    """Same MISSING-before-`to_object()` ordering concern as `load_params` above -- see its
    docstring."""
    schema = OmegaConf.structured(BuildDistillDatasetParams)
    values = OmegaConf.load(pathlib.Path(path))
    merged = OmegaConf.merge(schema, values)
    if overrides:
        merged = OmegaConf.merge(merged, overrides)
    return OmegaConf.to_object(merged)


@dataclass(frozen=True)
class BuildSyntheticDistillDatasetParams:
    """build_synthetic_distill_dataset.py's own hyperparameters -- same config-file pattern as
    BuildDistillDatasetParams (plain OmegaConf.load() + frozen dataclass schema), loaded via that
    script's own `--build_params <path>` CLI flag. A deliberately separate schema, not a reuse of
    BuildDistillDatasetParams: this build has no `noise_draws_per_batch` (there's no noise-level
    sweep here -- synthetic_generation.py's `generate_synthetic_targets` does one real multi-step
    sample per selected example, not repeated single-`denoise()`-call draws at varying sigma), and
    it adds `action_perturbation_std`/`num_denoising_steps` -- fields BuildDistillDatasetParams
    never had (that script never perturbs anything -- see its own module docstring)."""

    teacher_experiment_name: str = TEACHER_EXPERIMENT_NAME
    teacher_checkpoint: str = DEFAULT_TEACHER_CHECKPOINT

    data_dir: str = MISSING
    t5_text_embeddings_path: str = MISSING
    rollout_data_dir: str = ""
    # Restricts data_dir/rollout_data_dir to files whose name contains one of these (case-
    # insensitive substring match, e.g. ["ketchup"]) -- see LIBERODataset's own task_names
    # docstring (datasets/libero_dataset.py). Empty (default) uses every task, unchanged from
    # before.
    task_names: List[str] = field(default_factory=list)
    out_dir: str = MISSING  # the synthetic-batch loss term's --synthetic_dataset_dir points here

    num_batches: int = MISSING  # raw batches drawn from LIBERODataset (both demo and rollout pools)
    batch_size: int = 8

    # Same clipped-Gaussian formula perturb_action (synthetic_generation.py) applies to every
    # selected example's real action before querying the teacher.
    action_perturbation_std: float = 0.05
    # Matches real eval's own num_denoising_steps_future_state/num_denoising_steps_value defaults
    # (both 1 -- see synthetic_generation.py's module docstring for why "match eval" is the right
    # cost to use here), not train_kd.py's much cheaper single-denoise()-call shortcut.
    num_denoising_steps: int = 1

    examples_per_shard: int = 256

    device: str = "cuda:0"
    seed: int = 0


def load_build_synthetic_params(path: str, overrides: Optional[dict] = None) -> BuildSyntheticDistillDatasetParams:
    """Same MISSING-before-`to_object()` ordering concern as `load_params` above -- see its
    docstring."""
    schema = OmegaConf.structured(BuildSyntheticDistillDatasetParams)
    values = OmegaConf.load(pathlib.Path(path))
    merged = OmegaConf.merge(schema, values)
    if overrides:
        merged = OmegaConf.merge(merged, overrides)
    return OmegaConf.to_object(merged)


@dataclass(frozen=True)
class BuildTeacherNativeDistillDatasetParams:
    """build_teacher_native_distill_dataset.py's own hyperparameters -- same config-file pattern as
    BuildSyntheticDistillDatasetParams (plain OmegaConf.load() + frozen dataclass schema), loaded
    via that script's own `--build_params <path>` CLI flag. A deliberately separate schema, not a
    reuse of BuildSyntheticDistillDatasetParams: this build has no `action_perturbation_std` (the
    action here is generated by the teacher itself, never perturbed -- see
    synthetic_generation.py's `generate_teacher_native_targets` for why there is nothing to perturb)."""

    teacher_experiment_name: str = TEACHER_EXPERIMENT_NAME
    teacher_checkpoint: str = DEFAULT_TEACHER_CHECKPOINT

    data_dir: str = MISSING
    t5_text_embeddings_path: str = MISSING
    rollout_data_dir: str = ""
    # Restricts data_dir/rollout_data_dir to files whose name contains one of these (case-
    # insensitive substring match, e.g. ["ketchup"]) -- see LIBERODataset's own task_names
    # docstring (datasets/libero_dataset.py). Empty (default) uses every task, unchanged from
    # before.
    task_names: List[str] = field(default_factory=list)
    out_dir: str = MISSING  # the teacher-native-batch loss term's --synthetic_dataset_dir points here

    num_batches: int = MISSING  # raw batches drawn from LIBERODataset (both demo and rollout pools)
    batch_size: int = 8

    # Matches real eval's own num_denoising_steps_future_state/num_denoising_steps_value defaults
    # (both 1 -- see synthetic_generation.py's module docstring for why "match eval" is the right
    # cost to use here), not train_kd.py's much cheaper single-denoise()-call shortcut.
    num_denoising_steps: int = 1

    examples_per_shard: int = 256

    device: str = "cuda:0"
    seed: int = 0


def load_build_teacher_native_params(
    path: str, overrides: Optional[dict] = None
) -> BuildTeacherNativeDistillDatasetParams:
    """Same MISSING-before-`to_object()` ordering concern as `load_params` above -- see its
    docstring."""
    schema = OmegaConf.structured(BuildTeacherNativeDistillDatasetParams)
    values = OmegaConf.load(pathlib.Path(path))
    merged = OmegaConf.merge(schema, values)
    if overrides:
        merged = OmegaConf.merge(merged, overrides)
    return OmegaConf.to_object(merged)
