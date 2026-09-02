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
Structured-config *schema* for what submit_sweep.py can launch -- types and docs only, no values.
The actual runs (`train`, `smoketest`) are defined in conf/runs/*.yaml, composed into
`SweepConfig.runs` by conf/config.yaml's `defaults` list (each `runs@runs.<name>: <name>` entry).
Adding a new launchable run means adding a new conf/runs/<name>.yaml + one `defaults` line, not
editing this file. Every field remains overridable from the command line the normal Hydra way, e.g.:

    python -m cosmos_policy.scripts.cosmos_distill_experiments.submit_sweep \\
      launch.only=[smoketest] launch.submit=true runs.smoketest.max_iter=5

This is the one place in this folder that uses vanilla Hydra (@hydra.main + ConfigStore) rather
than this repo's own LazyConfig/override() machinery in _src/imaginaire/config.py. That machinery
(used by experiment.py) builds the *training* Config/Trainer object and is specific to that job;
submit_sweep.py is a plain submission script with no relationship to it, so it uses Hydra directly
instead of going through that layer.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass(kw_only=True)
class RunConfig:
    """Every field any of this folder's four launch mechanisms needs, flat -- no nested/composed
    sub-configs (that's the thing that broke here before, see params.py's docstring on job
    18781181: Hydra `defaults`-composing a *structured* dataclass into part of the training config
    struct-locks it). A flat dataclass with plain-value fields, which this already is, doesn't hit
    that failure mode -- so every hyperparameter any run needs, including the KD scripts' own
    (student size, lr, perturbation, ...), lives directly here instead of in a hand-maintained
    side file, and `conf/runs/<name>.yaml` is the one place that ever needs editing. Only the
    fields `run_type` actually uses apply to a given run -- see each field's own comment for which.
    """

    job_name: str = MISSING  # Becomes job.name/run_dir/checkpoint dir/loss CSV path -- must be unique across `runs`.

    # Which of the eleven things build_sbatch_script() generates for this run. "torchrun": the normal
    # Hydra/imaginaire Trainer path (experiment.py's registered experiment). "kd_live": train_kd.py,
    # querying the teacher every iteration. "kd_static": train_kd_static.py, training from a dataset
    # build_distill_dataset.py already built once. "kd_live_av"/"kd_static_av": train_kd_av.py/
    # train_kd_static_av.py -- byte-for-byte the same as "kd_live"/"kd_static" except the loss only
    # supervises the action-chunk and value slots of the shared video latent, not the future-state
    # ones. "kd_live_action"/"kd_static_action": train_kd_action.py/train_kd_static_action.py --
    # narrower still, the loss supervises ONLY the action-chunk slot (value excluded too). See those
    # scripts' own module docstrings. "build_distill_dataset": build_distill_dataset.py itself -- a
    # one-shot data-generation job, not a training run (no loss curve, no resume), pure real demo/
    # rollout data. "build_synthetic_distill_dataset": build_synthetic_distill_dataset.py -- a
    # SEPARATE one-shot data-generation job (not a mode of build_distill_dataset), perturbing each
    # selected example's action and querying the teacher for the resulting future state/value via
    # real multi-step sampling -- see that script's and synthetic_generation.py's own module
    # docstrings. "build_teacher_native_distill_dataset": build_teacher_native_distill_dataset.py --
    # ANOTHER separate one-shot data-generation job (not a mode of either build type above): the
    # teacher's own freely-generated action for each selected state, with action/future-state/value
    # ALL sampled jointly (nothing held fixed) -- see that script's and synthetic_generation.py's own
    # module docstrings. "kd_static_eval": kd/periodic_libero_eval_static.py, a SEPARATE Slurm job (not a
    # training run itself, and never feeds back into training) that continuously full-suite
    # sim-evals a running/finished kd_static(_av/_action) run's checkpoints/ as they appear -- see
    # that script's module docstring.
    run_type: str = "torchrun"

    # --- cluster-storage paths, every run_type. Same values across every run in practice (one
    # cluster account), but a plain per-run field rather than a shared conf/paths.yaml -- every
    # run's config is meant to be the one file that fully determines it, this included.
    data_root: str = MISSING  # Base dir `suites` resolves against, unless a suite already starts with "/".
    output_root: str = MISSING  # Checkpoints/loss-CSVs/wandb/DeviceMonitor/dataset_combos/ all land under here.

    # --- torchrun only: which registered Hydra experiment to launch. For train.yaml specifically,
    # this SAME field doubles as the name train.yaml's own architecture fields below register that
    # experiment under (see experiment.py) -- every other torchrun run (smoketest.yaml,
    # baseline_1b_train.yaml) just references an experiment defined elsewhere by name, same as
    # always.
    experiment: str = ""
    checkpoint_save_iter: int = 200
    extra_overrides: Dict[str, str] = field(default_factory=dict)  # Any other config.path=value overrides.
    # torchrun's --master_port. Default matches every existing run's prior hardcoded value, so this
    # is a no-op unless a yaml overrides it. Needs to be overridden when two torchrun runs might land
    # on the SAME physical node at once (Slurm can pack multiple 1-GPU jobs from the same user onto
    # one multi-GPU node) -- same port + same node = torchrun's rendezvous TCPStore.listen() fails
    # with EADDRINUSE, a real launch failure, not the "cancel and resubmit for a slow node" kind.
    master_port: int = 12341

    # --- train.yaml only: this run's `experiment` field is what train.yaml itself DEFINES (see
    # experiment.py, which reads these specific fields from conf/runs/train.yaml, filtered, to
    # build+register the Hydra experiment) -- not a per-launch override the way every other field
    # here is. smoketest.yaml/baseline_1b_train.yaml don't set these: they reference train.yaml's
    # already-registered experiment by name instead of defining their own. Defaults match this
    # job's actual "tiny" architecture, so only train.yaml needs to state them (matching that
    # default, explicitly, so the real values are visible in the one file that owns them, not
    # implied by a default sitting here instead).
    base_experiment: str = ""  # Real production LIBERO recipe this inherits from before tiny/single-GPU overrides.
    fsdp_shard_size: int = 1
    model_channels: int = 128
    num_blocks: int = 4
    num_heads: int = 4
    checkpoint_load_path: str = ""  # Empty skips loading the released 2B checkpoint -- trains from scratch.
    inference_sigma_max: float = 80.0
    inference_sigma_min: float = 4.0
    inference_job_group: str = "cosmos_v2_inference"

    # --- torchrun / kd_live / build_distill_dataset (raw LIBERO data) ---
    # One or more LIBERO suite directories (each relative to data_root unless it starts with "/"),
    # e.g. [libero_object_regen] or [libero_object_regen, libero_goal_regen]. A single entry is
    # passed straight through as data_dir; multiple entries get merged via a symlink dir under
    # output_root/dataset_combos/ (see submit_sweep.py's resolve_dataset_dir) --
    # LIBERODataset.data_dir only accepts one directory, but its file scan follows symlinks, so a
    # directory of per-suite symlinks reads as their union.
    suites: List[str] = field(default_factory=list)
    rollout_data_dir: str = ""
    # Restricts `suites`/`rollout_data_dir` to files whose name contains one of these (case-
    # insensitive substring match, e.g. ["ketchup"]) -- see LIBERODataset's own task_names
    # docstring (datasets/libero_dataset.py). Empty (default) loads every task, unchanged from
    # before. Lets a run train/build against a subset of a suite (down to one task) without a
    # separate physical directory of copied/symlinked files -- point `suites` at the real suite(s)
    # as usual and use this to pick which of their files actually get read.
    task_names: List[str] = field(default_factory=list)

    # --- torchrun / kd_live / kd_static ---
    max_iter: int = MISSING
    batch_size: int = MISSING  # build_distill_dataset: query batch size only, see its own module docstring

    # --- torchrun path only (the plain Trainer runs, e.g. baseline_*) ---
    # Dataloader worker processes. The imaginaire configs default this to 0 -- all HDF5 read + JPEG
    # decode + augmentation runs serially in the training process's main thread. Fine when that
    # thread gets a full CPU core; on a contended node (Vulcan packs multiple jobs per node) it
    # starves and the GPU sits idle -- a real run did ~175 s/iter at 0% GPU util this way
    # (experiment_journal.txt 2026-09-02). >0 parallelizes decode across the job's
    # --cpus-per-task; persistent_workers is turned on automatically when this is >0.
    num_workers: int = 0
    # Copy the resolved dataset dir to $SLURM_TMPDIR (node-local NVMe) at job start and train from
    # the copy -- kills the per-batch /project round-trip and makes the run immune to shared-FS
    # contention. Cheap for the LIBERO suites (~3 GB); do NOT enable for a data_dir of 100s of GB.
    stage_data_to_tmpdir: bool = False

    # --- kd_live / kd_static ---
    log_every: int = 10
    # kd_live overwrites model.pt/train_state.pt in place at this cadence (only the latest ever
    # matters for its own resume). kd_static instead KEEPS every checkpoint at this cadence under
    # checkpoints/iter_NNNNNNNNN/ -- no eval during training, so a separate kd_static_eval run needs
    # every one still on disk to sim-eval independently (see checkpoint_io.py's module docstring).
    checkpoint_every: int = 500
    lr: float = 1.0e-4
    student_net_experiment_name: str = ""  # e.g. "cosmos_kd_student_1b_libero" (kd/net_experiments.py)
    student_init_path: str = ""  # output of kd/init_student_from_teacher.py's model.pt
    student_device: str = "cuda:1"  # kd_static overrides this to "cuda:0" in its own conf/runs/*.yaml -- no
    # teacher ever gets loaded on that path, so there's no second accelerator to keep clear of.
    # See batch_prep.py's combined_kd_loss: linearly interpolates the student's loss between the
    # teacher's own prediction (0.0) and the real ground truth (1.0). 0.0 reproduces the original
    # KD-only design exactly.
    ground_truth_loss_weight: float = 0.0
    seed: int = 0

    # --- kd_live / build_distill_dataset ---
    teacher_experiment_name: str = ""  # empty -> teacher_loader.py's own TEACHER_EXPERIMENT_NAME default
    teacher_checkpoint: str = ""  # empty -> teacher_loader.py's own DEFAULT_TEACHER_CHECKPOINT default

    # --- kd_live only ---
    teacher_device: str = "cuda:0"

    # --- kd_static / build_distill_dataset ---
    # kd_static reads a dataset from here; build_distill_dataset writes one here -- same field,
    # since it's literally the same directory serving both roles across two different runs (a
    # "build" run and whichever "kd_static" run(s) train from what it produced).
    distill_dataset_dir: str = ""

    # --- kd_live(_av/_action) / kd_static(_av/_action) / build_synthetic_distill_dataset ---
    # kd_live*/kd_static* read an OPTIONAL second, always-distill-only loss term from here (see
    # e.g. train_kd.py's own module docstring for the two-term total loss); "" = disabled, same
    # convention as distill_dataset_dir/rollout_data_dir. build_synthetic_distill_dataset writes
    # here -- same dual-role field as distill_dataset_dir above, deliberately NEVER that same
    # field: a synthetic example has no real ground truth, so it can't be trained against the way
    # train_kd_static.py's real-data shards are (see build_synthetic_distill_dataset.py's own
    # module docstring for why it's a separate pipeline from build_distill_dataset entirely, not a
    # mode of it).
    synthetic_dataset_dir: str = ""

    # --- build_distill_dataset / build_synthetic_distill_dataset ---
    num_batches: int = MISSING  # raw batches drawn from LIBERODataset
    examples_per_shard: int = 256
    device: str = "cuda:0"  # exactly one GPU, no teacher/student split

    # --- build_distill_dataset only ---
    noise_draws_per_batch: int = 8

    # --- build_synthetic_distill_dataset / build_teacher_native_distill_dataset ---
    num_denoising_steps: int = 1  # matches real eval's own num_denoising_steps_future_state/
    # num_denoising_steps_value defaults (both 1) -- see synthetic_generation.py's module docstring.

    # --- build_synthetic_distill_dataset only ---
    action_perturbation_std: float = 0.05  # same clipped-Gaussian formula synthetic_generation.py's
    # perturb_action applies to every selected example's real action before querying the teacher.

    # --- build_teacher_native_distill_dataset only ---
    # Where build_teacher_native_distill_dataset.py writes its shards -- same dual-role-field
    # pattern as distill_dataset_dir/synthetic_dataset_dir above, but its own field, not a reuse of
    # synthetic_dataset_dir: a training run wanting to distill against these teacher-native examples
    # (instead of, or as a comparison against, the counterfactual ones) points its OWN
    # synthetic_dataset_dir at whichever build output it wants -- the two build pipelines' outputs
    # are drop-in interchangeable there (identical ShardWriter schema), so no new training-side field
    # is needed, only this one for the BUILD run's own output path.
    teacher_native_dataset_dir: str = ""

    # --- kd_static_eval only --- (see kd/periodic_libero_eval_static.py's module docstring)
    monitored_job_name: str = ""  # job_name of the kd_static run whose checkpoints/ to watch.
    eval_task_suite_name: str = ""  # e.g. "libero_object" -- must be one of LIBERO_{SPATIAL,OBJECT,GOAL,10}.
    eval_num_trials_per_task: int = 3  # Every task in eval_task_suite_name is evaluated, unless eval_task_keyword restricts to one.
    # Restricts eval to the single task in eval_task_suite_name whose language description
    # contains this (case-insensitive, e.g. "ketchup") -- see periodic_libero_eval_static.py's
    # --task_keyword. Empty (default) evaluates every task, unchanged from before.
    eval_task_keyword: str = ""
    # Only evaluate checkpoints whose iteration is a multiple of this (e.g. 100) -- see
    # periodic_libero_eval_static.py's --eval_every_n_iters. None (default) evaluates every
    # checkpoint that lands, per checkpoint_save_iter at train time.
    eval_every_n_iters: Optional[int] = None
    eval_inference_experiment: str = ""  # e.g. "cosmos_kd_student_500m_libero__inference_only" (net_experiments.py)
    eval_dataset_stats_path: str = ""
    eval_t5_text_embeddings_path: str = ""
    eval_poll_seconds: float = 30.0
    # "kd_static" (default, unchanged behavior): checkpoints are train_kd_static.py's own flat
    # iter_NNNNNNNNN/model.pt. "dcp": checkpoints are the standard torchrun/DistributedCheckpointer
    # sharded-directory format (iter_NNNNNNNNN/model/{.metadata,__0_0.distcp}) that a plain
    # non-KD torchrun run (e.g. baseline_1b_train) produces -- see periodic_libero_eval_static.py's
    # run_eval() for how this changes the --ckpt_path passed to run_libero_eval.py.
    eval_checkpoint_format: str = "kd_static"

    # --- sbatch resources, every run_type ---
    time_limit: str = MISSING  # sbatch --time
    mem: str = MISSING  # sbatch --mem
    cpus: int = MISSING  # sbatch --cpus-per-task
    account: str = MISSING  # sbatch --account
    filename: str = ""  # Output .sbatch filename; defaults to f"{job_name}.sbatch" if empty.
    # GPUs per job. kd_live needs 2 (teacher + student on cuda:0/cuda:1); every other run_type
    # needs 1. The torchrun path also passes this as --nproc_per_node, so gpus>1 there launches
    # that many DDP ranks (all other run_types are single-process and just get the extra devices).
    gpus: int = 1
    # Slurm GRES GPU type -- the "<type>" in "--gres=gpu:<type>:<gpus>". Cluster-specific
    # (rrg-gberseth: "h100"; Vulcan / aip-courvill: "l40s"); no longer hardcoded in
    # submit_sweep.py. Defaults from $COSMOS_POLICY_GPU_TYPE (exported by activate_cuda_env.sh
    # alongside COSMOS_POLICY_STORAGE/ACCOUNT) so switching clusters needs no per-run yaml edit;
    # still overridable per run in conf/runs/*.yaml or on the CLI (runs.<name>.gpu_type=...).
    gpu_type: str = field(default_factory=lambda: os.environ.get("COSMOS_POLICY_GPU_TYPE", "l40s"))


@dataclass
class LaunchOptions:
    # Which run(s) (keys of `runs` below) to process. Empty/unset means all of them.
    only: Optional[List[str]] = None
    # Actually call `sbatch`. Default False: (re)write the .sbatch scripts and print what would be
    # submitted, without touching the queue.
    submit: bool = False
    # Delete the selected run(s)' existing checkpoint dir (model/optim/scheduler/trainer shards,
    # latest_checkpoint.txt, DeviceMonitor, wandb offline run, loss CSV) before doing anything else,
    # so job.name auto-resume finds nothing and the run starts from iteration 0. Off by default --
    # this is a real rm -rf on real training artifacts. Only ever wipes the run(s) selected by
    # `only`, never everything, so e.g. `only=[smoketest] wipe=true` cannot touch train's checkpoints.
    wipe: bool = False


@dataclass
class SweepConfig:
    launch: LaunchOptions = field(default_factory=LaunchOptions)
    # Populated by conf/config.yaml's `runs@runs.<name>: <name>` defaults entries, not here -- see
    # conf/runs/*.yaml for the actual train/smoketest values.
    runs: Dict[str, RunConfig] = field(default_factory=dict)


cs = ConfigStore.instance()
cs.store(name="sweep_config", node=SweepConfig)
