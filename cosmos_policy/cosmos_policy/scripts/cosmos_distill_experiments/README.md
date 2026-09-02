# cosmos_distill_experiments

Stage 5 tiny-net LIBERO training job (real WAN2.1 tokenizer, real frame-replace layout, real
`HybridEDMSDE`, unmasked joint loss). All tunable values live in `conf/runs/*.yaml` -- `train.yaml`
carries this job's net/checkpoint/inference architecture fields plus every run's launch settings --
edit those, not the `.py` files, to change defaults.

## Choosing LIBERO suites

`conf/runs/*.yaml`'s `suites` field is a list, e.g. `suites: [libero_object_regen]` or
`suites: [libero_object_regen, libero_goal_regen]` to train on their union (any of
`libero_object_regen`, `libero_goal_regen`, `libero_spatial_regen`, `libero_10_regen`). A single
suite is passed straight through; more than one gets merged via a directory of symlinks under
`{output_root}/dataset_combos/<suite1>+<suite2>/` (`LIBERODataset` only accepts one `data_dir`, but
its file scan follows symlinks, so a directory of per-suite symlinks reads as their union) --
`submit_sweep.py`'s `resolve_dataset_dir` builds/reuses it automatically, nothing to set up by hand.
`train.sbatch`/`smoketest.sbatch` need regenerating (via `submit_sweep.py`, not by hand) after
changing `suites`, since the resolved `data_dir` is baked into them at generation time.

`sbatch train.sbatch` / `sbatch smoketest.sbatch` need nothing sourced first -- the script activates
`.venv` and `activate_cuda_env.sh` itself once it's running on a compute node. Any `python -m ...`
command below runs on the login node in your current shell instead, so it needs `.venv` on the
interpreter -- either `source .venv/bin/activate` first, or call `.venv/bin/python` directly as
shown.

## Knowledge distillation (KD)

`kd/` is a separate job living in this same folder: distills the released 2B LIBERO teacher into a
smaller, depth-reduced student via output distillation (`combined_kd_loss` in `batch_prep.py`,
optionally blended with the real ground truth -- see `ground_truth_loss_weight` below) -- see
`kd/train_kd.py`'s module docstring for the full design. It's additive to everything above:
`experiment.py`'s tiny-net-from-scratch job is untouched, and KD runs use a different launch path
(`run_type: kd_live` set on the run, see `sweep.py`'s `RunConfig`) rather than the torchrun/Trainer
path `train`/`smoketest` use. Every hyperparameter any run needs -- torchrun, KD, or the static-KD
path below -- lives directly in that run's own `conf/runs/<name>.yaml` entry; there's no separate
KD-only config file to keep in sync with it (`submit_sweep.py` generates each KD run's actual
params file from those same fields, fresh, every time it runs).

Every `model.training_step(...)`/`model.denoise(...)` call in `kd/` runs inside
`batch_prep.policy_autocast()` (`torch.autocast(device_type="cuda", dtype=torch.bfloat16)`) --
required because these scripts explicitly skip FSDP (`load_model_from_checkpoint(...,
enable_fsdp=False)`), and it's FSDP's mixed-precision wrapping that normally auto-casts
forward-pass inputs for the real torchrun/Trainer path. Without it, `condition.padding_mask` (a
float32 tensor the model never bf16-converts, since it's built fresh per batch, not part of any
checkpoint) reaches a bf16 Linear layer and fails with `RuntimeError: expected mat1 and mat2 to
have the same dtype` -- confirmed against the real released teacher checkpoint. See
`batch_prep.py`'s `policy_autocast()` docstring.

### Choosing a KD student size

Student width/attention/heads stay fixed at the teacher's own values (`model_channels=2048,
num_heads=16`) -- only transformer depth (`num_blocks`) is reduced, per `kd/net_experiments.py`'s
module docstring. Run:

```bash
.venv/bin/python -m cosmos_policy.scripts.cosmos_distill_experiments.kd.student_sizes
```

(needs a GPU node -- `minimal_v4_dit.py` imports `transformer_engine` at module level, which needs
`activate_cuda_env.sh`'s CUDA toolchain to import at all, even though the param count itself is
computed on a meta-device net, not real GPU memory) to print a `num_blocks -> param count` table,
then update `kd/net_experiments.py`'s `STUDENT_SIZES` dict if the registered `1b`/`700m`/`500m`
presets need adjusting.

### One-time prerequisite: teacher-derived student init

Before launching `kd_train`/`baseline_1b_train`, build the student's initial weights (teacher
block weights copied via the layer map, no projection needed since block shapes match exactly):

```bash
.venv/bin/python -m cosmos_policy.scripts.cosmos_distill_experiments.kd.init_student_from_teacher \
  --student_size 1b --out $COSMOS_POLICY_STORAGE/kd_inits/student_init_1b.pt
```

CPU is enough (no GPU/Slurm needed) -- this is a one-time weight copy, not training. The output
path must match `conf/runs/kd_train.yaml`'s `student_init_path` field (and
`baseline_1b_train.yaml`'s `checkpoint.load_path`, if launching that comparison run). `700m`/`500m`
are also registered presets (`kd/net_experiments.py`'s `STUDENT_SIZES`) if you want a smaller
student for a one-off run -- override `runs.kd_train.student_net_experiment_name`/
`student_init_path` on the command line rather than editing `kd_train.yaml` itself for a one-off.

### Launch (KD)

Edit `conf/runs/kd_train.yaml` directly for any hyperparameter change (student size, lr, batch
size, `action_perturbation_prob`/`std`, `ground_truth_loss_weight`, ...), then:

```bash
.venv/bin/python -m cosmos_policy.scripts.cosmos_distill_experiments.submit_sweep launch.only=[kd_train] launch.submit=true
```

Runs on 2 H100s (`teacher_device`/`student_device` default to `cuda:0`/`cuda:1`), as a plain
`python -m ...kd.train_kd` process -- not torchrun, since `train_kd.py` places the frozen teacher
and trainable student on two explicit devices itself rather than relying on FSDP/data-parallel
sharding.

### Test first (KD)

No separate smoketest config for this path -- override `kd_train`'s own fields for a quick,
disposable sanity run instead of maintaining a second file just to shrink a few numbers:

```bash
.venv/bin/python -m cosmos_policy.scripts.cosmos_distill_experiments.submit_sweep \
  launch.only=[kd_train] launch.wipe=true launch.submit=true \
  runs.kd_train.job_name=cosmos_distill_experiments_kd_smoketest \
  runs.kd_train.max_iter=1 runs.kd_train.batch_size=2 runs.kd_train.log_every=1 \
  runs.kd_train.checkpoint_every=1 runs.kd_train.student_net_experiment_name=cosmos_kd_student_500m_libero \
  runs.kd_train.student_init_path=$COSMOS_POLICY_STORAGE/kd_inits/student_init_500m.pt \
  runs.kd_train.filename=kd_smoketest.sbatch
```

Overriding `job_name` gives it its own checkpoint dir/loss CSV, separate from the real `kd_train`
run -- `launch.wipe=true` here only ever touches that overridden job_name's output, never the real
run's. `--student_size 500m` needs its own one-time init (see above) before this launches.

Also run the unit/integration test suite (some tests need a real GPU and real cluster LIBERO data,
so run these on a compute node, not the login node):

```bash
pytest --import-mode=importlib cosmos_policy/scripts/cosmos_distill_experiments/kd/ -v
```

`--import-mode=importlib` is required, not optional, here: the inner `cosmos_policy` package has no
`__init__.py` at its own root (a namespace package), so pytest's default import mode walks up from
each test file, stops there, and prepends that directory to `sys.path` -- which shadows the
third-party `tokenizers` package with this repo's own `cosmos_policy/tokenizers/` submodule and
breaks any test that imports `transformers`/`peft` transitively (as `model_loader.py` does).
`--import-mode=importlib` resolves modules by dotted name instead, sidestepping this entirely.

(`@pytest.mark.L1` labels the expensive/integration tests, matching `_src/predict2/tests/training_loss_test.py`'s own convention -- this repo checkout has no `--L1`-gated skip logic locally, so all tests just run every time `pytest` is invoked here, marker or not.)

### Static KD: precompute the teacher's answers once, train from them repeatedly

`kd_train` above is the *live* path: the teacher gets queried fresh every single iteration of every
single run. If you're comparing several student configs (size, lr, ...) against the same fixed data
distribution, that means re-paying the full 2B teacher's forward-pass cost once per config --
wasteful when the teacher's answers for a given input don't change between runs. The static path
splits this into two steps instead:

**1. Build the dataset once** (`run_type: build_distill_dataset`, see
`conf/runs/datasets/build_distill_dataset_std_0.05.yaml` and its `_std_0.1.yaml` sibling -- two
otherwise-identical builds differing only in `action_perturbation_std`, for comparing perturbation
magnitude): queries the teacher `num_batches x noise_draws_per_batch` times and writes every
individual example to `distill_dataset_dir`, split apart (not stored as sealed groups) so training
can sample any example independently later -- see `kd/distill_dataset.py`'s module docstring for
the on-disk format and why splitting per-example is safe here (proven by a dedicated round-trip
test, not by inspection). Not a training run -- no checkpoint, no resume; re-running without
`launch.wipe=true` overwrites the existing shards from the start rather than continuing them.

```bash
.venv/bin/python -m cosmos_policy.scripts.cosmos_distill_experiments.submit_sweep \
  launch.only=[build_distill_dataset_std_0_05,build_distill_dataset_std_0_10] launch.submit=true
```

**2. Train (possibly several) students from it** (`run_type: kd_static`, `kd/train_kd_static.py`):
needs only ONE GPU -- no teacher is ever loaded on this path, since its work already happened in
step 1. Every training batch is `batch_size` individually, independently, uniformly sampled
examples from the dataset, reassembled via `join_examples_into_micro_batch` -- a fresh random mix
every iteration, not a fixed regrouping of whatever the builder happened to batch together.
`conf/runs/kd_static_500m_std_0_05.yaml` and its `_std_0_10.yaml` sibling are two such runs
(500m student, one per dataset variant above, otherwise identical -- `student_device: cuda:0`
since there's no teacher to keep a second accelerator clear of, `ground_truth_loss_weight`/`lr`/
`max_iter` matching `kd_train.yaml`'s own values for a fair live-vs-static comparison):

```bash
.venv/bin/python -m cosmos_policy.scripts.cosmos_distill_experiments.submit_sweep \
  launch.only=[kd_static_500m_std_0_05,kd_static_500m_std_0_10] launch.submit=true
```

Add a new `conf/runs/<name>.yaml` with `run_type: kd_static`, `distill_dataset_dir` pointing at
step 1's `distill_dataset_dir`, and `gpus: 1` for any other student size/config against these same
datasets, then launch the same way as any other run.

### Continuous full-suite eval, decoupled from training

`train_loss.csv` alone can't tell you a static-KD run has plateaued or started overfitting -- it's
training loss on the same data the run is training on, so it keeps improving even once real
generalization stalls (see the "I fear overfitting" discussion this section grew out of). The fix is
a genuine simulator signal -- but training itself never runs, triggers, or waits on any of it: no
eval happens in the training process, and nothing eval finds feeds back into training's control
flow. Training just trains and keeps every checkpoint; a separate job independently evaluates
whichever ones exist, on its own schedule, restart-safely.

- **`train_kd_static.py`** writes a versioned checkpoint (`checkpoint_io.save_versioned_checkpoint`)
  to `run_dir/checkpoints/iter_NNNNNNNNN/{model.pt,train_state.pt}` every `checkpoint_every`
  iterations -- and, unlike `train_kd.py`'s live path (which overwrites a single `model.pt`/
  `train_state.pt` in place), NEVER overwrites or deletes one: every checkpoint persists, since a
  separate eval job needs each one still on disk to evaluate independently. Its own resume logic
  reads `run_dir/checkpoints/latest_checkpoint.txt` (same convention the torchrun path already uses)
  to pick up from the most recent one. The only other thing it writes is `run_dir/TRAINING_DONE`, a
  plain completion marker (not an eval) as the very last thing it does on exit.
- **`kd/periodic_libero_eval_static.py`** (`run_type: kd_static_eval`) is a genuinely SEPARATE Slurm
  job -- not a subprocess of training, since the two may land on different nodes, so coordination is
  entirely file-based (shared storage), not PID-based. It polls `checkpoints/` for new ones and, for
  each, runs a FULL-SUITE eval (every task in `eval_task_suite_name`, `eval_num_trials_per_task`
  rollouts each -- real coverage, not a narrow cheap subset), appending every result to its own
  `eval_results.csv`. It stops polling once `TRAINING_DONE` exists and it has caught up on every
  checkpoint -- but is otherwise indefinite, so if training's `checkpoint_every` produces checkpoints
  faster than full-suite evals complete, this job may still be catching up long after training itself
  finishes; that's expected, not a bug.
- **Restart safety**: if this eval job is halted (preemption, walltime, manual cancel) and
  resubmitted, it does NOT start over -- `main()` reads its own `eval_results.csv` at startup and
  seeds which iterations are already evaluated from every row already logged there, before comparing
  against what's on disk. Resubmitting it is always safe and always makes forward progress.

`conf/runs/kd_static_500m_std_0_05_eval.yaml` and its `_std_0_10_eval.yaml` sibling are the eval
companions to the two training runs above (`monitored_job_name` ties each to its training run's
`job_name`). Launch a training run and its eval companion together -- the eval job just polls idly
until the first checkpoint exists, so submitting it early is harmless:

```bash
.venv/bin/python -m cosmos_policy.scripts.cosmos_distill_experiments.submit_sweep \
  launch.only=[kd_static_500m_std_0_05,kd_static_500m_std_0_05_eval] launch.submit=true
```

Cost to be aware of: at the default `checkpoint_every=100` over a 30000-iteration run, that's 300
checkpoints, each getting a full-suite eval (10 tasks x 3 trials = 30 episodes for `libero_object`)
-- this eval job will very likely need to be resubmitted more than once (safely, per the restart
guarantee above) to fully catch up, rather than finishing within one Slurm allocation. Raise
`checkpoint_every` if you'd rather trade checkpoint granularity for a shorter eval backlog.

## Launch

```bash
sbatch train.sbatch
```

Or regenerate + submit via the sweep script (picks up any `conf/` edits first):

```bash
.venv/bin/python -m cosmos_policy.scripts.cosmos_distill_experiments.submit_sweep launch.only=[train] launch.submit=true
```

Override any field without editing files, e.g. `runs.train.max_iter=5000`.

## Test first

```bash
sbatch smoketest.sbatch
```

1-iteration run, separate checkpoint dir/job name from `train`. Not idempotent by default (reuses
`job.name`, so it auto-resumes) -- force a clean run from iteration 0 with:

```bash
.venv/bin/python -m cosmos_policy.scripts.cosmos_distill_experiments.submit_sweep launch.only=[smoketest] launch.wipe=true launch.submit=true
```

## Resume

`train.sbatch` reuses `job.name=cosmos_distill_experiments` every time, so re-running it
auto-resumes from the latest checkpoint instead of retraining from scratch.

`kd_train.sbatch` resumes the same way, but from `train_state.pt` (not
`checkpoints/latest_checkpoint.txt` -- see Output layout below) since `train_kd.py` isn't going
through the Trainer's own checkpointer at all.

## Evaluate a checkpoint

```bash
.venv/bin/python -m cosmos_policy.experiments.robot.libero.run_libero_eval \
  --model_family cosmos \
  --config cosmos_dit_wm_tiny_libero_from_scratch__inference_only \
  --ckpt_path $COSMOS_POLICY_STORAGE/cosmos_dit_wm_output/cosmos_policy/cosmos_v2_finetune/cosmos_distill_experiments/checkpoints/iter_XXXXXXXXX \
  --task_suite_name libero_object \
  --dataset_stats_path $COSMOS_POLICY_STORAGE/LIBERO-Cosmos-Policy/success_only/libero_object_regen/dataset_statistics.json \
  --t5_text_embeddings_path $COSMOS_POLICY_STORAGE/LIBERO-Cosmos-Policy/success_only/t5_embeddings.pkl \
  --local_log_dir $COSMOS_POLICY_STORAGE/cosmos_dit_wm_output/cosmos_policy/cosmos_v2_finetune/cosmos_distill_experiments/eval
```

`--local_log_dir` isn't required (it defaults to `./experiments/logs`, wherever you happen to run
the command from) but is set here to keep eval logs/results next to everything else this job
produced instead of scattering them into a fourth location -- see Output layout below.

Unlike `submit_sweep.py` (pure Hydra/OmegaConf, no CUDA-touching imports), this one loads the actual
model for inference, so it needs `activate_cuda_env.sh` sourced too (not just `.venv`) and a GPU --
run it inside an `salloc`/`srun` session, the same way the interactive training command elsewhere in
this repo's docs is run.

### Evaluate a KD student

Same command, same pipeline -- no new eval code exists for KD students, just a different
`--config`/`--ckpt_path`:

```bash
.venv/bin/python -m cosmos_policy.experiments.robot.libero.run_libero_eval \
  --model_family cosmos \
  --config cosmos_kd_student_1b_libero__inference_only \
  --ckpt_path $COSMOS_POLICY_STORAGE/cosmos_dit_wm_output/cosmos_policy/cosmos_v2_finetune/cosmos_distill_experiments_kd_train/model.pt \
  --task_suite_name libero_object \
  --dataset_stats_path $COSMOS_POLICY_STORAGE/LIBERO-Cosmos-Policy/success_only/libero_object_regen/dataset_statistics.json \
  --t5_text_embeddings_path $COSMOS_POLICY_STORAGE/LIBERO-Cosmos-Policy/success_only/t5_embeddings.pkl \
  --local_log_dir $COSMOS_POLICY_STORAGE/cosmos_dit_wm_output/cosmos_policy/cosmos_v2_finetune/cosmos_distill_experiments_kd_train/eval
```

`--ckpt_path` here is `model.pt` (a bare state dict), not a `checkpoints/iter_XXXXXXXXX/` directory
-- `run_libero_eval.py`'s loader already handles a `.pt` path directly (see `kd/checkpoint_io.py`'s
module docstring). Before the sim eval, `kd/offline_eval.py` gives a much cheaper fidelity check
(denoiser MSE / decoded action error against the teacher on held-out data, no simulator needed):

```bash
.venv/bin/python -m cosmos_policy.scripts.cosmos_distill_experiments.kd.offline_eval \
  --student_experiment_name cosmos_kd_student_1b_libero \
  --student_checkpoint .../cosmos_distill_experiments_kd_train/model.pt \
  --data_dir $COSMOS_POLICY_STORAGE/LIBERO-Cosmos-Policy/success_only/libero_object_regen \
  --t5_text_embeddings_path $COSMOS_POLICY_STORAGE/LIBERO-Cosmos-Policy/success_only/t5_embeddings.pkl \
  --dataset_stats_path $COSMOS_POLICY_STORAGE/LIBERO-Cosmos-Policy/success_only/libero_object_regen/dataset_statistics.json
```

## Output layout

Every output this job produces -- for a given `job.name` (`cosmos_distill_experiments` for
`train`, `cosmos_distill_experiments_smoketest` for `smoketest`) -- lands in one place in project
storage:

```
{output_root}/cosmos_policy/cosmos_v2_finetune/{job.name}/
├── checkpoints/iter_XXXXXXXXX/    # model/optim/scheduler/trainer shards + latest_checkpoint.txt
├── train_loss.csv                 # loss logging is CSV-only -- wandb is disabled (WANDB_MODE=disabled
│                                  #   + job.wandb_mode=disabled, set by submit_sweep.py for every run)
├── DeviceMonitor/
├── config.yaml                    # resolved training config, dumped once at startup
├── slurm/{job.name}_<jobid>.{out,err}
└── eval/                          # only if you pass --local_log_dir as shown above
```

KD runs (`kd_train`, `kd_static`-type runs, and the `baseline_1b_train` comparison run) land in the
exact same `{output_root}/cosmos_policy/cosmos_v2_finetune/{job.name}/` tree (`job_output_dir()` is
reused unmodified), but the checkpoint layout underneath differs by run type -- see
`kd/checkpoint_io.py`'s module docstring:
- `kd_train` (`run_type: kd_live`) produces a single overwritten `model.pt` + `train_state.pt`
  directly under `run_dir` -- only the latest ever matters, since nothing else reads this run's
  checkpoints.
- `kd_static`-type runs instead produce `checkpoints/iter_NNNNNNNNN/{model.pt,train_state.pt}` +
  `checkpoints/latest_checkpoint.txt` -- the SAME layout as the torchrun path below, just via a
  different mechanism, since a separate `kd_static_eval` job needs every checkpoint kept, not just
  the latest (see "Continuous full-suite eval" above). It also gets a `TRAINING_DONE` marker,
  written once, as the last thing training does on exit.
- `baseline_1b_train` goes through the normal torchrun/Trainer path (it's a non-KD comparison run),
  so it produces the usual `checkpoints/iter_XXXXXXXXX/` layout for the same underlying reason.

All four KD-ish `run_type`s (`kd_live`/`kd_static`/`build_distill_dataset`/`kd_static_eval`) also
get a `params.yaml` here -- `submit_sweep.py`-generated fresh from `conf/runs/<name>.yaml` every
time, the actual `--kd_params`/`--static_params`/`--build_params` file that run used, kept for
provenance the same way `config.yaml` already is for the torchrun path (`kd_static_eval` is the one
exception -- plain CLI args, no generated params file, since it's a monitor rather than a training
run; see `kd/periodic_libero_eval_static.py`'s own module docstring). `build_distill_dataset` runs
are another exception to "real output lives under `job_output_dir()`" -- their real output (the
dataset shards) lives at that run's own `distill_dataset_dir` instead; only Slurm logs +
`params.yaml` land here.

The companion `kd_static_eval` job itself lands in its OWN separate `{job.name}_eval/` tree (its own
`monitored_job_name` field points back at the training run it watches, via `checkpoints/` there) --
`eval_results.csv` there, plus `run_libero_eval.py`'s own per-call results CSVs and Slurm logs, never
mixed into the training run's own tree. That CSV is also what makes the eval job restart-safe (see
"Continuous full-suite eval" above) -- it's read back at startup, not just written to.

One sibling directory lives outside any single job's tree, since it's shared across runs/jobs
rather than owned by one: `{output_root}/dataset_combos/<suite1>+<suite2>/` -- the multi-suite
symlink dirs described above under Choosing LIBERO suites. `launch.wipe=true` never touches it
(wipe only removes a `job.name`'s own tree). The teacher-derived student-init `.pt` files
(`kd/init_student_from_teacher.py`'s output) are similarly shared across runs rather than owned by
one job -- this repo's convention keeps them under `{output_root}/kd_inits/`.

Both cases that would silently affect existing output require confirmation before proceeding --
even a dry run, since dry-run only skips the `sbatch` call, not these checks:
- A run whose output dir already has a checkpoint prints a `[WARNING]` (e.g. "will AUTO-RESUME from
  iter_000000005, not start from iteration 0") and requires typing `yes` before continuing.
- `launch.wipe=true` prints its own `[WARNING]` listing exactly what's about to be permanently
  deleted, and likewise requires typing `yes` before it actually removes anything.

Both prompts abort safely (nothing deleted, nothing resumed, nothing submitted) if there's no TTY
to prompt on -- so a script/cron invocation fails closed instead of hanging or silently proceeding.

`output_root` is that run's own `conf/runs/*.yaml` value (a plain `RunConfig` field, same across
every run in practice since they share one cluster account). `launch.wipe=true` deletes this entire
tree for the selected run(s) -- including `slurm/`, so wipe before you've read a run's Slurm log,
not after. train.sbatch/smoketest.sbatch themselves are the one thing that does NOT live here --
they're generated launcher code, so they stay in this folder regardless of what `output_root` points at.

## Layout

- `conf/runs/*.yaml` -- every launchable run's full settings, including cluster-storage paths
  (`data_root`/`output_root`) and, for `train.yaml` specifically, this job's net/checkpoint/
  inference architecture fields too (schema in `sweep.py`'s `RunConfig`) -- `train`/`smoketest`/
  `baseline_1b_train` (`run_type: torchrun`, the default), `kd_train` (`run_type: kd_live`),
  `kd_static_500m_std_0_05`/`_std_0_10` (`run_type: kd_static`), and their
  `_eval.yaml` continuous-eval companions (`run_type: kd_static_eval`, see "Continuous full-suite
  eval" above). This is the ONE place any run's hyperparameters get edited -- KD/architecture/path
  values all live directly as `RunConfig` fields, not in a separate config file.
- `conf/runs/datasets/*.yaml` -- one-shot dataset builds (`run_type: build_distill_dataset`), same
  `RunConfig` schema and `runs@runs.<name>` composition mechanism as `conf/runs/*.yaml`, just kept
  in their own subfolder since they build data rather than train anything -- `build_distill_dataset`.
- `experiment.py` -- Hydra experiment definition, registered via
  `../../config/experiment/cosmos_distill_experiments_registration.py`
- `submit_sweep.py` -- generates every run's `.sbatch` script from `conf/`, branching on
  `run.run_type`; for the three KD-ish training run types, also generates that run's own
  `--kd_params`/`--static_params`/`--build_params` file (written to `{run_dir}/params.yaml`, fresh
  every time, for provenance) directly from `RunConfig`'s fields. `kd_static_eval` is plain CLI args
  instead (see `kd/periodic_libero_eval_static.py`'s own module docstring for why -- it's a monitor,
  not a training run with hyperparameters worth a provenance file).
- `kd/` -- the KD job itself (see "Knowledge distillation (KD)" above): `net_experiments.py`
  (registers the `cosmos_kd_student_{1b,700m,500m}_libero` experiments), `student_sizes.py`
  (param-count calibration), `init_student_from_teacher.py` (one-time teacher -> student weight
  copy), `train_kd.py` (the bespoke 2-GPU live-KD training loop, plain Python -- not
  torchrun/FSDP), `train_kd_static.py` (the static-KD training loop, 1 GPU, no teacher loaded),
  `build_distill_dataset.py` (builds a static-KD dataset once, see `distill_dataset.py`'s on-disk
  format), `periodic_libero_eval_static.py` (the continuous full-suite eval companion, see
  "Continuous full-suite eval" above), `params.py` (`KDParams`/`StaticTrainParams`/
  `BuildDistillDatasetParams` schemas -- loaded from whatever `submit_sweep.py` generated, not a
  hand-maintained file), `checkpoint_io.py` (`model.pt`/`train_state.pt` save/load -- both the
  overwritten-in-place scheme `train_kd.py` uses and the versioned, nothing-ever-deleted scheme
  `train_kd_static.py` uses), `offline_eval.py` (fidelity diagnostics), `batch_prep.py` (cross-device
  condition move, `combined_kd_loss`), `synthetic_generation.py` (two standalone synthetic data
  pipelines, both feeding the existing distill_dataset.py `ShardWriter` format -- see its own module
  docstring for how they relate: `generate_synthetic_targets`/`build_synthetic_distill_dataset.py`
  perturbs the recorded action and holds it fixed as conditioning; `generate_teacher_native_targets`/
  `build_teacher_native_distill_dataset.py` lets the teacher generate its own action, jointly with
  future state and value, nothing held fixed),
  `teacher_loader.py` (shared model-loading helper), and colocated `*_test.py` files (`pytest
  cosmos_policy/scripts/cosmos_distill_experiments/kd/ -v` -- `@pytest.mark.L1` labels the
  expensive/GPU/data-gated ones, but nothing locally skips them without it).

`train.yaml`'s architecture fields (the torchrun job's net/checkpoint config, read via `params.py`'s
`load_params()`) are applied as a plain dict overlay, not a Hydra `defaults`-composed structured
config -- that's been tried twice and broke both times (see `params.py`'s docstring before changing
this). `sweep.py`'s `RunConfig`, by contrast, is a flat dataclass with plain-value fields (never
composed as a structured sub-config), which doesn't hit that failure mode -- that's what makes
folding every run type's hyperparameters (and cluster paths) directly into it safe.
