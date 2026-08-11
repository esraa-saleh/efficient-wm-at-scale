# cosmos_distill_experiments

Stage 5 tiny-net LIBERO training job (real WAN2.1 tokenizer, real frame-replace layout, real
`HybridEDMSDE`, unmasked joint loss). All tunable values live in `conf/params.yaml` (net/checkpoint/
inference) and `conf/runs/*.yaml` (train/smoketest launch settings) -- edit those, not the `.py`
files, to change defaults.

`sbatch train.sbatch` / `sbatch smoketest.sbatch` need nothing sourced first -- the script activates
`.venv` and `activate_cuda_env.sh` itself once it's running on a compute node. Any `python -m ...`
command below runs on the login node in your current shell instead, so it needs `.venv` on the
interpreter -- either `source .venv/bin/activate` first, or call `.venv/bin/python` directly as
shown.

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

## Evaluate a checkpoint

```bash
.venv/bin/python -m cosmos_policy.experiments.robot.libero.run_libero_eval \
  --model_family cosmos \
  --config cosmos_dit_wm_tiny_libero_from_scratch__inference_only \
  --ckpt_path /project/rrg-gberseth/esraa1/cosmos_policy_storage/cosmos_dit_wm_output/cosmos_policy/cosmos_v2_finetune/cosmos_distill_experiments/checkpoints/iter_XXXXXXXXX \
  --task_suite_name libero_object \
  --dataset_stats_path /project/rrg-gberseth/esraa1/cosmos_policy_storage/LIBERO-Cosmos-Policy/success_only/libero_object_regen/dataset_statistics.json \
  --t5_text_embeddings_path /project/rrg-gberseth/esraa1/cosmos_policy_storage/LIBERO-Cosmos-Policy/success_only/t5_embeddings.pkl \
  --local_log_dir /project/rrg-gberseth/esraa1/cosmos_policy_storage/cosmos_dit_wm_output/cosmos_policy/cosmos_v2_finetune/cosmos_distill_experiments/eval
```

`--local_log_dir` isn't required (it defaults to `./experiments/logs`, wherever you happen to run
the command from) but is set here to keep eval logs/results next to everything else this job
produced instead of scattering them into a fourth location -- see Output layout below.

Unlike `submit_sweep.py` (pure Hydra/OmegaConf, no CUDA-touching imports), this one loads the actual
model for inference, so it needs `activate_cuda_env.sh` sourced too (not just `.venv`) and a GPU --
run it inside an `salloc`/`srun` session, the same way the interactive training command elsewhere in
this repo's docs is run.

## Output layout

Every output this job produces -- for a given `job.name` (`cosmos_distill_experiments` for
`train`, `cosmos_distill_experiments_smoketest` for `smoketest`) -- lands in one place in project
storage:

```
{output_root}/cosmos_policy/cosmos_v2_finetune/{job.name}/
├── checkpoints/iter_XXXXXXXXX/    # model/optim/scheduler/trainer shards + latest_checkpoint.txt
├── train_loss.csv
├── wandb/offline-run-.../         # WANDB_MODE=offline -- `wandb sync` to upload
├── DeviceMonitor/
├── config.yaml                    # resolved training config, dumped once at startup
├── slurm/{job.name}_<jobid>.{out,err}
└── eval/                          # only if you pass --local_log_dir as shown above
```

Both cases that would silently affect existing output require confirmation before proceeding --
even a dry run, since dry-run only skips the `sbatch` call, not these checks:
- A run whose output dir already has a checkpoint prints a `[WARNING]` (e.g. "will AUTO-RESUME from
  iter_000000005, not start from iteration 0") and requires typing `yes` before continuing.
- `launch.wipe=true` prints its own `[WARNING]` listing exactly what's about to be permanently
  deleted, and likewise requires typing `yes` before it actually removes anything.

Both prompts abort safely (nothing deleted, nothing resumed, nothing submitted) if there's no TTY
to prompt on -- so a script/cron invocation fails closed instead of hanging or silently proceeding.

`output_root` is `conf/paths.yaml`'s value. `launch.wipe=true` deletes this entire tree for the
selected run(s) -- including `slurm/`, so wipe before you've read a run's Slurm log, not after.
train.sbatch/smoketest.sbatch themselves are the one thing that does NOT live here -- they're
generated launcher code, so they stay in this folder regardless of what `paths.yaml` points at.

## Layout

- `conf/params.yaml` -- net/checkpoint/inference values (schema in `params.py`)
- `conf/paths.yaml` -- cluster storage paths (schema in `sweep.py`'s `PathsConfig`)
- `conf/runs/*.yaml` -- train/smoketest launch settings (schema in `sweep.py`)
- `experiment.py` -- Hydra experiment definition, registered via
  `../../config/experiment/cosmos_distill_experiments_registration.py`
- `submit_sweep.py` -- generates `train.sbatch` / `smoketest.sbatch` from `conf/`

Values are applied to the training config as a plain dict overlay, not a Hydra `defaults`-composed
structured config -- that's been tried twice and broke both times (see `params.py`'s docstring
before changing this).
