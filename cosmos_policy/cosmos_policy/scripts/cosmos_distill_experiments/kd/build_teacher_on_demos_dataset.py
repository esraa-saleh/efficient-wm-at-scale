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
Builds the "teacher_on_demos" sidecar dataset: for every timestep of every demo in a LIBERO suite,
ask the teacher (``nvidia/Cosmos-Policy-LIBERO-Predict2-2B``) what IT would do from that exact
state -- its own freely-generated action chunk plus the future image / future proprio / value it
imagines following that action -- and store the answer keyed to (source episode, timestep).

This is the read-through half of the ``teacher_on_demos_{500m,1b}_train`` experiment: those runs
are byte-for-byte copies of ``baseline_{500m,1b}_train`` (same data states, same
torchrun/Trainer path, same joint EDM objective, same init, same eval) EXCEPT the four joint
denoising targets come from this sidecar instead of the human demo. ``LIBERODataset`` consumes it
via its ``teacher_on_demos_dir`` kwarg (default "" -> untouched baseline behavior); nothing in the
training/loss path changes. See ``kd/TEACHER_ON_DEMOS_PLAN.md`` for the full design + the two data
transformation chains.

Deliberately NOT ``build_teacher_native_distill_dataset.py``: that writes the ShardWriter
``(xt, sigma, condition, teacher_x0, x0)`` KD format for ``train_kd_static.py``. This writes a
per-(episode, timestep) sidecar HDF5 tree that mirrors the source suite, consumed at
``__getitem__`` time by the plain torchrun path -- no ``kd_*`` run type. It DOES reuse the same
teacher query (``synthetic_generation.generate_teacher_native_targets``) and teacher loader.

How the states are sourced (all deliberate deviations from ``batch_prep.DATASET_KWARGS``):
  - ``rollout_data_dir=""``            -- demos only; the sidecar is demo-keyed, and the baseline
                                          training runs also leave rollout data unset.
  - ``use_image_aug=False``,
    ``use_stronger_image_aug=False``   -- the teacher must condition on CLEAN current obs so the
                                          targets are deterministic and resumable. At *training*
                                          time the student's current-obs pipeline keeps
                                          ``use_image_aug=True`` (standard KD: aug as student-side
                                          regularization).
  - ``demonstration_sampling_prob=1``  -- cosmetic; with ``rollout_data_dir=""`` the epoch is
                                          all demo steps regardless.
NORMALIZATION (this build's trickiest detail -- read before changing anything here): the teacher
GENERATES its action / future-proprio latent in ITS OWN ``dataset_statistics.json`` convention
(the one it was finetuned with: ``<teacher_checkpoint>/libero_dataset_statistics.json``), which
differs from the local suite's ``libero_object_regen/dataset_statistics.json`` -- NVIDIA's file is
pooled across all four LIBERO suites, ours is computed from just this one's replay-success subset.

Two earlier versions of this build got this only half right (see experiment_journal.txt
2026-09-03/04 for the full story):
  1. Stored the teacher's action/future_proprio verbatim (teacher-normalized) while the student
     trains + ``run_libero_eval`` unnormalizes with the LOCAL suite's stats -- a normalize-with-X,
     unnormalize-with-Y mismatch on the one dimension family (rotation) where the two files
     actually disagree. Result: wrist rotation came out scaled toward zero (yaw ~0.2x the demo's
     magnitude), 0/3 closed-loop despite matched training loss.
  2. Fixed action's magnitude with a per-dim raw-unit round-trip (teacher-norm -> raw -> local-
     norm), which is valid for action deltas (a shared control convention, just a wider *observed*
     range once pooled across suites) -- but applied the SAME round-trip to future_proprio, which
     is an ABSOLUTE end-effector/joint position: different suites' tasks sit in different regions
     of that space, so libero_object's much narrower slice of NVIDIA's pooled range doesn't cover
     where the teacher's own proprio values actually land. Result: future_proprio blew up to ~6x
     outside [-1,1] on some dims (100% of one dim's states exceeded |1|) -- a corrupted training
     target in the same joint transformer as the action, not a fix.

This version sidesteps per-field renormalization entirely by putting the WHOLE build -- the
teacher's conditioning input (current proprio) as well as its output (action, future_proprio) --
in the teacher's OWN normalization, via ``LIBERODataset(dataset_stats_override_path=<teacher's
libero_dataset_statistics.json>)``. Consequences, all deliberate:
  - The teacher is now conditioned on CORRECTLY-scaled current proprio (previously it was fed
    local-normalized proprio while expecting its own convention -- a second, independent scale
    bug on the INPUT side that neither earlier version caught).
  - action_chunks / future_proprio are stored VERBATIM -- no renorm, no round-trip, nothing that
    can silently blow up on a dimension where the two files diverge.
  - The matching training run (``teacher_on_demos_500m_train``) must ALSO construct its
    ``LIBERODataset`` with this same ``dataset_stats_override_path`` (see that run's yaml /
    launch override), so the student's current proprio -- read straight from the demo, not
    overridden by this sidecar -- is in the SAME convention as the future_proprio target it's
    trained against. Its eval companion must point ``--dataset_stats_path`` at the same teacher
    stats file, so the student's predicted action un-normalizes correctly.
  - This makes the run's loss values NOT numerically comparable to the demo baseline's (different
    latent scale) -- irrelevant to the actual comparison, which is closed-loop success rate
    (simulator task completion), a convention-agnostic, physical-units measurement.
``value`` is a Monte-Carlo return, not a min/max-rescaled quantity -- unaffected either way, still
just the existing [0,1]->[-1,1] shift.

MULTIPLE TEACHER SAMPLES (``num_teacher_samples`` / ``--build_params``' ``num_teacher_samples``):
a single diffusion sample per state is a noisy point estimate of the teacher's implied
distribution over "what happens next" -- the demo<->teacher correlation on the rotation channels
(0.68-0.80) is exactly this per-sample noise (see experiment_journal.txt 2026-09-03/04). Setting
``num_teacher_samples = K > 1`` draws K INDEPENDENT samples per (episode, timestep) -- same
conditioning, different initial diffusion noise per k (``arch_invariant_rand`` is seeded purely by
the ``seed`` argument to ``generate_samples_from_batch``, nothing else, so distinct seeds are
guaranteed distinct noise draws) -- and stores all K, UNAVERAGED, as a leading K axis on every
field. ``LIBERODataset._load_teacher_on_demos_sidecar`` / ``__getitem__`` then picks one k
uniformly at random each time a state is sampled (same k across all four fields for one call,
since one k is one coherent generation). This is a deliberate choice over pre-averaging the K
samples into a single target: the student is a DIFFUSION model, built to score-match against a
possibly-multimodal target distribution -- collapsing K samples to their mean before training ever
sees them would throw that away, and could even produce an invalid target (the mean of two
genuinely different valid wrist orientations isn't necessarily itself a valid grasp). Cost is
linear in K (K independent ``generate_teacher_native_targets`` calls per batch); storage grows K x
on the two JPEG image fields (the dominant size cost). ``num_teacher_samples = 1`` (default) is
today's behavior, byte-identical except for the added (now size-1) K axis.

Resumable at PER-EPISODE granularity: each episode's HDF5 group is appended to the sidecar (with
``episode_complete=True`` + ``flush()``) the moment its every timestep is generated, so a
preemption / timeout / crash loses at most the one in-flight episode. A rerun opens the partial
sidecar, skips groups already marked ``episode_complete``, and regenerates only the rest;
``--force`` deletes the sidecar and starts over. A sidecar with every episode present gets root
attr ``complete=True`` (``_finalize_sidecar``); ``_is_complete`` then skips the whole file on a
later run.

Sharding: ``--task_names ketchup`` restricts to matching task files -- the recommended way to
split the ~60k-timestep libero_object build into ~1h per-task jobs, each writing its own
independent sidecar file.

Usage:
    python -m cosmos_policy.scripts.cosmos_distill_experiments.kd.build_teacher_on_demos_dataset \\
        --build_params <path/to/build_params.yaml> --data_dir <suite dir> \\
        --t5_text_embeddings_path <.../t5_embeddings.pkl> --out_dir <sidecar root>
"""

import argparse
import datetime as _dt
import os
import subprocess
from collections import OrderedDict

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from cosmos_policy.constants import ACTION_DIM, NUM_ACTIONS_CHUNK, PROPRIO_DIM
from cosmos_policy.datasets.libero_dataset import LIBERODataset
from cosmos_policy.experiments.robot.cosmos_utils import resolve_path
from cosmos_policy.scripts.cosmos_distill_experiments.kd import batch_prep
from cosmos_policy.scripts.cosmos_distill_experiments.kd import params as kd_params
from cosmos_policy.scripts.cosmos_distill_experiments.kd.synthetic_generation import generate_teacher_native_targets
from cosmos_policy.scripts.cosmos_distill_experiments.kd.teacher_loader import load_teacher
from cosmos_policy.utils.utils import jpeg_encode_image

_SIDECAR_SUFFIX = ".teacher_on_demos"
# Same construction as the baseline's libero_all_4_suites_dataset
# (config/experiment/cosmos_policy_experiment_configs.py), minus the deviations documented in this
# module's docstring.
_DATASET_KWARGS = dict(
    chunk_size=NUM_ACTIONS_CHUNK,
    use_wrist_images=True,
    use_third_person_images=True,
    use_proprio=True,
    normalize_proprio=True,
    normalize_actions=True,
    num_duplicates_per_image=4,
    return_value_function_returns=True,
    gamma=0.99,
    use_image_aug=False,
    use_stronger_image_aug=False,
    demonstration_sampling_prob=1.0,
    success_rollout_sampling_prob=0.5,
)


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def sidecar_path_for(source_file_path: str, data_dir: str, out_dir: str) -> str:
    """Mirror the source file's path relative to ``data_dir`` under ``out_dir``, inserting
    ``.teacher_on_demos`` before the extension. ``LIBERODataset._load_teacher_on_demos_sidecar``
    reconstructs this exact path from ``(file_path, data_dir, teacher_on_demos_dir)``."""
    rel = os.path.relpath(source_file_path, data_dir)
    base, ext = os.path.splitext(rel)
    return os.path.join(out_dir, base + _SIDECAR_SUFFIX + ext)


def _is_complete(sidecar_path: str) -> bool:
    if not os.path.exists(sidecar_path):
        return False
    try:
        with h5py.File(sidecar_path, "r") as f:
            return bool(f.attrs.get("complete", False))
    except OSError:
        return False


def _new_buffer(num_steps: int, num_teacher_samples: int) -> dict:
    K = num_teacher_samples
    return dict(
        action_chunks=np.zeros((num_steps, K, NUM_ACTIONS_CHUNK, ACTION_DIM), dtype=np.float32),
        future_proprio=np.zeros((num_steps, K, PROPRIO_DIM), dtype=np.float32),
        value=np.zeros((num_steps, K), dtype=np.float32),
        future_image_jpeg=[[None] * K for _ in range(num_steps)],
        future_wrist_image_jpeg=[[None] * K for _ in range(num_steps)],
        filled=0,
    )


def _completed_episodes(sidecar_path: str) -> set:
    """demo_keys already fully written to an existing (possibly partial) sidecar -- i.e. groups
    whose episode_complete=True. Used for per-episode resume: these are not regenerated."""
    if not os.path.exists(sidecar_path):
        return set()
    try:
        with h5py.File(sidecar_path, "r") as f:
            return {k for k in f if bool(f[k].attrs.get("episode_complete", False))}
    except OSError:
        return set()


def _write_episode_group(sidecar_path: str, meta: dict, demo_key: str, buf: dict) -> None:
    """Append ONE episode's group to the sidecar (create the file + write meta on the first call).
    Incremental: called as soon as an episode's every timestep is generated, so a crash/preemption
    loses at most the one in-flight episode. Idempotent -- replaces an existing partial group.
    episode_complete=True + flush() is done LAST, so a crash mid-write leaves the group without
    that flag and _completed_episodes() will not skip it on resume."""
    os.makedirs(os.path.dirname(sidecar_path) or ".", exist_ok=True)
    assert buf["filled"] == buf["action_chunks"].shape[0], (
        f"{sidecar_path}::{demo_key}: filled {buf['filled']}/{buf['action_chunks'].shape[0]} timesteps"
    )
    vlen = h5py.vlen_dtype(np.dtype("uint8"))
    num_teacher_samples = buf["action_chunks"].shape[1]
    with h5py.File(sidecar_path, "a") as f:
        if "created_utc" not in f.attrs:
            for k, v in meta.items():
                f.attrs[k] = v
            f.attrs["complete"] = False
        if demo_key in f:
            del f[demo_key]  # replace a partial group left by an earlier crash
        g = f.create_group(demo_key)
        g.create_dataset("action_chunks", data=buf["action_chunks"])  # (T,K,chunk,7)
        g.create_dataset("future_proprio", data=buf["future_proprio"])  # (T,K,9)
        g.create_dataset("value", data=buf["value"])  # (T,K)
        di = g.create_dataset("future_image_jpeg", (buf["filled"], num_teacher_samples), dtype=vlen)
        dwi = g.create_dataset("future_wrist_image_jpeg", (buf["filled"], num_teacher_samples), dtype=vlen)
        for i in range(buf["filled"]):
            for k_idx in range(num_teacher_samples):
                di[i, k_idx] = buf["future_image_jpeg"][i][k_idx]
                dwi[i, k_idx] = buf["future_wrist_image_jpeg"][i][k_idx]
        g.attrs["num_steps"] = buf["action_chunks"].shape[0]
        g.attrs["num_teacher_samples"] = num_teacher_samples
        g.attrs["chunk_size"] = NUM_ACTIONS_CHUNK
        g.attrs["episode_complete"] = True
        f.flush()


def _finalize_sidecar(sidecar_path: str, all_demo_keys: list, complete: bool) -> None:
    """Mark a sidecar's source file done -- every one of its episodes is present. complete=False
    only when max_episodes truncated the file mid-way (LIBERODataset would then KeyError on an
    un-built episode; _is_complete() re-selects it for a full rebuild, and per-episode resume
    keeps the episodes it already has)."""
    with h5py.File(sidecar_path, "a") as f:
        f.attrs["demo_keys"] = sorted(all_demo_keys)
        f.attrs["complete"] = bool(complete)
        f.flush()


def build(
    *,
    teacher_checkpoint: str,
    teacher_experiment_name: str,
    data_dir: str,
    t5_text_embeddings_path: str,
    out_dir: str,
    task_names: list,
    batch_size: int,
    dataloader_num_workers: int,
    num_denoising_steps: int,
    num_teacher_samples: int,
    max_episodes: int,
    device: str,
    seed: int,
    force: bool,
) -> None:
    torch.manual_seed(seed)

    # The teacher must be conditioned on ITS OWN proprio convention (see module docstring's
    # NORMALIZATION section), and its generated action/future_proprio are stored verbatim in that
    # same convention -- resolve_path handles the HF-repo-id case (a cache hit: load_teacher below
    # already fetched the whole snapshot via snapshot_download).
    teacher_stats_path = resolve_path(f"{teacher_checkpoint}/libero_dataset_statistics.json")
    print(f"[norm] LIBERODataset stats override: {teacher_stats_path}", flush=True)

    dataset = LIBERODataset(
        data_dir=data_dir,
        t5_text_embeddings_path=t5_text_embeddings_path,
        rollout_data_dir="",
        task_names=task_names or None,
        dataset_stats_override_path=teacher_stats_path,
        **_DATASET_KWARGS,
    )
    assert len(dataset) == dataset.num_steps, (
        f"expected a demo-only epoch (len == num_steps), got {len(dataset)} vs {dataset.num_steps}"
    )

    # Group episodes by their source HDF5 file, preserving load order. self.data is inserted in
    # file-walk order with each file's demos contiguous, so a file's episodes form one run.
    files = OrderedDict()  # source_file_path -> [(episode_idx, demo_key, num_steps), ...]
    for episode_idx, ep in dataset.data.items():
        files.setdefault(ep["file_path"], []).append((episode_idx, ep["demo_key"], ep["num_steps"]))

    # max_episodes (smoketest / quick iteration): keep only the first N episodes across all files.
    # A source file that gets truncated mid-way is still written, but complete=False so a later
    # uncapped run rebuilds it in full (see _finalize_sidecar); per-episode resume keeps the
    # episodes it already has.
    file_full = {fp: True for fp in files}
    if max_episodes and max_episodes > 0:
        flat = [(fp, ei, dk, ns) for fp, eps in files.items() for (ei, dk, ns) in eps]
        if max_episodes < len(flat):
            print(f"[max_episodes={max_episodes}] limiting to the first {max_episodes} of {len(flat)} episodes")
            allowed = {ei for (_, ei, _, _) in flat[:max_episodes]}
            trimmed = OrderedDict()
            for fp, eps in files.items():
                kept = [(ei, dk, ns) for (ei, dk, ns) in eps if ei in allowed]
                if kept:
                    trimmed[fp] = kept
                    file_full[fp] = len(kept) == len(eps)
            files = trimmed

    pending = OrderedDict()
    for fp, eps in files.items():
        sc = sidecar_path_for(fp, data_dir, out_dir)
        if force and os.path.exists(sc):
            os.remove(sc)
        elif _is_complete(sc):
            print(f"[skip] already complete: {sc}")
            continue
        pending[fp] = eps
    if not pending:
        print("Nothing to build -- every source file already has a complete sidecar.")
        return

    ep_to_file, ep_to_demo_key, ep_num_steps = {}, {}, {}
    file_all_demo_keys, file_remaining = {}, {}
    todo_ep_idxs = set()
    for fp, eps in pending.items():
        file_all_demo_keys[fp] = sorted(dk for (_, dk, _) in eps)
        done_eps = _completed_episodes(sidecar_path_for(fp, data_dir, out_dir))  # per-episode resume
        file_remaining[fp] = 0
        for (ei, dk, ns) in eps:
            ep_to_file[ei], ep_to_demo_key[ei], ep_num_steps[ei] = fp, dk, ns
            if dk not in done_eps:
                file_remaining[fp] += 1
                todo_ep_idxs.add(ei)

    resumed = sum(len(eps) for eps in pending.values()) - len(todo_ep_idxs)
    if resumed:
        print(f"[resume] {resumed} episode(s) already written -- regenerating only the rest")

    files_written = set()
    # Files whose every episode is already on disk (a prior run finished them but died before the
    # final metadata write, or a max_episodes cap that now matches) -> just finalize, no teacher.
    for fp in list(pending):
        if file_remaining[fp] == 0:
            sc = sidecar_path_for(fp, data_dir, out_dir)
            _finalize_sidecar(sc, file_all_demo_keys[fp], file_full.get(fp, True))
            files_written.add(fp)
            print(f"[done] {sc} (all {len(file_all_demo_keys[fp])} episodes already present)")

    if not todo_ep_idxs:
        print(f"Done: nothing left to generate ({len(files_written)} sidecar file(s) finalized).")
        return

    print(f"Loading teacher ({teacher_checkpoint}) on {device}...")
    teacher, _ = load_teacher(
        to_device=device,
        checkpoint=teacher_checkpoint,
        experiment_name=teacher_experiment_name,
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    sampler_indices = [
        g for g in range(dataset.num_steps) if dataset._step_to_episode_map[g][0] in todo_ep_idxs
    ]
    total = len(sampler_indices)
    print(
        f"Building teacher_on_demos targets for {len(pending) - len(files_written)} source file(s), "
        f"{len(todo_ep_idxs)} episode(s), {total} timesteps -> {out_dir}"
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler_indices,
        num_workers=dataloader_num_workers,
        persistent_workers=dataloader_num_workers > 0,
        collate_fn=default_collate,
    )

    buffers = {}  # episode_idx -> _new_buffer, lazily; freed as soon as the episode is written
    meta_common = dict(
        teacher_checkpoint=teacher_checkpoint,
        teacher_experiment_name=teacher_experiment_name,
        num_denoising_steps=int(num_denoising_steps),
        num_teacher_samples=int(num_teacher_samples),
        seed=int(seed),
        chunk_size=NUM_ACTIONS_CHUNK,
        build_git_sha=_git_sha(),
        created_utc=_dt.datetime.now(_dt.timezone.utc).isoformat(),
        # action_chunks / future_proprio are stored VERBATIM in the teacher's own normalization --
        # the matching training run's LIBERODataset must be constructed with the same
        # dataset_stats_override_path (see module docstring's NORMALIZATION section).
        norm_convention="teacher_checkpoint",
        dataset_stats_override_path=teacher_stats_path,
    )

    # Large prime-ish stride so each of the K samples' seeds never collides with another batch's
    # seed_for_batch (which only ever ranges over seed + [0, dataset.num_steps)).
    _SEED_STRIDE = 1_000_003

    done = 0
    for data_batch in loader:
        gidx = [int(x) for x in data_batch["__key__"].tolist()]
        seed_for_batch = seed + min(gidx)
        data_batch = batch_prep.move_batch_to_device(data_batch, device)

        # K independent teacher samples for this SAME batch of states -- different initial
        # diffusion noise per k (see arch_invariant_rand), same conditioning. Kept SEPARATE (not
        # averaged) so training sees the teacher's genuine multimodality instead of a collapsed,
        # possibly-invalid point estimate; see kd/build_teacher_on_demos_dataset.py's module
        # docstring and experiment_journal.txt 2026-09-04.
        per_k = []
        for k_idx in range(num_teacher_samples):
            with torch.no_grad(), batch_prep.policy_autocast():
                targets_k = generate_teacher_native_targets(
                    teacher, data_batch, num_steps=num_denoising_steps,
                    seed=seed_for_batch + k_idx * _SEED_STRIDE,
                )
            per_k.append(dict(
                native_action=targets_k["native_action"].detach().float().cpu().numpy(),  # (b, chunk, act)
                future_proprio=targets_k["future_proprio"].detach().float().cpu().numpy(),  # (b, proprio)
                value=targets_k["value"].detach().float().cpu().numpy(),  # (b,) in [0, 1]
                future_image=np.asarray(targets_k["future_image"], dtype=np.uint8),  # (b, H, W, 3)
                future_wrist_image=np.asarray(targets_k["future_wrist_image"], dtype=np.uint8),
            ))
        # native_action / future_proprio stored VERBATIM -- already in the teacher's own
        # normalization, which is also what `dataset` above was constructed with
        # (dataset_stats_override_path=teacher_stats_path) -- no renorm needed.

        for j, g in enumerate(gidx):
            episode_idx, rel = dataset._step_to_episode_map[g]
            buf = buffers.get(episode_idx)
            if buf is None:
                buf = buffers[episode_idx] = _new_buffer(ep_num_steps[episode_idx], num_teacher_samples)
            for k_idx, tk in enumerate(per_k):
                buf["action_chunks"][rel, k_idx] = tk["native_action"][j]
                buf["future_proprio"][rel, k_idx] = tk["future_proprio"][j]
                # generate_teacher_native_targets returns value in [0, 1] (clamp((v+1)/2)); the
                # base LIBERODataset value_function_return convention is [-1, 1]
                # (compute_monte_carlo_returns).
                buf["value"][rel, k_idx] = float(tk["value"][j]) * 2.0 - 1.0
                buf["future_image_jpeg"][rel][k_idx] = jpeg_encode_image(
                    np.ascontiguousarray(tk["future_image"][j]), quality=95
                )
                buf["future_wrist_image_jpeg"][rel][k_idx] = jpeg_encode_image(
                    np.ascontiguousarray(tk["future_wrist_image"][j]), quality=95
                )
            buf["filled"] += 1

            if buf["filled"] == ep_num_steps[episode_idx]:
                # Episode done -> write its group to the sidecar NOW and free the buffer.
                fp = ep_to_file[episode_idx]
                sc = sidecar_path_for(fp, data_dir, out_dir)
                _write_episode_group(
                    sc, dict(meta_common, source_relpath=os.path.relpath(fp, data_dir)),
                    ep_to_demo_key[episode_idx], buf,
                )
                del buffers[episode_idx]
                file_remaining[fp] -= 1
                print(
                    f"  [episode {ep_to_demo_key[episode_idx]}] {ep_num_steps[episode_idx]} steps written"
                    f" -- {file_remaining[fp]} left in {os.path.basename(sc)}",
                    flush=True,
                )
                if file_remaining[fp] == 0:
                    _finalize_sidecar(sc, file_all_demo_keys[fp], file_full.get(fp, True))
                    files_written.add(fp)
                    print(f"[done] {sc}", flush=True)

        done += len(gidx)
        if done % max(batch_size * 20, 1) < len(gidx):
            print(f"  {done}/{total} timesteps", flush=True)

    leftover = [fp for fp in pending if fp not in files_written]
    if leftover:
        raise RuntimeError(
            f"loader exhausted but {len(leftover)} source file(s) never completed: {sorted(leftover)}"
        )
    print(f"Done: {len(files_written)} sidecar file(s) written under {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build_params", required=True)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--t5_text_embeddings_path", default=None)
    parser.add_argument(
        "--task_names",
        nargs="+",
        default=None,
        help="Restrict to source files whose name contains one of these (case-insensitive), e.g. --task_names ketchup",
    )
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--force", action="store_true", help="Rebuild source files whose sidecar is already complete")
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help="Smoketest / quick iteration: build only the first N demo episodes (0 = all)",
    )
    args = parser.parse_args()

    overrides = {
        k: v
        for k, v in dict(
            data_dir=args.data_dir,
            t5_text_embeddings_path=args.t5_text_embeddings_path,
            task_names=args.task_names,
            out_dir=args.out_dir,
        ).items()
        if v is not None
    }
    if args.force:
        overrides["force"] = True
    if args.max_episodes is not None:
        overrides["max_episodes"] = args.max_episodes
    params = kd_params.load_build_teacher_on_demos_params(args.build_params, overrides=overrides or None)

    build(
        teacher_checkpoint=params.teacher_checkpoint,
        teacher_experiment_name=params.teacher_experiment_name,
        data_dir=params.data_dir,
        t5_text_embeddings_path=params.t5_text_embeddings_path,
        out_dir=params.out_dir,
        task_names=params.task_names,
        batch_size=params.batch_size,
        dataloader_num_workers=params.dataloader_num_workers,
        num_denoising_steps=params.num_denoising_steps,
        num_teacher_samples=params.num_teacher_samples,
        max_episodes=params.max_episodes,
        device=params.device,
        seed=params.seed,
        force=params.force,
    )


if __name__ == "__main__":
    main()
