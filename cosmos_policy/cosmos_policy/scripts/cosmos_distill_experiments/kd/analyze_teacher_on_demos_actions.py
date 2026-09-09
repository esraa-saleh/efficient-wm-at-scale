# SPDX-License-Identifier: Apache-2.0
"""Throwaway: per-dimension comparison of a teacher_on_demos sidecar's action targets against the
human demo actions, over every ketchup timestep -- the data behind the "Teacher action error by
dimension" artifact.

Pure h5py + numpy (no torch, no LIBERODataset), so it runs on a CPU node:
  - demo actions: read raw from <data_dir>/<task>_demo.hdf5, then local-normalize with
    <data_dir>/dataset_statistics.json (exactly what LIBERODataset.rescale_data does).
  - teacher actions: read straight from the sidecar (already local-normalized post-fix; older
    sidecars are teacher-normalized -- the root attr norm_convention says which).

Writes <out>.json with the artifact's data contract:
  task, n_episodes, n_timesteps, chunk, build, overall_rmse, demo_chunk_rms,
  dims: [{name, edges(41), hist_demo(40), hist_teacher(40), demo_mean, demo_std,
          teacher_mean, teacher_std, demo_absmean, teacher_absmean, atten, corr, rmse,
          err_by_prog(10), scatter(<=1500 [demo,teacher] executed-action pairs)}]

    python -m cosmos_policy.scripts.cosmos_distill_experiments.kd.analyze_teacher_on_demos_actions \
        --sidecar_dir <.../libero_object_regen__teacher_on_demos> \
        --data_dir    <.../libero_object_regen> \
        --task ketchup --out /home/esraa1/tod_actdist_renorm.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import h5py
import numpy as np

DIMS = ["x", "y", "z", "roll", "pitch", "yaw", "grip"]
CHUNK = 16


def _norm(raw: np.ndarray, amin: np.ndarray, amax: np.ndarray) -> np.ndarray:
    return 2.0 * (raw - amin) / (amax - amin) - 1.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sidecar_dir", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--task", default="ketchup")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    stats = json.load(open(os.path.join(args.data_dir, "dataset_statistics.json")))
    amin, amax = np.array(stats["actions_min"]), np.array(stats["actions_max"])

    raw_matches = [p for p in glob.glob(os.path.join(args.data_dir, "*_demo.hdf5")) if args.task in os.path.basename(p)]
    assert len(raw_matches) == 1, f"expected 1 raw demo file for {args.task!r}, got {raw_matches}"
    raw_path = raw_matches[0]
    side_matches = [
        p for p in glob.glob(os.path.join(args.sidecar_dir, "*.teacher_on_demos.hdf5")) if args.task in os.path.basename(p)
    ]
    assert len(side_matches) == 1, f"expected 1 sidecar for {args.task!r}, got {side_matches}"
    side_path = side_matches[0]

    demo_exec, teach_exec = [], []          # (N, 7) executed action = chunk step 0
    demo_chunk, teach_chunk = [], []        # (M, 7) all real (non-padded) chunk entries
    prog = []                               # (M,) episode progress 0..1 for each chunk entry
    n_ep = 0

    with h5py.File(raw_path, "r") as fr, h5py.File(side_path, "r") as fs:
        rroot = fr["data"] if "data" in fr else fr
        build_note = f"num_denoising_steps={fs.attrs.get('num_denoising_steps')}, norm={fs.attrs.get('norm_convention', 'teacher (pre-fix)')}"
        keys = [k for k in fs.keys() if k in rroot]
        for dk in keys:
            n_ep += 1
            raw_a = np.asarray(rroot[dk]["actions"], dtype=np.float64)          # (T, 7) raw
            demo_n = _norm(raw_a, amin, amax)                                   # (T, 7) local-norm
            teach_c = np.asarray(fs[dk]["action_chunks"], dtype=np.float64)     # (T, 16, 7) local-norm
            T = demo_n.shape[0]
            for t in range(T):
                demo_exec.append(demo_n[t])
                teach_exec.append(teach_c[t, 0])
                L = min(CHUNK, T - t)                                          # real (non-padded) steps
                demo_chunk.append(demo_n[t : t + L])
                teach_chunk.append(teach_c[t, :L])
                prog.extend([t / max(T - 1, 1)] * L)

    demo_exec = np.stack(demo_exec)           # (N, 7)
    teach_exec = np.stack(teach_exec)
    demo_chunk = np.concatenate(demo_chunk)   # (M, 7)
    teach_chunk = np.concatenate(teach_chunk)
    prog = np.asarray(prog)
    N = demo_exec.shape[0]

    overall_rmse = float(np.sqrt(np.mean((teach_chunk - demo_chunk) ** 2)))
    demo_chunk_rms = float(np.sqrt(np.mean(demo_chunk ** 2)))

    sc_idx = rng.choice(N, size=min(1500, N), replace=False)
    dims = []
    for i, name in enumerate(DIMS):
        d_ex, t_ex = demo_exec[:, i], teach_exec[:, i]
        d_ch, t_ch = demo_chunk[:, i], teach_chunk[:, i]
        lo = float(min(d_ex.min(), t_ex.min()))
        hi = float(max(d_ex.max(), t_ex.max()))
        pad = 0.05 * (hi - lo + 1e-9)
        edges = np.linspace(lo - pad, hi + pad, 41)
        hd, _ = np.histogram(d_ex, bins=edges, density=True)
        ht, _ = np.histogram(t_ex, bins=edges, density=True)
        err = np.abs(t_ch - d_ch)
        ebp = [float(err[(prog >= k / 10) & (prog < (k + 1) / 10)].mean() if np.any((prog >= k / 10) & (prog < (k + 1) / 10)) else 0.0) for k in range(10)]
        d_abs, t_abs = float(np.abs(d_ch).mean()), float(np.abs(t_ch).mean())
        dims.append(dict(
            name=name,
            edges=[round(float(x), 4) for x in edges],
            hist_demo=[round(float(x), 4) for x in hd],
            hist_teacher=[round(float(x), 4) for x in ht],
            demo_mean=round(float(d_ch.mean()), 4), demo_std=round(float(d_ch.std()), 4),
            teacher_mean=round(float(t_ch.mean()), 4), teacher_std=round(float(t_ch.std()), 4),
            demo_absmean=round(d_abs, 4), teacher_absmean=round(t_abs, 4),
            atten=round(t_abs / (d_abs + 1e-9), 4),
            corr=round(float(np.corrcoef(d_ch, t_ch)[0, 1]), 4),
            rmse=round(float(np.sqrt(np.mean((t_ch - d_ch) ** 2))), 4),
            err_by_prog=[round(x, 5) for x in ebp],
            scatter=[[round(float(d_ex[j]), 3), round(float(t_ex[j]), 3)] for j in sc_idx],
        ))

    out = dict(
        task="pick_up_the_ketchup_and_place_it_in_the_basket",
        n_episodes=n_ep, n_timesteps=N, chunk=CHUNK, build=build_note,
        overall_rmse=round(overall_rmse, 4), demo_chunk_rms=round(demo_chunk_rms, 4),
        dims=dims,
    )
    json.dump(out, open(args.out, "w"))
    print(f"wrote {args.out}: {n_ep} episodes, {N} timesteps, overall_rmse {overall_rmse:.4f}, "
          f"demo_chunk_rms {demo_chunk_rms:.4f}")
    for d in dims:
        print(f"  {d['name']:>5}  demo|.|={d['demo_absmean']:.3f}  teach|.|={d['teacher_absmean']:.3f}  "
              f"atten={d['atten']:.2f}  corr={d['corr']:+.2f}  rmse={d['rmse']:.3f}")


if __name__ == "__main__":
    main()
