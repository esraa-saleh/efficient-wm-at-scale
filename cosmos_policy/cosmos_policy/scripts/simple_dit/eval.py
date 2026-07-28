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
Closed-loop LIBERO rollout evaluation for a `simple_dit_bc` checkpoint.

Mirrors what `cosmos_policy.experiments.robot.libero.run_libero_eval` does for the full Cosmos
Policy model, but stripped down to `SimpleDiTBC`'s much simpler interface: no VAE/text
conditioning, no best-of-N/value-function search, no DDPM-style noise schedule -- just Euler
integration of the trained flow-matching velocity field over one action chunk at a time.

Usage (point --checkpoint at a specific iter_*.pt file):
    python -m cosmos_policy.scripts.simple_dit_bc.eval \
        --checkpoint /tmp/simple_dit_bc_demo/checkpoints/iter_000005000.pt \
        --num-trials 10

Or point it at the run's --work-dir (or its checkpoints/ dir) to auto-pick the
highest-iteration checkpoint, and rely on --t5-embeddings-path defaulting to
whatever train.py was run with (recorded in every checkpoint's "args", and in
plain-text form in the run's config.json):
    python -m cosmos_policy.scripts.simple_dit_bc.eval \
        --checkpoint /tmp/simple_dit_bc_demo \
        --task-suite libero_object --task-keyword alphabet_soup --num-trials 10

Omit --task-keyword to evaluate every task in --task-suite instead of a single one (useful
for a checkpoint trained across a whole suite, e.g. via train.py's --data-dir pointed at
LIBERO-Cosmos-Policy/success_only/libero_object_regen/ -- see train.py's docstring). Per-task and
overall success rates are printed, and a per-task CSV is written to --summary-csv (default
'<rollout-dir>/eval_results.csv'):
    python -m cosmos_policy.scripts.simple_dit_bc.eval \
        --checkpoint /tmp/simple_dit_bc_demo --task-suite libero_object --num-trials 10
"""

import argparse
import csv
import json
import pathlib
import time
from collections import deque

import numpy as np
import torch
from libero.libero import benchmark

from cosmos_policy.datasets.dataset_utils import resize_images
from cosmos_policy.experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    save_rollout_video,
)
from cosmos_policy.scripts.periodic_libero_eval import resolve_task_id
from cosmos_policy.scripts.simple_dit_bc.dataset import load_t5_embeddings
from cosmos_policy.scripts.simple_dit_bc.model import SimpleDiTBC

# Copied from run_libero_eval.py rather than imported -- that module transitively pulls in the
# full Cosmos model stack (cosmos_utils.py) that simple_dit_bc deliberately has no dependency on.
TASK_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}
NUM_STEPS_WAIT = 10  # no-op steps at episode start to let objects settle, matches run_libero_eval.py


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to a simple_dit_bc iter_*.pt checkpoint, OR a run directory (train.py's "
        "--work-dir, or its checkpoints/ subdir) -- in the latter case the highest-iteration "
        "checkpoint under it is used.",
    )
    parser.add_argument(
        "--dataset-stats-path",
        default=None,
        help="Path to dataset_stats.json. Defaults to '<checkpoint's work_dir>/dataset_stats.json' "
        "(checkpoints live at '<work_dir>/checkpoints/iter_*.pt').",
    )
    parser.add_argument(
        "--t5-embeddings-path",
        default=None,
        help="Path to t5_embeddings.pkl -- must contain the eval task's instruction. Defaults to "
        "whatever path train.py was run with (recorded in the checkpoint's saved args); pass this "
        "explicitly to evaluate with a different pickle (e.g. one covering more task instructions).",
    )
    parser.add_argument("--task-suite", default="libero_object")
    parser.add_argument(
        "--task-keyword",
        default=None,
        help="Substring matched against one task's language description within --task-suite "
        "(e.g. 'alphabet_soup'). If omitted, evaluates every task in --task-suite.",
    )
    parser.add_argument(
        "--summary-csv",
        default=None,
        help="Where to write the per-task success-rate CSV. Defaults to '<rollout-dir>/eval_results.csv'.",
    )
    parser.add_argument("--num-trials", type=int, default=10, help="Trials per task.")
    parser.add_argument("--num-denoising-steps", type=int, default=10, help="Euler ODE integration steps.")
    parser.add_argument(
        "--num-open-loop-steps",
        type=int,
        default=None,
        help="Actions to execute from a sampled chunk before requerying. Defaults to the checkpoint's chunk_size.",
    )
    parser.add_argument("--env-img-res", type=int, default=256, help="Render resolution (video quality only).")
    parser.add_argument("--flip-images", type=lambda s: s.lower() != "false", default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rollout-dir", default="./rollouts/simple_dit_bc")
    return parser.parse_args()


def resolve_checkpoint_path(path: str) -> pathlib.Path:
    """Accepts either a specific `iter_*.pt` file, or a run directory (train.py's --work-dir, or
    its checkpoints/ subdir directly) -- in the latter case, resolves to the highest-iteration
    checkpoint found under it."""
    p = pathlib.Path(path)
    if p.is_file():
        return p
    if not p.is_dir():
        raise FileNotFoundError(f"Checkpoint path {path!r} does not exist.")

    search_dirs = [p, p / "checkpoints"]
    candidates = [f for d in search_dirs if d.is_dir() for f in d.glob("iter_*.pt")]
    if not candidates:
        raise FileNotFoundError(f"No iter_*.pt checkpoints found under {path!r} (looked in {search_dirs}).")
    return max(candidates, key=lambda f: int(f.stem.split("_")[1]))


def load_model(checkpoint_path: str, device: torch.device) -> tuple[SimpleDiTBC, dict]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    ckpt_args = ckpt["args"]
    model = SimpleDiTBC(
        action_dim=7,
        proprio_dim=9,
        task_emb_dim=ckpt_args.get("task_emb_dim", 1024),
        chunk_size=ckpt_args["chunk_size"],
        embed_dim=ckpt_args["embed_dim"],
        num_blocks=ckpt_args["num_blocks"],
        num_heads=ckpt_args["num_heads"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint from iteration {ckpt['iteration']} (chunk_size={ckpt_args['chunk_size']}, "
          f"image_size={ckpt_args['image_size']}, embed_dim={ckpt_args['embed_dim']}, "
          f"num_blocks={ckpt_args['num_blocks']}, num_heads={ckpt_args['num_heads']}).")
    return model, ckpt_args


def load_stats(path) -> dict[str, np.ndarray]:
    with open(path, "r") as f:
        raw = json.load(f)
    return {k: np.asarray(v, dtype=np.float32) for k, v in raw.items()}


def normalize(x: np.ndarray, stats: dict, key: str) -> np.ndarray:
    lo, hi = stats[f"{key}_min"], stats[f"{key}_max"]
    return 2 * (x - lo) / (hi - lo) - 1


def unnormalize(x: np.ndarray, stats: dict, key: str) -> np.ndarray:
    lo, hi = stats[f"{key}_min"], stats[f"{key}_max"]
    return (x + 1) / 2 * (hi - lo) + lo


def preprocess_image(img: np.ndarray, image_size: int, device: torch.device) -> torch.Tensor:
    """(H,W,3) uint8 -> (1,3,image_size,image_size) float32 in [0,1], matching dataset.py's pipeline."""
    resized = resize_images(img[None], image_size)[0]
    tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
    return tensor.unsqueeze(0).to(device)


def build_proprio(obs: dict, stats: dict, device: torch.device) -> torch.Tensor:
    """Matches data/demo_N/robot_states's composition (confirmed against run_libero_eval.py's
    own prepare_observation, which builds proprio identically for the full Cosmos Policy model)."""
    raw = np.concatenate((obs["robot0_gripper_qpos"], obs["robot0_eef_pos"], obs["robot0_eef_quat"])).astype(
        np.float32
    )
    normalized = normalize(raw, stats, "proprio")
    return torch.from_numpy(normalized).float().unsqueeze(0).to(device)


@torch.no_grad()
def sample_action_chunk(
    model: SimpleDiTBC,
    agentview_img: torch.Tensor,
    wrist_img: torch.Tensor,
    proprio: torch.Tensor,
    task_emb: torch.Tensor,
    num_steps: int,
    stats: dict,
) -> np.ndarray:
    """Euler-integrate dx/dt = v_theta(x, t, context) from x0 ~ N(0,I) at t=0 to t=1, then unnormalize."""
    device = proprio.device
    x = torch.randn(1, model.chunk_size, 7, device=device)
    dt = 1.0 / num_steps
    for step in range(num_steps):
        t = torch.full((1,), step * dt, device=device)
        v = model(agentview_img, wrist_img, proprio, x, t, task_emb)
        x = x + v * dt
    return unnormalize(x[0].cpu().numpy(), stats, "actions")


def run_episode(
    env,
    task_description: str,
    model: SimpleDiTBC,
    stats: dict,
    task_emb: torch.Tensor,
    image_size: int,
    max_steps: int,
    num_denoising_steps: int,
    num_open_loop_steps: int,
    flip_images: bool,
    initial_state,
    device: torch.device,
) -> tuple[bool, list[np.ndarray]]:
    env.reset()
    obs = env.set_init_state(initial_state)

    action_queue = deque(maxlen=num_open_loop_steps)
    replay_images = []
    t = 0
    success = False
    while t < max_steps + NUM_STEPS_WAIT:
        if t < NUM_STEPS_WAIT:
            obs, _, _, _ = env.step(get_libero_dummy_action("simple_dit_bc"))
            t += 1
            continue

        agentview_raw = get_libero_image(obs, flip_images)
        wrist_raw = get_libero_wrist_image(obs, flip_images)
        replay_images.append(agentview_raw)

        if len(action_queue) == 0:
            agentview_img = preprocess_image(agentview_raw, image_size, device)
            wrist_img = preprocess_image(wrist_raw, image_size, device)
            proprio = build_proprio(obs, stats, device)
            action_chunk = sample_action_chunk(
                model, agentview_img, wrist_img, proprio, task_emb, num_denoising_steps, stats
            )
            action_queue.extend(action_chunk[:num_open_loop_steps])

        action = action_queue.popleft()
        obs, _, done, _ = env.step(action.tolist())
        if done:
            success = True
            break
        t += 1

    return success, replay_images


def evaluate_task(
    task_id: int,
    task_description: str,
    task_suite,
    model: SimpleDiTBC,
    stats: dict,
    task_emb: torch.Tensor,
    image_size: int,
    max_steps: int,
    num_trials: int,
    num_denoising_steps: int,
    num_open_loop_steps: int,
    flip_images: bool,
    env_img_res: int,
    device: torch.device,
) -> tuple[int, int]:
    """Runs `num_trials` closed-loop episodes of one LIBERO task, returns (num_successes, num_trials).

    Creates and closes its own env (rather than reusing one across tasks) since each task has a
    different bddl scene/object set -- `get_libero_env` builds the env from `task`'s own bddl file.
    """
    task = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)
    env, _ = get_libero_env(task, "simple_dit_bc", resolution=env_img_res)

    num_successes = 0
    try:
        for episode_idx in range(num_trials):
            start = time.time()
            success, replay_images = run_episode(
                env=env,
                task_description=task_description,
                model=model,
                stats=stats,
                task_emb=task_emb,
                image_size=image_size,
                max_steps=max_steps,
                num_denoising_steps=num_denoising_steps,
                num_open_loop_steps=num_open_loop_steps,
                flip_images=flip_images,
                initial_state=initial_states[episode_idx],
                device=device,
            )
            num_successes += int(success)
            elapsed = time.time() - start
            print(
                f"[{task_description}] Episode {episode_idx + 1}/{num_trials}: success={success} ({elapsed:.1f}s). "
                f"Running success rate: {num_successes}/{episode_idx + 1} "
                f"({100 * num_successes / (episode_idx + 1):.1f}%)"
            )
            save_rollout_video(replay_images, episode_idx, success, task_description)
    finally:
        if hasattr(env, "close"):
            env.close()

    return num_successes, num_trials


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    print(f"Using checkpoint: {checkpoint_path}")
    stats_path = (
        pathlib.Path(args.dataset_stats_path)
        if args.dataset_stats_path is not None
        else checkpoint_path.parent.parent / "dataset_stats.json"
    )
    stats = load_stats(stats_path)
    print(f"Loaded dataset stats from {stats_path}")

    model, ckpt_args = load_model(checkpoint_path, device)
    num_open_loop_steps = args.num_open_loop_steps or ckpt_args["chunk_size"]

    t5_embeddings_path = args.t5_embeddings_path or ckpt_args.get("t5_embeddings_path")
    if t5_embeddings_path is None:
        raise ValueError(
            "--t5-embeddings-path not given and not found in the checkpoint's saved args -- pass it explicitly."
        )
    print(f"Using T5 embeddings from: {t5_embeddings_path}")
    t5_embeddings = load_t5_embeddings(t5_embeddings_path)

    trained_instructions_path = checkpoint_path.parent.parent / "task_instructions.json"
    trained_instructions = None
    if trained_instructions_path.is_file():
        with open(trained_instructions_path, "r") as f:
            trained_instructions = json.load(f)

    task_suite = benchmark.get_benchmark_dict()[args.task_suite]()
    max_steps = TASK_MAX_STEPS[args.task_suite]

    rollout_dir = pathlib.Path(args.rollout_dir)
    rollout_dir.mkdir(parents=True, exist_ok=True)

    if args.task_keyword is not None:
        task_id, task_description = resolve_task_id(args.task_suite, args.task_keyword)
        print(f"Resolved task_id={task_id} ({task_description!r}) in suite {args.task_suite!r}")
        tasks_to_eval = [(task_id, task_description)]
    else:
        tasks_to_eval = [(i, task_suite.get_task(i).language) for i in range(task_suite.n_tasks)]
        print(f"No --task-keyword given -- evaluating all {len(tasks_to_eval)} tasks in suite {args.task_suite!r}.")

    results = []
    for task_id, task_description in tasks_to_eval:
        if task_description not in t5_embeddings:
            print(
                f"SKIPPING task_id={task_id} ({task_description!r}): no T5 embedding for it in "
                f"{t5_embeddings_path!r}. Available instructions: {sorted(t5_embeddings.keys())}"
            )
            continue
        if trained_instructions is not None and task_description not in trained_instructions:
            print(
                f"WARNING: {task_description!r} is not in this checkpoint's trained task list "
                f"({trained_instructions}) -- evaluating a task it never saw during training."
            )

        print(f"\n=== Evaluating task_id={task_id}: {task_description!r} ===")
        task_emb = torch.from_numpy(t5_embeddings[task_description]).float().unsqueeze(0).to(device)
        num_successes, num_trials = evaluate_task(
            task_id=task_id,
            task_description=task_description,
            task_suite=task_suite,
            model=model,
            stats=stats,
            task_emb=task_emb,
            image_size=ckpt_args["image_size"],
            max_steps=max_steps,
            num_trials=args.num_trials,
            num_denoising_steps=args.num_denoising_steps,
            num_open_loop_steps=num_open_loop_steps,
            flip_images=args.flip_images,
            env_img_res=args.env_img_res,
            device=device,
        )
        success_rate = num_successes / num_trials
        results.append(
            {
                "task_id": task_id,
                "task_description": task_description,
                "num_trials": num_trials,
                "num_successes": num_successes,
                "success_rate": success_rate,
            }
        )
        print(
            f"Task {task_id} ({task_description!r}) success rate: {num_successes}/{num_trials} "
            f"({100 * success_rate:.1f}%)"
        )

    print("\n=== Final results ===")
    for r in results:
        print(
            f"  [{r['task_id']}] {r['task_description']!r}: {r['num_successes']}/{r['num_trials']} "
            f"({100 * r['success_rate']:.1f}%)"
        )
    total_trials = sum(r["num_trials"] for r in results)
    total_successes = sum(r["num_successes"] for r in results)
    if total_trials > 0:
        overall_success_rate = total_successes / total_trials
        print(
            f"Overall success rate: {overall_success_rate:.4f} ({overall_success_rate * 100:.1f}%) "
            f"across {total_successes}/{total_trials} episodes, {len(results)} task(s)"
        )

    summary_csv_path = pathlib.Path(args.summary_csv) if args.summary_csv else rollout_dir / "eval_results.csv"
    summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["task_id", "task_description", "num_trials", "num_successes", "success_rate"]
        )
        writer.writeheader()
        writer.writerows(results)
    print(f"Wrote per-task results to {summary_csv_path}")


if __name__ == "__main__":
    main()
