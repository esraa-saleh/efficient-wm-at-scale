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
Closed-loop LIBERO rollout evaluation for a `simple_dit_bc` policy + `SimpleDiTWorldModel`
best-of-N planner.

Mirrors the full Cosmos Policy model's own best-of-N planning loop
(`run_libero_eval.py`'s `run_episode()`: sample N candidate action chunks, imagine each one's
future state and value, execute only the best-scoring chunk, replan) but with both the action
proposer and the value/dynamics scorer swapped for the small `simple_dit_bc` models, at a small
fraction of the 2B teacher's per-step compute cost.

Usage:
    python -m cosmos_policy.scripts.simple_dit_bc.eval_mpc \
        --bc-checkpoint /path/to/simple_dit_bc_run \
        --world-model-checkpoint /path/to/world_model_run \
        --task-suite libero_object --num-trials 10 --num-queries-best-of-n 4

Compare its success rate and wall-clock against plain `eval.py` (no search, num-queries=1
equivalent) to check whether the added planning actually pays for itself.
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

from cosmos_policy.experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    save_rollout_video,
)
from cosmos_policy.scripts.periodic_libero_eval import resolve_task_id
from cosmos_policy.scripts.simple_dit_bc.dataset import load_t5_embeddings
from cosmos_policy.scripts.simple_dit_bc.eval import (
    NUM_STEPS_WAIT,
    TASK_MAX_STEPS,
    build_proprio,
    load_stats,
    normalize,
    preprocess_image,
    resolve_checkpoint_path,
    sample_action_chunk,
)
from cosmos_policy.scripts.simple_dit_bc.model import SimpleDiTBC, SimpleDiTWorldModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bc-checkpoint", required=True, help="simple_dit_bc checkpoint or run dir.")
    parser.add_argument("--world-model-checkpoint", required=True, help="SimpleDiTWorldModel checkpoint or run dir.")
    parser.add_argument(
        "--dataset-stats-path",
        default=None,
        help="Path to dataset_stats.json. Defaults to the world model checkpoint's own "
        "'<work_dir>/dataset_stats.json' -- used as the single source of truth for both "
        "unnormalizing the BC policy's actions and normalizing them back for the world model.",
    )
    parser.add_argument("--t5-embeddings-path", default=None)
    parser.add_argument("--task-suite", default="libero_object")
    parser.add_argument("--task-keyword", default=None)
    parser.add_argument("--summary-csv", default=None)
    parser.add_argument("--num-trials", type=int, default=10, help="Trials per task.")
    parser.add_argument("--num-queries-best-of-n", type=int, default=4, help="N candidate action chunks per plan.")
    parser.add_argument("--num-denoising-steps", type=int, default=10, help="BC policy's Euler ODE steps.")
    parser.add_argument(
        "--num-open-loop-steps",
        type=int,
        default=None,
        help="Actions to execute from the winning chunk before replanning. Defaults to chunk_size.",
    )
    parser.add_argument("--env-img-res", type=int, default=256)
    parser.add_argument("--flip-images", type=lambda s: s.lower() != "false", default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rollout-dir", default="./rollouts/simple_dit_bc_mpc")
    return parser.parse_args()


def load_bc_model(checkpoint_path, device: torch.device) -> tuple[SimpleDiTBC, dict]:
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
    print(f"Loaded BC policy from iteration {ckpt['iteration']} (chunk_size={ckpt_args['chunk_size']}).")
    return model, ckpt_args


def load_world_model(checkpoint_path, device: torch.device) -> tuple[SimpleDiTWorldModel, dict]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    ckpt_args = ckpt["args"]
    model = SimpleDiTWorldModel(
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
    print(f"Loaded world model from iteration {ckpt['iteration']} (chunk_size={ckpt_args['chunk_size']}).")
    return model, ckpt_args


@torch.no_grad()
def score_action_chunk(
    world_model: SimpleDiTWorldModel,
    agentview_img: torch.Tensor,
    wrist_img: torch.Tensor,
    proprio: torch.Tensor,
    action_chunk_normalized: torch.Tensor,
    task_emb: torch.Tensor,
) -> float:
    _, _, value = world_model(agentview_img, wrist_img, proprio, action_chunk_normalized, task_emb)
    return value.item()


def run_episode_mpc(
    env,
    bc_model: SimpleDiTBC,
    world_model: SimpleDiTWorldModel,
    stats: dict,
    task_emb: torch.Tensor,
    image_size: int,
    max_steps: int,
    num_denoising_steps: int,
    num_open_loop_steps: int,
    num_queries_best_of_n: int,
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

            best_actions, best_value = None, -float("inf")
            for _ in range(num_queries_best_of_n):
                candidate_actions = sample_action_chunk(
                    bc_model, agentview_img, wrist_img, proprio, task_emb, num_denoising_steps, stats
                )  # unnormalized actions, (chunk_size, action_dim)
                candidate_normalized = normalize(candidate_actions, stats, "actions")
                candidate_tensor = torch.from_numpy(candidate_normalized).float().unsqueeze(0).to(device)
                value = score_action_chunk(world_model, agentview_img, wrist_img, proprio, candidate_tensor, task_emb)
                if value > best_value:
                    best_value = value
                    best_actions = candidate_actions

            action_queue.extend(best_actions[:num_open_loop_steps])

        action = action_queue.popleft()
        obs, _, done, _ = env.step(action.tolist())
        if done:
            success = True
            break
        t += 1

    return success, replay_images


def evaluate_task_mpc(
    task_id: int,
    task_description: str,
    task_suite,
    bc_model: SimpleDiTBC,
    world_model: SimpleDiTWorldModel,
    stats: dict,
    task_emb: torch.Tensor,
    image_size: int,
    max_steps: int,
    num_trials: int,
    num_denoising_steps: int,
    num_open_loop_steps: int,
    num_queries_best_of_n: int,
    flip_images: bool,
    env_img_res: int,
    device: torch.device,
) -> tuple[int, int]:
    task = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)
    env, _ = get_libero_env(task, "simple_dit_bc_mpc", resolution=env_img_res)

    num_successes = 0
    try:
        for episode_idx in range(num_trials):
            start = time.time()
            success, replay_images = run_episode_mpc(
                env=env,
                bc_model=bc_model,
                world_model=world_model,
                stats=stats,
                task_emb=task_emb,
                image_size=image_size,
                max_steps=max_steps,
                num_denoising_steps=num_denoising_steps,
                num_open_loop_steps=num_open_loop_steps,
                num_queries_best_of_n=num_queries_best_of_n,
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

    bc_checkpoint_path = resolve_checkpoint_path(args.bc_checkpoint)
    world_model_checkpoint_path = resolve_checkpoint_path(args.world_model_checkpoint)
    print(f"Using BC checkpoint: {bc_checkpoint_path}")
    print(f"Using world model checkpoint: {world_model_checkpoint_path}")

    stats_path = (
        pathlib.Path(args.dataset_stats_path)
        if args.dataset_stats_path is not None
        else world_model_checkpoint_path.parent.parent / "dataset_stats.json"
    )
    stats = load_stats(stats_path)
    print(f"Loaded dataset stats from {stats_path}")

    bc_model, bc_ckpt_args = load_bc_model(bc_checkpoint_path, device)
    world_model, world_model_ckpt_args = load_world_model(world_model_checkpoint_path, device)
    num_open_loop_steps = args.num_open_loop_steps or bc_ckpt_args["chunk_size"]

    t5_embeddings_path = args.t5_embeddings_path or bc_ckpt_args.get("t5_embeddings_path")
    if t5_embeddings_path is None:
        raise ValueError(
            "--t5-embeddings-path not given and not found in the BC checkpoint's saved args -- pass it explicitly."
        )
    print(f"Using T5 embeddings from: {t5_embeddings_path}")
    t5_embeddings = load_t5_embeddings(t5_embeddings_path)

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
            print(f"SKIPPING task_id={task_id} ({task_description!r}): no T5 embedding for it.")
            continue

        print(f"\n=== Evaluating task_id={task_id}: {task_description!r} (best-of-{args.num_queries_best_of_n}) ===")
        task_emb = torch.from_numpy(t5_embeddings[task_description]).float().unsqueeze(0).to(device)
        num_successes, num_trials = evaluate_task_mpc(
            task_id=task_id,
            task_description=task_description,
            task_suite=task_suite,
            bc_model=bc_model,
            world_model=world_model,
            stats=stats,
            task_emb=task_emb,
            image_size=bc_ckpt_args["image_size"],
            max_steps=max_steps,
            num_trials=args.num_trials,
            num_denoising_steps=args.num_denoising_steps,
            num_open_loop_steps=num_open_loop_steps,
            num_queries_best_of_n=args.num_queries_best_of_n,
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
