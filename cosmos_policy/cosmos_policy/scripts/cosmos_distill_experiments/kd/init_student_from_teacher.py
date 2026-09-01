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
One-time offline script: builds a depth-reduced student initialized from the released 2B teacher's
own block weights, via the layer map from the KD plan (j_i = round(i * (L_T-1)/(L_S-1))). CPU is
enough -- this is a weight copy, no forward/backward pass.

Per the KD plan's finding #2, the teacher net is fixed-width (COSMOS_V2_2B_NET: model_channels=2048,
num_heads=16 -- see ../kd/net_experiments.py's module docstring) and every registered student size
only reduces `num_blocks`, so block shapes are byte-identical between teacher and student -- no
projection is needed anywhere here, only a plain `load_state_dict`.

Usage:
    python -m cosmos_policy.scripts.cosmos_distill_experiments.kd.init_student_from_teacher \\
      --student_size 1b --out /path/to/student_init_1b.pt

`--dcp_out <dir>` additionally writes the SAME weights as a torch.distributed.checkpoint (DCP)
directory (`<dir>/model/{.metadata,__0_0.distcp}`), needed only for run types that load an init via
the generic torchrun/DistributedCheckpointer path (e.g. baseline_1b_train.yaml's `checkpoint.
load_path`) -- that loader's keys_to_resume_during_load() (checkpointer/dcp.py:493) explicitly
excludes any load_path ending in ".pt" from being treated as an init to load, so the flat --out file
alone is NOT sufficient there (it works fine for kd_static's own student_init_path loading, which
reads the flat .pt directly with its own code -- this flag doesn't change or replace that). Reuses
the exact save call the real training loop's own Checkpointer.save() uses for its "model" key
(dcp.py:684-686), so the result loads via the identical code path a real checkpoint would.
"""

import argparse
import json
import os
import pathlib

import torch
import torch.distributed as dist
from torch.distributed.checkpoint import DefaultSavePlanner, FileSystemWriter
from torch.distributed.checkpoint import save as dcp_save

from cosmos_policy._src.imaginaire.utils.count_params import count_params
from cosmos_policy._src.predict2.checkpointer.dcp import ModelWrapper
from cosmos_policy.scripts.cosmos_distill_experiments.kd.net_experiments import STUDENT_SIZES
from cosmos_policy.scripts.cosmos_distill_experiments.kd.teacher_loader import (
    DEFAULT_TEACHER_CHECKPOINT,
    load_policy_model,
    load_teacher,
)


def save_dcp_model_checkpoint(student, dcp_out_dir: pathlib.Path) -> None:
    """Writes `student`'s weights as a DCP directory at `<dcp_out_dir>/model/`, matching
    Checkpointer.save()'s own "model" key exactly (dcp.py:684-686: ModelWrapper(model).state_dict()
    via dcp.save with a FileSystemWriter + DefaultSavePlanner(dedup_save_to_lowest_rank=True)) --
    the only key `checkpoint.load_path` resume actually reads (dcp.py:558-561) when
    `load_training_state=False`, i.e. seeding a fresh run's weights rather than resuming full
    trainer state. torch.distributed.checkpoint.save needs a process group even for one process --
    initialized here (gloo, world_size=1) since this script otherwise never touches torch.distributed
    (CPU only, no forward/backward pass -- see module docstring)."""
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29501")
        dist.init_process_group(backend="gloo", rank=0, world_size=1)

    model_dir = dcp_out_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_state = ModelWrapper(student).state_dict()
    dcp_save(
        model_state,
        storage_writer=FileSystemWriter(path=str(model_dir)),
        planner=DefaultSavePlanner(dedup_save_to_lowest_rank=True),
    )


def layer_map(teacher_depth: int, student_depth: int) -> list[int]:
    """j_i = round(i * (L_T - 1) / (L_S - 1)) -- exact formula from the KD plan."""
    if student_depth == 1:
        return [0]
    return [round(i * (teacher_depth - 1) / (student_depth - 1)) for i in range(student_depth)]


def init_student(teacher, student) -> dict:
    """Mutates `student` in place: copies block weights per the layer map, and every non-block
    weight/buffer verbatim (identical shapes, since only depth differs -- see module docstring).
    Returns a small report dict (layer_map, param counts) for the caller to print/save."""
    teacher_blocks = teacher.net.blocks
    student_blocks = student.net.blocks
    mapping = layer_map(len(teacher_blocks), len(student_blocks))

    for student_idx, teacher_idx in enumerate(mapping):
        student_blocks[student_idx].load_state_dict(teacher_blocks[teacher_idx].state_dict())

    # Copy every net.* weight/buffer NOT belonging to a transformer block (x_embedder, t_embedder,
    # final_layer, pos_embedder, extra_pos_embedder, ...) verbatim, by filtering the full net
    # state dict rather than hand-listing submodule names -- robust to any submodule this file
    # doesn't explicitly know about, since none of them depend on num_blocks for their shape.
    teacher_net_state = teacher.net.state_dict()
    non_block_state = {k: v for k, v in teacher_net_state.items() if not k.startswith("blocks.")}
    missing, unexpected = student.net.load_state_dict(non_block_state, strict=False)
    # `missing` is expected to be exactly the student's own blocks.* keys (never in non_block_state);
    # `unexpected` must be empty -- a non-empty value here means the teacher's net has a non-block
    # submodule the student's net doesn't, which would indicate a real architecture mismatch.
    unexpected_non_block = [k for k in unexpected if not k.startswith("blocks.")]
    if unexpected_non_block:
        raise RuntimeError(f"Unexpected keys when copying non-block weights: {unexpected_non_block}")

    return {
        "layer_map": mapping,
        "teacher_num_blocks": len(teacher_blocks),
        "student_num_blocks": len(student_blocks),
        "teacher_params": count_params(teacher.net),
        "student_params": count_params(student.net),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student_size", required=True, choices=sorted(STUDENT_SIZES.keys()))
    parser.add_argument("--out", required=True, help="Output path for the initialized student's model.pt")
    parser.add_argument(
        "--dcp_out",
        default=None,
        help="Optional: also write a DCP-format directory here (see module docstring) for run "
        "types that load an init via checkpoint.load_path (the generic torchrun path), as opposed "
        "to kd_static's own student_init_path (which reads --out's flat .pt directly).",
    )
    parser.add_argument("--teacher_checkpoint", default=DEFAULT_TEACHER_CHECKPOINT)
    args = parser.parse_args()

    student_experiment_name = f"cosmos_kd_student_{args.student_size}_libero"

    print(f"Loading teacher ({args.teacher_checkpoint}) on CPU...")
    teacher, _ = load_teacher(to_device="cpu", checkpoint=args.teacher_checkpoint)
    teacher.eval()

    print(f"Building student skeleton ({student_experiment_name}) on CPU...")
    student, _ = load_policy_model(
        to_device="cpu", experiment_name=student_experiment_name, skip_load_model=True
    )
    student.eval()

    report = init_student(teacher, student)
    print(
        f"Copied {report['student_num_blocks']} blocks from teacher's {report['teacher_num_blocks']} "
        f"(layer_map={report['layer_map']})"
    )
    print(f"teacher params: {report['teacher_params'] / 1e9:.3f}B  student params: {report['student_params'] / 1e9:.3f}B")

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(student.state_dict(), out_path)
    print(f"Saved student state dict to {out_path}")

    sidecar_path = out_path.with_suffix(out_path.suffix + ".json")
    sidecar_path.write_text(json.dumps(report, indent=2))
    print(f"Saved layer_map/param-count sidecar to {sidecar_path}")

    if args.dcp_out:
        dcp_out_dir = pathlib.Path(args.dcp_out)
        print(f"Also writing DCP-format checkpoint to {dcp_out_dir}/model/ ...")
        save_dcp_model_checkpoint(student, dcp_out_dir)
        print(f"Saved DCP checkpoint to {dcp_out_dir}/model/ -- point checkpoint.load_path at {dcp_out_dir}")


if __name__ == "__main__":
    main()
