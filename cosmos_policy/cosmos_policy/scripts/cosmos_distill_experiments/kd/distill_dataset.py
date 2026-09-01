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
On-disk format for a precomputed KD distillation dataset: shards of (xt, sigma, condition,
teacher_x0, x0) tuples -- the exact same tuple train_kd.py's live loop already computes fresh every
iteration via `teacher.training_step(...)` (see batch_prep.py's module docstring for where that
tuple comes from). `x0` is the real ground-truth target (same clean, frame-replace-injected target
the base non-KD job trains against) -- stored alongside `teacher_x0` so `train_kd_static.py` can
use `batch_prep.combined_kd_loss` exactly like the live path does, not distillation-only.
build_distill_dataset.py builds these once; train_kd_static.py trains a student against them
repeatedly, without ever loading the teacher again.

Each stored "example" is one INDIVIDUAL sample, not a whole micro-batch: `ShardWriter.add` takes
one full micro-batch per call (exactly as one `teacher.training_step(...)` call produces it) but
splits it, via `split_micro_batch`, before ever writing anything to disk -- so
`DistillShardDataset` yields individual examples, freely sample-able and mixable across whatever
micro-batch originally produced them, not sealed groups.

Splitting per sample means correctly re-slicing every `condition` field along its batch dimension,
including non-tensor ones like `data_type` (a shared enum, not a per-example tensor) -- getting
that per-field classification wrong fails silently, not loudly, which is why `split_micro_batch`
and its inverse, `join_examples_into_micro_batch`, are proven safe by a dedicated round-trip test
(distill_dataset_test.py) rather than by manual inspection of every field: splitting a micro-batch
and rejoining it must reproduce the original exactly, or the test fails.
`join_examples_into_micro_batch` is also what assembles a real training batch out of
individually-sampled examples at train time (see train_kd_static.py).

`condition` is persisted via `condition.to_dict(skip_underscore=False)` -- the same idiom
batch_prep.move_condition already uses for cross-device transfer -- plus the condition's own
fully-qualified class name, so it can be reconstructed via `cls(**stored_dict)` without this module
needing to import that class itself (it's defined deep in vendored `_src` config wiring, not
something KD-owned code should hardcode a path to).

Shards are plain `torch.save`'d lists of individual-example dicts, not one file per example --
avoids flooding the filesystem with tiny files (a real concern here, see the disk-quota
investigation this project already ran into) while still letting a dataset built once be reused
many times, sampled freely at the individual-example level.
"""

from __future__ import annotations

import importlib
import pathlib
import random
from typing import Any

import torch
from torch.utils.data import Dataset

_SHARD_GLOB = "shard_*.pt"


def _condition_class_path(condition: Any) -> str:
    cls = type(condition)
    return f"{cls.__module__}.{cls.__qualname__}"


def _import_class(dotted_path: str) -> type:
    module_path, _, class_name = dotted_path.rpartition(".")
    return getattr(importlib.import_module(module_path), class_name)


def condition_to_storable_dict(condition: Any) -> dict[str, Any]:
    """CPU-detaches every tensor field so shards don't pin GPU memory or a specific device."""
    return {
        k: (v.cpu() if torch.is_tensor(v) else v) for k, v in condition.to_dict(skip_underscore=False).items()
    }


def condition_from_storable_dict(condition_type: str, stored: dict[str, Any]) -> Any:
    return _import_class(condition_type)(**stored)


def split_micro_batch(example: dict[str, Any]) -> list[dict[str, Any]]:
    """Splits one stored micro-batch (batch dimension intact, as `ShardWriter.add` wrote it) into a
    list of individual, single-example dicts (each keeping a size-1 leading dimension, so
    `join_examples_into_micro_batch` can rejoin them with a plain `torch.cat`).

    Splitting rule, applied to every field of `xt`/`sigma`/`teacher_x0`/`condition` alike:
      - a tensor whose leading dimension equals the micro-batch's batch size is per-item ("sliced"):
        cut into batch_size single-example slices.
      - a tensor whose leading dimension is 1 is a broadcast value shared by the whole group --
        confirmed real, not hypothetical: `condition.use_video_condition` is exactly this (one
        random draw of whether to use video conditioning per micro-batch, not per example, stored
        as a shape-(1,) tensor regardless of batch size) -- copied unchanged into every example.
      - a non-tensor value (e.g. `condition.data_type`, an enum) is likewise shared and copied
        unchanged. This is the same is-it-shared distinction batch_prep.py's
        `move_batch_to_device`/`move_condition` make elsewhere, extended to cover tensors that are
        shaped like a broadcast value instead of just "not a tensor at all" -- an earlier version
        of this function used "tensor => per-item" as the whole rule and shipped with that gap; a
        real build against the real teacher hit it immediately (RuntimeError-free, but a batch_size
        mismatch this function's own leading-dimension check correctly refused to guess past).
      - any other leading dimension still raises rather than guessing.

    Which fields were sliced vs shared is recorded per example (`field_kinds`) because shape alone
    can't tell them apart anymore once every field has been reduced to a batch dimension of 1 --
    `join_examples_into_micro_batch` needs that record to reverse each field correctly.

    See distill_dataset_test.py's round-trip test for why this is trustworthy: splitting then
    rejoining via `join_examples_into_micro_batch` must reproduce the original micro-batch exactly,
    or the test fails loudly instead of this silently training on scrambled data.
    """
    batch_size = example["xt"].shape[0]

    def _split(value: Any, field_name: str) -> tuple[list[Any], str]:
        if torch.is_tensor(value):
            if value.shape[0] == batch_size:
                # `.clone()` is required, not cosmetic: `value[i:i+1]` is a VIEW, and PyTorch's
                # (de)serialization preserves a view's link to its full parent storage -- so without
                # cloning, every stored "single example" secretly still carries the ENTIRE
                # batch_size-example micro-batch's storage underneath it. Confirmed against a real
                # built dataset: loading materialized ~8x (exactly batch_size) more memory than the
                # examples' own logical byte size, which is what actually OOM'd a real kd_static
                # training run reading these shards -- not a hypothetical concern.
                return [value[i : i + 1].clone() for i in range(batch_size)], "sliced"
            if value.shape[0] == 1:
                return [value] * batch_size, "shared"
            raise ValueError(
                f"Field {field_name!r}: tensor with leading dimension {value.shape[0]} matches "
                f"neither this micro-batch's batch size ({batch_size}) nor a broadcastable 1 -- "
                f"refusing to guess whether it's per-item or shared."
            )
        return [value] * batch_size, "shared"

    field_kinds: dict[str, str] = {}
    top_level_split: dict[str, list[Any]] = {}
    for name in ("xt", "sigma", "teacher_x0", "x0"):
        values, kind = _split(example[name], name)
        top_level_split[name] = values
        field_kinds[name] = kind

    condition_split: dict[str, list[Any]] = {}
    for key, value in example["condition"].items():
        values, kind = _split(value, f"condition.{key}")
        condition_split[key] = values
        field_kinds[f"condition.{key}"] = kind

    return [
        dict(
            xt=top_level_split["xt"][i],
            sigma=top_level_split["sigma"][i],
            teacher_x0=top_level_split["teacher_x0"][i],
            x0=top_level_split["x0"][i],
            condition_type=example["condition_type"],
            condition={key: values[i] for key, values in condition_split.items()},
            field_kinds=field_kinds,
        )
        for i in range(batch_size)
    ]


def join_examples_into_micro_batch(examples: list[dict[str, Any]]) -> dict[str, Any]:
    """Inverse of `split_micro_batch`: stacks a list of individual (size-1-batch) examples back
    into one micro-batch. Uses each example's own `field_kinds` record (see `split_micro_batch`) to
    decide, per field, `torch.cat` (a "sliced" -- genuinely per-item -- field) vs. a
    strict-equality check that returns a single shared copy unchanged (a "shared" field, tensor or
    not) -- shape alone can't make that call once every field has a batch dimension of 1. Used both
    by the round-trip test that proves `split_micro_batch` is safe, and -- once that's trusted -- by
    whatever assembles a real training batch out of individually, randomly sampled examples.
    """
    if not examples:
        raise ValueError("Cannot join an empty list of examples.")
    condition_type = examples[0]["condition_type"]
    if any(ex["condition_type"] != condition_type for ex in examples):
        raise ValueError("Cannot join examples with different condition_type values.")
    field_kinds = examples[0]["field_kinds"]
    if any(ex["field_kinds"] != field_kinds for ex in examples):
        raise ValueError("Cannot join examples recorded with different field_kinds.")

    def _join(values: list[Any], kind: str) -> Any:
        if kind == "sliced":
            return torch.cat(values, dim=0)
        first = values[0]
        is_equal = torch.equal if torch.is_tensor(first) else (lambda a, b: a == b)
        if any(not is_equal(v, first) for v in values[1:]):
            raise ValueError(
                "A shared field differs across the examples being joined -- refusing to guess "
                "which value is correct."
            )
        return first

    condition_keys = examples[0]["condition"].keys()
    return dict(
        xt=_join([ex["xt"] for ex in examples], field_kinds["xt"]),
        sigma=_join([ex["sigma"] for ex in examples], field_kinds["sigma"]),
        teacher_x0=_join([ex["teacher_x0"] for ex in examples], field_kinds["teacher_x0"]),
        x0=_join([ex["x0"] for ex in examples], field_kinds["x0"]),
        condition_type=condition_type,
        condition={
            key: _join([ex["condition"][key] for ex in examples], field_kinds[f"condition.{key}"])
            for key in condition_keys
        },
    )


class ShardWriter:
    """Splits every incoming micro-batch into individual examples (`split_micro_batch`, proven
    safe by distill_dataset_test.py's round-trip test) before buffering them, and flushes to
    numbered shard files once `examples_per_shard` individual examples have accumulated (or on
    `close()`). Used by build_distill_dataset.py -- callers still hand it one whole micro-batch per
    `add()` call (exactly what a `teacher.training_step(...)` call produces); splitting into
    per-example storage is entirely internal to this class."""

    def __init__(self, out_dir: str, examples_per_shard: int):
        self.out_dir = pathlib.Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.examples_per_shard = examples_per_shard
        self._buffer: list[dict[str, Any]] = []
        self._next_shard_idx = 0
        self.num_written = 0

    def add(
        self, xt: torch.Tensor, sigma: torch.Tensor, condition: Any, teacher_x0: torch.Tensor, x0: torch.Tensor
    ) -> None:
        micro_batch = dict(
            xt=xt.cpu(),
            sigma=sigma.cpu(),
            condition_type=_condition_class_path(condition),
            condition=condition_to_storable_dict(condition),
            teacher_x0=teacher_x0.cpu(),
            x0=x0.cpu(),  # ground truth -- see batch_prep.py's combined_kd_loss
        )
        self._buffer.extend(split_micro_batch(micro_batch))
        if len(self._buffer) >= self.examples_per_shard:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        shard_path = self.out_dir / f"shard_{self._next_shard_idx:05d}.pt"
        torch.save(self._buffer, shard_path)
        print(f"Wrote {len(self._buffer)} examples to {shard_path}")
        self.num_written += len(self._buffer)
        self._buffer = []
        self._next_shard_idx += 1

    def close(self) -> None:
        self.flush()


class DistillShardDataset(Dataset):
    """Reads every `shard_*.pt` under `dataset_dir` into memory once at construction. A static KD
    dataset is built to be small (individual `xt`/`teacher_x0`/etc. tensors, not raw images) and
    reused, not to require lazy per-item disk reads.

    Each item is one INDIVIDUAL example (`xt`, `sigma`, `condition_type`, `condition`,
    `teacher_x0`, `x0`, every tensor field with a leading batch dimension of 1) -- `ShardWriter` already
    split every micro-batch before writing (see its docstring), so nothing here needs to. Sample
    indices freely (e.g. `random.sample`) and assemble a real training batch with
    `join_examples_into_micro_batch` -- do NOT wrap this in a `DataLoader` with its own
    batching/collation (see this module's docstring for why: `condition`'s non-tensor fields need
    the same shared-vs-per-item handling `join_examples_into_micro_batch` already gets right and
    `default_collate` doesn't know about).
    """

    def __init__(self, dataset_dir: str):
        self.dataset_dir = pathlib.Path(dataset_dir)
        shard_paths = sorted(self.dataset_dir.glob(_SHARD_GLOB))
        if not shard_paths:
            raise FileNotFoundError(f"No {_SHARD_GLOB} files found under {self.dataset_dir}")
        self._examples: list[dict[str, Any]] = []
        for shard_path in shard_paths:
            self._examples.extend(torch.load(shard_path, weights_only=False))

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self._examples[idx]


def sample_micro_batch(dataset: DistillShardDataset, batch_size: int, device: str) -> dict[str, Any]:
    """`batch_size` individually, independently, uniformly sampled examples from `dataset`,
    assembled into one micro-batch on `device` -- the exact random-sample + join_examples_into_
    micro_batch + move_batch_to_device + condition_from_storable_dict sequence train_kd_static.py
    used to do inline, factored out so every training script consuming a `DistillShardDataset` (the
    static path's real-data term, and any script's optional synthetic-batch term -- see
    build_synthetic_distill_dataset.py's module docstring for why a synthetic dataset is stored in
    this exact same format) shares one implementation instead of six near-identical copies.

    Returns a dict with `xt`/`sigma`/`teacher_x0`/`x0` (tensors, on `device`) and `condition` (a
    reconstructed condition OBJECT -- via `condition_from_storable_dict`, not the raw storable
    dict `join_examples_into_micro_batch` itself returns), ready to pass straight to
    `student.denoise(xt, sigma, condition)`.
    """
    # Local import: avoids batch_prep.py <-> distill_dataset.py becoming a circular import at
    # module-load time (batch_prep.py never imports this module, but keeping the dependency
    # one-directional and lazy here costs nothing and avoids relying on import order).
    from cosmos_policy.scripts.cosmos_distill_experiments.kd.batch_prep import move_batch_to_device

    indices = random.sample(range(len(dataset)), batch_size)
    micro_batch = join_examples_into_micro_batch([dataset[i] for i in indices])
    micro_batch = move_batch_to_device(micro_batch, device)
    # move_batch_to_device only moves top-level tensors; `condition` is itself a nested dict (not a
    # tensor), so its inner tensor fields need the same treatment separately -- it's a plain flat
    # dict of tensor/non-tensor values, the exact shape move_batch_to_device expects.
    condition_kwargs = move_batch_to_device(micro_batch["condition"], device)
    condition = condition_from_storable_dict(micro_batch["condition_type"], condition_kwargs)
    return dict(xt=micro_batch["xt"], sigma=micro_batch["sigma"], condition=condition, teacher_x0=micro_batch["teacher_x0"], x0=micro_batch["x0"])
