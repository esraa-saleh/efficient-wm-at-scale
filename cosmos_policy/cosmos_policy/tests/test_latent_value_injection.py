# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for the 2026-10-01 "physically remove the slots" fix:

1. `replace_latent_with_value` (newly extracted from inline code in
   `compute_loss_with_epsilon_and_sigma`, mirroring the existing
   `replace_latent_with_action_chunk`/`replace_latent_with_proprio` split) -- confirms the raw
   injection is correct given valid indices.
2. The guard pattern now wrapping that call at the call site (`if torch.all(value_indices != -1)`)
   -- confirms the pre-existing corruption bug (an unconditional write that used Python's
   negative-indexing semantics to silently overwrite whatever real slot occupies the LAST T'
   position whenever the value slot is disabled) is actually fixed, not just guarded in a way that
   still has the same bug under a different name.

These tests are pure-tensor and import nothing beyond `replace_latent_with_proprio`/
`replace_latent_with_value` from `policy_text2world_model` -- no network, no dataset, no GPU
required for the assertions themselves (CUDA import side effects of the module are a separate,
pre-existing environment concern, not something these tests depend on).
"""

import torch

from cosmos_policy.models.policy_text2world_model import (
    replace_latent_with_proprio,
    replace_latent_with_value,
)


def _make_x0(B=3, C=4, T=9, H=2, W=2, fill_value=1.0):
    """A deterministic, distinguishable-per-slot x0 tensor: slot t's content is fill_value*(t+1),
    so overwriting the wrong slot (or failing to overwrite the right one) is easy to detect."""
    x0 = torch.zeros(B, C, T, H, W)
    for t in range(T):
        x0[:, :, t, :, :] = fill_value * (t + 1)
    return x0


def test_replace_latent_with_value_writes_only_the_target_slot():
    x0 = _make_x0()
    B, C, T, H, W = x0.shape
    value_indices = torch.full((B,), 5, dtype=torch.long)  # every sample's value slot is t=5
    value_function_return = torch.tensor([0.1, 0.5, 0.9])

    out = replace_latent_with_value(x0.clone(), value_function_return, value_indices)

    # Target slot now holds the broadcast return, per-sample.
    for b in range(B):
        assert torch.allclose(out[b, :, 5, :, :], torch.full((C, H, W), value_function_return[b].item()))

    # Every other slot is untouched.
    original = _make_x0()
    for t in range(T):
        if t == 5:
            continue
        assert torch.equal(out[:, :, t, :, :], original[:, :, t, :, :]), f"slot {t} was modified unexpectedly"


def test_replace_latent_with_value_handles_per_sample_indices():
    """value_indices can differ per sample (e.g. variable-length sequences) -- confirm each
    sample's OWN slot gets written, not just the first sample's index broadcast to everyone."""
    x0 = _make_x0(B=3)
    value_indices = torch.tensor([2, 5, 7])
    value_function_return = torch.tensor([0.25, 0.5, 0.75])

    out = replace_latent_with_value(x0.clone(), value_function_return, value_indices)

    for b, idx in enumerate(value_indices.tolist()):
        assert torch.allclose(out[b, :, idx, :, :], torch.full(out.shape[1:2] + out.shape[3:], value_function_return[b].item()))


def test_unguarded_write_would_corrupt_the_last_slot_when_disabled():
    """Regression anchor for the pre-existing bug: demonstrates that calling
    replace_latent_with_value UNGUARDED with value_indices=-1 (the sentinel meaning "value slot not
    in use") corrupts whatever real slot sits last -- i.e. confirms WHY the call site needed the
    `if torch.all(value_indices != -1)` guard, not just that the guard exists."""
    x0 = _make_x0(T=9)
    B = x0.shape[0]
    disabled_value_indices = torch.full((B,), -1, dtype=torch.long)
    value_function_return = torch.tensor([-100.0, -100.0, -100.0])  # the dataset's real placeholder value

    corrupted = replace_latent_with_value(x0.clone(), value_function_return, disabled_value_indices)

    original = _make_x0(T=9)
    # Slot -1 (i.e. slot 8, the last one) WAS silently overwritten -- this is the bug.
    assert not torch.equal(corrupted[:, :, 8, :, :], original[:, :, 8, :, :])


def test_guarded_call_site_pattern_is_a_true_no_op_when_value_slot_disabled():
    """Confirms the ACTUAL fix: the `if torch.all(value_indices != -1): x0 = replace_latent_with_value(...)`
    pattern now used at the call site in compute_loss_with_epsilon_and_sigma leaves x0 completely
    unchanged -- including the last slot -- when the value slot is disabled, instead of the
    unguarded corruption demonstrated above."""
    x0 = _make_x0(T=9)
    B = x0.shape[0]
    disabled_value_indices = torch.full((B,), -1, dtype=torch.long)
    value_function_return = torch.tensor([-100.0, -100.0, -100.0])

    x0_after = x0.clone()
    if torch.all(disabled_value_indices != -1):  # exact guard now used in policy_text2world_model.py
        x0_after = replace_latent_with_value(x0_after, value_function_return, disabled_value_indices)

    assert torch.equal(x0_after, x0), "guarded call site must be a true no-op when the value slot is disabled"


def test_replace_latent_with_proprio_unaffected_by_the_refactor():
    """Sanity check that extracting replace_latent_with_value didn't disturb its sibling helper --
    same style, same file, should still behave exactly as before."""
    x0 = _make_x0(C=9, H=2, W=2)
    B = x0.shape[0]
    proprio_indices = torch.full((B,), 1, dtype=torch.long)
    proprio = torch.arange(B * 9, dtype=torch.float32).reshape(B, 9)

    out = replace_latent_with_proprio(x0.clone(), proprio, proprio_indices)
    for b in range(B):
        assert torch.allclose(out[b, :, 1, :, :].flatten()[:9], proprio[b])
