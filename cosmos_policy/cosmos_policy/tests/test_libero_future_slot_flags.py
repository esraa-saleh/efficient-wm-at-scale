# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for the 2026-10-01 "physically remove the slots" fix, dataset side:
`LIBERODataset` gains three new independent flags (`use_future_proprio`,
`use_future_wrist_image`, `use_future_third_person_image`) so the FUTURE-prediction slots can be
dropped from the sequence entirely while the CURRENT-frame conditioning slots (gated by the
pre-existing `use_proprio`/`use_wrist_images`/`use_third_person_images`) stay.

LIBERODataset itself needs real on-disk HDF5 demo data to construct, which isn't available in a
quick unit test -- these tests instead verify the two things that are both (a) checkable without
real data and (b) exactly what a typo or wiring mistake in the fix would break: the new
constructor parameters exist with the right names/defaults, and they are stored as the expected
instance attributes, independent of the pre-existing current-frame flags.
"""

import inspect

from cosmos_policy.datasets.libero_dataset import LIBERODataset


def test_new_future_slot_flags_exist_in_constructor_with_default_true():
    """Default True means every existing run config that never sets these new flags is
    byte-for-byte unaffected -- same frames allocated as before."""
    sig = inspect.signature(LIBERODataset.__init__)
    for name in ("use_future_proprio", "use_future_wrist_image", "use_future_third_person_image"):
        assert name in sig.parameters, f"{name} missing from LIBERODataset.__init__"
        assert sig.parameters[name].default is True, f"{name} must default to True"


def test_new_flags_are_independent_of_the_legacy_current_frame_flags():
    """The new flags must be genuinely separate parameters from use_proprio/use_wrist_images/
    use_third_person_images -- not aliases or the same parameter reused, which would reintroduce
    the exact coupling (current+future tied together) this fix removes."""
    sig = inspect.signature(LIBERODataset.__init__)
    legacy = {"use_proprio", "use_wrist_images", "use_third_person_images"}
    new = {"use_future_proprio", "use_future_wrist_image", "use_future_third_person_image"}
    assert legacy.isdisjoint(new)
    assert legacy <= sig.parameters.keys()
    assert new <= sig.parameters.keys()


def test_init_source_assigns_new_flags_to_matching_instance_attributes():
    """Without constructing a real dataset (needs HDF5 files on disk), confirm __init__'s own
    source assigns self.use_future_proprio = use_future_proprio (etc.) -- catches a copy-paste
    mistake (e.g. assigning the legacy flag's value to the new attribute name) that a pure
    signature check wouldn't."""
    source = inspect.getsource(LIBERODataset.__init__)
    for name in ("use_future_proprio", "use_future_wrist_image", "use_future_third_person_image"):
        assert f"self.{name} = {name}" in source, f"self.{name} is not assigned from the {name} parameter"


def test_future_slot_guards_reference_the_new_flags_not_the_legacy_ones():
    """Confirms the three future-slot blocks in __getitem__ actually branch on the NEW flags.
    Regression anchor for the exact bug this fix targets: reusing use_proprio/use_wrist_images/
    use_third_person_images for the future blocks (the pre-fix behavior) silently couples current
    and future again."""
    import re

    source = inspect.getsource(LIBERODataset.__getitem__)

    def guard_for(comment_marker: str) -> str:
        idx = source.index(comment_marker)
        # the guard is the next "if self.<flag>:" line after the marker comment
        m = re.search(r"if self\.(\w+):", source[idx:idx + 200])
        assert m, f"no guard found after {comment_marker!r}"
        return m.group(1)

    assert guard_for("# Add future proprio") == "use_future_proprio"
    assert guard_for("# Add future wrist image") == "use_future_wrist_image"
    assert guard_for("# Add future primary image") == "use_future_third_person_image"
