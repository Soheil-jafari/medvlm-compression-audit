"""Which settings a hand-off is checked against depends on its kind (2026-11-07).

E2 seed 0 stopped at once: the teacher cache was written by E0, whose config names no student
(``TODO``), and E2's names the real one. The teacher's hand-offs do not depend on the student.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from medcompress.config import load_experiment
from medcompress.utils.bundles import Bundle, BundleError, check_compat, compat_values

# the settings recorded in the committed cache's manifest (d6ba0cb9..., written by RC-E0b)
E0_CACHE = {
    "inputs.image_longest_side": 672,
    "seeds.data_split": 20261012,
    "student": "TODO",
    "student_revision": None,
    "teacher": "models/qwen2_5_vl_7b_teacher.yaml",
    "teacher_revision": "cc594898137f460bfe9f0759e9844b3ce807cfb5",
    "training.data_subset": None,
}


def _bundle(kind: str, compat: dict, experiment: str | None = None) -> Bundle:
    experiment = experiment or ("E0" if kind.startswith("teacher") else "E2")
    return Bundle(root=Path("/in/x"), kind=kind, meta={"experiment": experiment, "compat": compat},
                  files={})  # fmt: skip


def test_the_committed_teacher_cache_attaches_to_e2() -> None:
    e2, _ = load_experiment("E2")
    check_compat(_bundle("teacher_cache", E0_CACHE), e2)  # no stop
    check_compat(_bundle("teacher_adapter", E0_CACHE), e2)  # RC-R1 and E9 read it too


@pytest.mark.parametrize(
    ("key", "value"),
    [("teacher", "models/qwen3_vl_8b_teacher.yaml"), ("inputs.image_longest_side", 448),
     ("teacher_revision", "0" * 40), ("seeds.data_split", 1)],
)  # fmt: skip
def test_a_cache_of_another_teacher_or_image_side_is_still_refused(key: str, value: object) -> None:
    e2, _ = load_experiment("E2")
    with pytest.raises(BundleError, match=key.replace(".", r"\.")):
        check_compat(_bundle("teacher_cache", {**E0_CACHE, key: value}), e2)


def test_an_adapter_of_another_student_is_still_refused() -> None:
    e2, _ = load_experiment("E2")
    other = {**compat_values(e2), "student": "models/qwen3_vl_2b_student.yaml",
             "student_revision": "1" * 40}  # fmt: skip
    for kind in ("adapter", "resume"):
        with pytest.raises(BundleError, match="student"):
            check_compat(_bundle(kind, other), e2)
    check_compat(_bundle("adapter", compat_values(e2)), e2)  # its own student attaches


def test_a_todo_is_never_compared_as_a_value() -> None:
    e2, _ = load_experiment("E2")
    open_student = {**compat_values(e2), "student": "TODO", "student_revision": "TODO"}
    check_compat(_bundle("adapter", open_student), e2)
    e0, _ = load_experiment("E0")
    assert compat_values(e0)["student"] != "TODO"  # E0's placeholder is no longer recorded


# ---------------------------------------------- E9's quantized decoder (2026-11-13)

# what the e9-seed0-quantized manifest recorded (exported with the base config at 54b5b6d)
E9_AT_54B5B6D = {
    **E0_CACHE,
    "student": "models/qwen2_5_vl_3b_student.yaml",
    "student_revision": "66285546d2b821cf421d4f5eb2576359d3770cd3",
}


def test_e9s_quantized_handoff_is_compared_as_the_teachers() -> None:
    e9, _ = load_experiment("E9")
    check_compat(_bundle("quantized", E9_AT_54B5B6D, "E9"), e9)  # the false stop is gone
    check_compat(_bundle("outcomes", E9_AT_54B5B6D, "E9"), e9)
    with pytest.raises(BundleError, match="teacher_revision"):  # another teacher still stops
        check_compat(_bundle("quantized", {**E9_AT_54B5B6D, "teacher_revision": "0" * 40},
                             "E9"), e9)  # fmt: skip
    e3, _ = load_experiment("E3")  # a student's quantized decoder still compares the student
    other = {**compat_values(e3), "student": "models/qwen3_vl_2b_student.yaml"}
    with pytest.raises(BundleError, match="student"):
        check_compat(_bundle("quantized", other, "E3"), e3)


def test_each_handoff_records_its_own_runs_settings() -> None:
    from medcompress.card_runner import run_config
    from medcompress.cards import Run

    base = {"model": {"config": "models/qwen2_5_vl_3b_student.yaml"}}
    e9 = compat_values(run_config(Run("E9"), base))
    assert e9["student"] == "models/qwen2_5_vl_7b_teacher.yaml"  # E9's model is the teacher
    assert e9["student_revision"] == "cc594898137f460bfe9f0759e9844b3ce807cfb5"
    e10 = compat_values(run_config(Run("E10"), base))
    assert e10["student"] == "models/qwen3_vl_2b_student.yaml"  # not the 3B of base.yaml
    assert run_config(Run("E99"), base) is base  # unknown: the session's config
