"""Image-disjoint re-split (D02 amendment 2026-10-18): grouping, stratification, checks (CPU)."""

from __future__ import annotations

import random
from collections import Counter

import pytest

from medcompress.data.base import VQASample
from medcompress.data.dedup import min_distances, near_duplicate_groups
from medcompress.data.splits import assign_groups_stratified, make_image_disjoint_splits

RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def _fp(bits: int) -> str:
    return f"{bits:016x}"


def test_near_duplicate_groups_are_transitive_and_order_free() -> None:
    fingerprints = {
        "img/c": _fp(0b0111),  # 1 bit from b, 2 bits from a
        "img/a": _fp(0b0001),
        "img/b": _fp(0b0011),  # 1 bit from a
        "img/d": _fp(0xFFFF_0000_0000_0000),  # far from all
        "img/e": _fp(0xFFFF_0000_0000_0001),  # 1 bit from d
    }
    groups = near_duplicate_groups(fingerprints, threshold=1)
    assert groups["img/a"] == groups["img/b"] == groups["img/c"] == "img/a"  # a~b~c, not a~c
    assert groups["img/d"] == groups["img/e"] == "img/d" != groups["img/a"]
    shuffled = dict(sorted(fingerprints.items(), reverse=True))
    assert near_duplicate_groups(shuffled, threshold=1) == groups
    assert len(set(near_duplicate_groups(fingerprints, threshold=0).values())) == 5
    assert min_distances({"x": _fp(0)}, {"y": _fp(0b111)}) == {"y": 3}


def _groups(seed: int = 7) -> tuple[dict[str, int], dict[str, frozenset[str]]]:
    rng = random.Random(seed)
    sizes = {f"g{i:03d}": rng.choice((1, 1, 1, 2, 3)) for i in range(400)}
    names = sorted(sizes)
    labels: dict[str, set[str]] = {g: set() for g in names}
    for g in rng.sample(names, 3):
        labels[g].add("answer:rare")
    for g in rng.sample(names, 2):
        labels[g].add("answer:two_groups")
    for g in rng.sample(names, 40):
        labels[g].add("answer:polyp")
    for g in rng.sample(names, 80):
        labels[g].add("masked")
    return sizes, {g: frozenset(v) for g, v in labels.items()}


def test_groups_are_stratified_by_finding_and_mask_and_balanced_by_images() -> None:
    sizes, labels = _groups()
    assigned = assign_groups_stratified(sizes, labels, RATIOS, seed=20261012)
    assert set(assigned) == set(sizes)  # every group exactly once
    per_label = {
        label: Counter(assigned[g] for g, ls in labels.items() if label in ls)
        for label in ("answer:rare", "answer:polyp", "masked", "answer:two_groups")
    }
    assert per_label["answer:rare"] == {"train": 1, "val": 1, "test": 1}  # every split has it
    for label, total in (("answer:polyp", 40), ("masked", 80)):
        for split, ratio in RATIOS.items():
            assert abs(per_label[label][split] - ratio * total) <= 1.5, (label, split)
    assert sum(per_label["answer:two_groups"].values()) == 2  # too few for three splits
    total = sum(sizes.values())
    for split, ratio in RATIOS.items():
        share = sum(n for g, n in sizes.items() if assigned[g] == split) / total
        assert abs(share - ratio) <= 0.02, (split, share)
    assert assign_groups_stratified(sizes, labels, RATIOS, seed=20261012) == assigned
    assert assign_groups_stratified(sizes, labels, RATIOS, seed=1) != assigned
    with pytest.raises(ValueError, match="sum to 1"):
        assign_groups_stratified(sizes, labels, {"train": 0.7, "val": 0.2}, seed=0)


def test_no_image_and_no_group_is_in_two_splits() -> None:
    samples = [
        VQASample(f"s{i}", "d", f"images/{i % 30}.jpg", f"q{i}", "a", split="train")
        for i in range(120)
    ]
    group_of = {f"images/{i}.jpg": f"images/{i - i % 3}.jpg" for i in range(30)}
    splits = make_image_disjoint_splits(samples, group_of, {}, RATIOS, seed=3)
    where: dict[str, set[str]] = {}
    groups: dict[str, set[str]] = {}
    for name, members in splits.items():
        for s in members:
            assert s.split == name
            where.setdefault(s.image_path, set()).add(name)
            groups.setdefault(group_of[s.image_path], set()).add(name)
    assert all(len(v) == 1 for v in where.values()) and all(len(v) == 1 for v in groups.values())
    assert sum(len(m) for m in splits.values()) == 120
    with pytest.raises(ValueError, match="no group"):
        make_image_disjoint_splits(samples, {}, {}, RATIOS, seed=3)
