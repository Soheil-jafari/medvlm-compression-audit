"""Shared data and hand-offs between accounts: manifests, verification, linking (CPU)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from medcompress.config import RunContext, experiment_config_path, load_config
from medcompress.handoffs import handoff_items
from medcompress.utils import bundles

REPO = Path(__file__).resolve().parents[1]
RULES = {
    "rules": [
        {
            "question_contains": "are there any abnormalities in the image",
            "values": {"polyp": "polyp"},
        }
    ]
}
ABNORMAL = "Are there any abnormalities in the image? Check all that are present."


def _run(script: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(REPO / "scripts" / script), *args],
        capture_output=True, text=True, cwd=REPO, check=False, timeout=300,
    )  # fmt: skip


def _image(path: Path, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    small = np.random.default_rng(seed).integers(0, 255, size=(12, 16, 3), dtype=np.uint8)
    Image.fromarray(small).resize((128, 96), Image.Resampling.BILINEAR).save(path)


def _row(img: str, answer: str) -> dict:
    return {
        "img_id": img, "question": "Is there a polyp?", "answer": "yes", "complexity": 1,
        "question_class": ["polyp_count"], "original": json.dumps([{"q": ABNORMAL, "a": answer}]),
    }  # fmt: skip


def _shared_data(root: Path) -> None:
    """A tiny shared-data folder as RC-D1 writes it: data/ + private/."""
    from medcompress.data.prepare import prepare, write_private_outputs

    data = root / "data"
    split_rows = {
        "train": [_row("t0", "polyp"), _row("t1", "none")],
        "test": [_row(f"v{i}", "polyp" if i % 2 else "none") for i in range(8)],
    }
    for split, rows in split_rows.items():
        path = data / "kvasir_vqa_x1" / f"{split}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        for i, row in enumerate(rows):
            _image(data / "kvasir_vqa_x1" / "images" / f"{row['img_id']}.jpg", seed=10 * i + 3)
    _image(data / "kvasir_seg" / "images" / "s1.jpg", seed=3)
    _image(data / "kvasir_seg" / "masks" / "s1.jpg", seed=99)
    config = {"name": "Kvasir-VQA-x1", "label_mapping": RULES}
    prepared = prepare(
        data, config, split_seed=1, close_bits=4, mask_threshold=4, confirm_threshold=0.9
    )
    write_private_outputs(prepared, root / "private", data)


# ------------------------------------------------------------------ the manifest itself


def test_a_manifest_finds_every_changed_missing_or_edited_file(tmp_path: Path) -> None:
    root = tmp_path / "upload" / "inner"  # Kaggle may mount a folder one level down
    (root / "a").mkdir(parents=True)
    (root / "a" / "x.bin").write_bytes(b"x" * 10)
    (root / "y.txt").write_text("y")
    made = bundles.write_bundle_manifest(root, kind="outcomes", meta={"experiment": "E2"})
    assert set(made.files) == {"a/x.bin", "y.txt"} and made.meta["commit"]
    assert bundles.verify_bundle(tmp_path / "upload").root == root
    with pytest.raises(bundles.BundleError, match="needs"):
        bundles.verify_bundle(root, kind="adapter")
    (root / "y.txt").write_text("changed")
    with pytest.raises(bundles.BundleError, match="hash differs: 1 \\(first: y.txt\\)"):
        bundles.verify_bundle(root)
    (root / "y.txt").write_text("y")
    (root / "a" / "x.bin").unlink()
    with pytest.raises(bundles.BundleError, match="missing: 1"):
        bundles.verify_bundle(root)
    (root / "a" / "x.bin").write_bytes(b"x" * 10)
    manifest = json.loads((root / bundles.MANIFEST_NAME).read_text())
    manifest["files"][0]["sha256"] = "0" * 64
    (root / bundles.MANIFEST_NAME).write_text(json.dumps(manifest))
    with pytest.raises(bundles.BundleError, match="changed after it was written"):
        bundles.verify_bundle(root)
    with pytest.raises(bundles.BundleError, match="outside the repository"):
        bundles.write_bundle_manifest(REPO / "docs", kind="outcomes")
    with pytest.raises(bundles.BundleError, match="no bundle_manifest"):
        bundles.verify_bundle(tmp_path / "upload" / "inner" / "a")


# ------------------------------------------------------------------ the shared data (Task 1)


def test_rc_d1_writes_one_shared_data_folder_that_later_runs_verify(tmp_path: Path) -> None:
    shared = tmp_path / "working" / "medcompress_data"
    incomplete = _run("prepare_data.py", "bundle", "--bundle-dir", str(shared))
    assert incomplete.returncode == 2 and "missing" in incomplete.stderr
    _shared_data(shared)
    made = _run("prepare_data.py", "bundle", "--bundle-dir", str(shared))
    assert made.returncode == 0, made.stderr
    manifest = json.loads((shared / bundles.MANIFEST_NAME).read_text())
    assert manifest["kind"] == "data" and len(manifest["commit"]) == 40
    assert "CC BY-NC 4.0" in manifest["licences"]["kvasir_vqa_x1"]
    assert "never public" in manifest["sharing"]
    names = {f["path"] for f in manifest["files"]}
    assert {
        "private/splits.json",
        "private/mask_links.json",
        "data/kvasir_seg/masks/s1.jpg",
    } <= names
    assert all(len(f["sha256"]) == 64 for f in manifest["files"])

    ok = _run("prepare_data.py", "verify", "--data-root", str(tmp_path / "working"))
    assert ok.returncode == 0 and "OK:" in ok.stdout, ok.stderr
    config: dict = {}
    bundles.attach_data(config, tmp_path / "working")
    assert config["paths"] == {
        "data_root": str(shared / "data"),
        "private_root": str(shared / "private"),
    }
    # the second RC-D1 pass: counts again from the shared data, with the first pass's settings
    out = tmp_path / "results"
    first = ("--dedup-threshold", "4", "--mask-threshold", "4", "--confirm-threshold", "0.9",
             "--results-root", str(out))  # fmt: skip
    counted = _run(
        "prepare_data.py", "counts", "--data-root", str(shared), "--split-seed", "1", *first
    )
    assert counted.returncode == 0, counted.stderr
    assert "qa_rows:train" in (out / "data" / "summary.csv").read_text()
    other = next(
        str(seed) for seed in range(2, 40) if _split_hash(shared, seed) != manifest["split_sha256"]
    )
    wrong = _run(
        "prepare_data.py", "counts", "--data-root", str(shared), "--split-seed", other, *first
    )
    assert wrong.returncode == 5 and "split_sha256" in wrong.stderr

    image = next((shared / "data" / "kvasir_vqa_x1" / "images").iterdir())
    image.write_bytes(image.read_bytes() + b"\0")  # one changed image stops everything
    bad = _run("prepare_data.py", "verify", "--data-root", str(shared))
    assert bad.returncode == 5 and "hash differs" in bad.stderr
    stage = _run("02_train_student.py", "--exp", "E2", "--seed", "0", "--data-root", str(shared))
    assert stage.returncode == 5 and "check failed" in stage.stderr
    assert "Traceback" not in stage.stderr


def test_the_shared_data_must_hold_the_pinned_split(tmp_path: Path) -> None:
    from medcompress.data.prepared import pinned_split_sha256

    shared = tmp_path / "medcompress_data"
    _shared_data(shared)
    made = _run("prepare_data.py", "bundle", "--bundle-dir", str(shared))
    assert made.returncode == 0, made.stderr
    held = json.loads((shared / bundles.MANIFEST_NAME).read_text())["split_sha256"]

    def pinning(value: str) -> dict:
        path = tmp_path / f"dataset_{value[:4]}.yaml"
        path.write_text(f'dataset:\n  split_sha256: "{value}"\n')
        return {"components": {"datasets": {"train_qa": str(path)}}}

    bundles.attach_data(pinning(held), shared)  # the pinned split: attached
    with pytest.raises(bundles.BundleError, match="the config pins ffff"):
        bundles.attach_data(pinning("f" * 64), shared)
    bundles.attach_data(pinning("TODO"), shared)  # an open pin checks nothing
    bundles.attach_data({}, shared)  # RC-D1's own steps name no dataset config
    real = load_config("experiments/E0.yaml")  # the real config pins RC-D1 run 4b's split
    with pytest.raises(bundles.BundleError, match="54dfa70a9aa99ea8"):
        bundles.attach_data(real, shared)
    dataset = load_config("data/kvasir_vqa_x1.yaml")["dataset"]
    assert pinned_split_sha256(dataset) == (
        "54dfa70a9aa99ea86262447acfd16fe4d4e871272781a31f39c9f330f4f7d090")  # fmt: skip
    assert pinned_split_sha256({"split_sha256": "TODO"}) is None
    from medcompress.config import ConfigError

    for bad in (0, "54dfa70a", "G" * 64):  # unquoted digits read as a number, short, not hex
        with pytest.raises(ConfigError, match="64 hex digits"):
            pinned_split_sha256({"split_sha256": bad})
    teacher = _run("train_teacher.py", "--data-root", str(shared), "--dry-run")
    assert teacher.returncode == 5 and "the config pins" in teacher.stderr  # before any model


def _split_hash(shared: Path, seed: int) -> str:
    from medcompress.config import config_root
    from medcompress.data.prepare import prepare

    dataset = load_config(config_root() / "data" / "kvasir_vqa_x1.yaml")["dataset"]
    return prepare(
        shared / "data", dataset, split_seed=seed, close_bits=4, mask_threshold=4,
        confirm_threshold=0.9,
    ).split_sha256  # fmt: skip


def test_a_stage_without_the_shared_data_stops_with_a_hint(tmp_path: Path) -> None:
    from medcompress.data.prepared import load_prepared

    with pytest.raises(Exception, match="--data-root"):
        load_prepared(tmp_path, {"name": "x"}, tmp_path / "private")


# ------------------------------------------------------------------ hand-offs (Task 2)


def _config(root: Path, **overrides: object) -> dict:
    config = load_config(experiment_config_path("E2"))
    config["paths"] = {
        name: str(root / name.removesuffix("_root"))
        for name in ("data_root", "cache_root", "checkpoint_root", "output_root", "results_root")
    }
    for key, value in overrides.items():
        section, name = key.split("__")
        config[section][name] = value
    return config


def test_an_e2_adapter_travels_to_another_account_and_is_checked(tmp_path: Path) -> None:
    producer = tmp_path / "account1"
    adapter = producer / "checkpoint" / "E2" / "seed1" / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"weights" * 100)
    (adapter / "adapter_config.json").write_text('{"r": 16}')
    out = tmp_path / "working" / "handoff" / "E2_seed1_adapter"
    exported = _run(
        "handoff.py", "export", "--kind", "adapter", "--exp", "E2", "--seed", "1",
        "-o", f"paths.checkpoint_root={producer / 'checkpoint'}", "--out", str(out),
    )  # fmt: skip
    assert exported.returncode == 0, exported.stderr
    manifest = json.loads((out / bundles.MANIFEST_NAME).read_text())
    assert (manifest["experiment"], manifest["seed"], manifest["kind"]) == ("E2", 1, "adapter")
    assert len(manifest["config_sha256"]) == 64 and len(manifest["commit"]) == 40
    assert manifest["compat"]["inputs.image_longest_side"] == 672
    assert manifest["compat"]["student_revision"] == "66285546d2b821cf421d4f5eb2576359d3770cd3"
    assert manifest["items"] == [{"root": "checkpoint_root", "path": "E2/seed1/adapter"}]
    assert _run("handoff.py", "verify", str(out)).returncode == 0

    consumer = _config(tmp_path / "account2")
    made = bundles.attach_handoff(consumer, out)
    linked = Path(consumer["paths"]["checkpoint_root"]) / "E2" / "seed1" / "adapter"
    assert made.label == "E2 seed 1 adapter" and linked.is_symlink()
    assert (linked / "adapter_config.json").read_text() == '{"r": 16}'
    bundles.attach_handoff(consumer, out)  # the same link again is fine

    with pytest.raises(bundles.BundleError, match="image_longest_side: 672 there, 448 here"):
        bundles.attach_handoff(_config(tmp_path / "account3", inputs__image_longest_side=448), out)
    local = _config(tmp_path / "account4")
    (Path(local["paths"]["checkpoint_root"]) / "E2" / "seed1" / "adapter").mkdir(parents=True)
    with pytest.raises(bundles.BundleError, match="already exists here"):
        bundles.attach_handoff(local, out)

    (out / "checkpoint_root" / "E2" / "seed1" / "adapter" / "adapter_config.json").write_text("{}")
    stage = _run("02_train_student.py", "--exp", "E2", "--seed", "1", "--from", str(out))
    assert stage.returncode == 5 and "hash differs" in stage.stderr
    assert "Traceback" not in stage.stderr


def test_where_each_kind_of_hand_off_lives(tmp_path: Path) -> None:
    config = _config(tmp_path)
    items = {
        ("teacher_cache", None): [("cache_root", "teacher")],
        ("adapter", RunContext("E1", seed=2)): [("checkpoint_root", "E1/seed2/adapter")],
        ("pruned", RunContext("E7", seed=0, budget="B2")): [
            ("checkpoint_root", "E7/B2/seed0/pruned")
        ],
        ("quantized", RunContext("E3", seed=1, variant="w4")): [
            ("checkpoint_root", "E3/w4/nobudget/seed1/quantized")
        ],
        ("outcomes", RunContext("E5", seed=0, budget="B1", variant="taylor")): [
            ("output_root", "E5/taylor/B1/seed0")
        ],
        ("resume", RunContext("E2", seed=0)): [("checkpoint_root", "E2/seed0/resume")],
    }
    for (kind, run), expected in items.items():
        assert handoff_items(kind, config, run) == expected
    with pytest.raises(Exception, match="saves no adapter"):
        handoff_items("adapter", config, RunContext("E7", seed=0, budget="B2"))


def test_a_resume_hand_off_is_copied_so_training_can_write_into_it(tmp_path: Path) -> None:
    producer = _config(tmp_path / "a")
    state = Path(producer["paths"]["checkpoint_root"]) / "E2" / "seed0" / "resume"
    state.mkdir(parents=True)
    (state / "progress.json").write_text('{"step": 3}')
    out = tmp_path / "out"
    meta = bundles.handoff_meta(producer, experiment="E2", seed=0)
    bundles.export_handoff(
        producer, [("checkpoint_root", "E2/seed0/resume")], out, kind="resume", meta=meta
    )
    consumer = _config(tmp_path / "b")
    bundles.attach_handoff(consumer, out, kind="resume")
    copied = Path(consumer["paths"]["checkpoint_root"]) / "E2" / "seed0" / "resume"
    assert copied.is_dir() and not copied.is_symlink()
    (copied / "progress.json").write_text('{"step": 4}')  # writable, the dataset is untouched
    assert (out / "checkpoint_root" / "E2" / "seed0" / "resume" / "progress.json").read_text() == (
        '{"step": 3}'
    )
    with pytest.raises(bundles.BundleError, match="not empty"):
        bundles.export_handoff(producer, [], out, kind="resume", meta=meta)
