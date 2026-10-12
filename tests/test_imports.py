"""Every module imports with light dependencies only, and stubs are documented."""

from __future__ import annotations

import importlib
import inspect
import os
import pkgutil
import subprocess
import sys
from pathlib import Path

import pytest

import medcompress

REPO_ROOT = Path(__file__).resolve().parents[1]
HEAVY_MODULES = ("torch", "transformers", "peft", "accelerate", "datasets", "bitsandbytes")
MODULES = sorted(
    info.name for info in pkgutil.walk_packages(medcompress.__path__, prefix="medcompress.")
)

EXPECTED_MODULES = {
    "medcompress.config",
    "medcompress.registry",
    "medcompress.budgets",
    "medcompress.data.base",
    "medcompress.data.manifests",
    "medcompress.data.splits",
    "medcompress.data.masks",
    "medcompress.data.dedup",
    "medcompress.data.probes",
    "medcompress.models.base",
    "medcompress.models.loading",
    "medcompress.models.teacher_cache",
    "medcompress.compression.quantization.ptq",
    "medcompress.compression.quantization.qakd",
    "medcompress.compression.pruning.units",
    "medcompress.compression.pruning.apply",
    "medcompress.compression.pruning.criteria.base",
    "medcompress.compression.pruning.criteria.magnitude",
    "medcompress.compression.pruning.criteria.activation",
    "medcompress.compression.pruning.criteria.taylor",
    "medcompress.compression.pruning.criteria.output_similarity",
    "medcompress.compression.pruning.criteria.random",
    "medcompress.compression.pruning.criteria.clinical",
    "medcompress.compression.distillation.losses",
    "medcompress.compression.distillation.offline_kd",
    "medcompress.compression.recovery",
    "medcompress.evaluation.clinical",
    "medcompress.evaluation.small_findings",
    "medcompress.evaluation.rare_findings",
    "medcompress.evaluation.image_reliance",
    "medcompress.evaluation.calibration",
    "medcompress.evaluation.hallucination",
    "medcompress.evaluation.language",
    "medcompress.evaluation.efficiency",
    "medcompress.evaluation.statistics",
    "medcompress.utils.logging",
    "medcompress.utils.seed",
    "medcompress.utils.io",
    "medcompress.utils.hardware",
}


def test_expected_modules_exist() -> None:
    missing = EXPECTED_MODULES - set(MODULES)
    assert not missing, f"Missing modules: {sorted(missing)}"


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name: str) -> None:
    importlib.import_module(name)


def test_no_heavy_library_imported_at_top_level() -> None:
    code = (
        "import importlib, pkgutil, sys, medcompress\n"
        "for info in pkgutil.walk_packages(medcompress.__path__, prefix='medcompress.'):\n"
        "    importlib.import_module(info.name)\n"
        f"print(','.join(m for m in {HEAVY_MODULES!r} if m in sys.modules))\n"
    )
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True
    )
    assert result.stdout.strip() == "", f"Heavy modules imported: {result.stdout.strip()}"


def _public_members(module_name: str) -> list[tuple[str, object]]:
    module = importlib.import_module(module_name)
    return [
        (name, obj)
        for name, obj in vars(module).items()
        if not name.startswith("_")
        and (inspect.isfunction(obj) or inspect.isclass(obj))
        and getattr(obj, "__module__", None) == module_name
    ]


@pytest.mark.parametrize("name", MODULES)
def test_public_functions_and_classes_have_docstrings(name: str) -> None:
    module = importlib.import_module(name)
    assert inspect.getdoc(module), f"{name} has no module docstring"
    undocumented = [n for n, obj in _public_members(name) if not inspect.getdoc(obj)]
    assert not undocumented, f"{name}: missing docstrings for {undocumented}"


@pytest.mark.parametrize("name", MODULES)
def test_public_functions_have_return_annotations(name: str) -> None:
    missing = [
        n
        for n, obj in _public_members(name)
        if inspect.isfunction(obj)
        and inspect.signature(obj).return_annotation is inspect.Signature.empty
    ]
    assert not missing, f"{name}: missing return annotations for {missing}"


def test_stubs_raise_not_implemented_with_todo() -> None:
    from medcompress.compression.pruning.criteria.clinical import aggregate_importance
    from medcompress.evaluation.small_findings import small_finding_recall
    from medcompress.evaluation.statistics import aggregate_results

    for call in (
        lambda: aggregate_importance({}, {}),
        lambda: aggregate_results(Path("results"), Path("results/summary")),
        lambda: small_finding_recall([True], [0]),
    ):
        with pytest.raises(NotImplementedError, match="TODO"):
            call()


def test_all_criteria_registered() -> None:
    from medcompress.compression.pruning.criteria import CRITERIA

    assert set(CRITERIA) == {
        "magnitude",
        "activation",
        "taylor",
        "output_similarity",
        "random",
        "clinical",
    }
