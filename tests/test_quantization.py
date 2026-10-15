"""The quantization grid, the packed storage and the GPTQ reference (NumPy only)."""

from __future__ import annotations

import numpy as np
import pytest

from medcompress.compression.quantization import grid
from medcompress.compression.quantization.gptq import (
    HessianAccumulator,
    gptq_numpy,
    output_error,
)


def _layer(seed: int, out: int = 12, inp: int = 64) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(out, inp))


# ------------------------------------------------------------------------- grid


@pytest.mark.parametrize("bits", [4, 8])
def test_round_to_nearest_stays_on_the_grid_and_close(bits: int) -> None:
    w = _layer(0)
    codes, scales = grid.round_to_nearest(w, bits, 16)
    assert codes.dtype == np.uint8 and codes.max() <= grid.max_code(bits)
    back = grid.dequantize(codes, scales, bits, 16)
    # every weight is within half a step of its group's grid, except the clipped top value
    step = grid.expand_groups(scales, 16)
    assert np.all(np.abs(back - w) <= step / 2 + 1e-12)
    # re-quantizing the rounded weights with the same scales gives the same codes
    assert np.array_equal(grid.quantize_codes(back, scales, bits, 16), codes)


def test_the_grid_is_the_symmetric_gptq_grid() -> None:
    w = np.array([[-1.5, 0.0, 0.75, 1.5]])
    scales = grid.symmetric_scales(w, 4, 4)
    assert scales[0, 0] == pytest.approx(2 * 1.5 / 15)
    assert grid.zero_point(4) == 8 and grid.max_code(4) == 15
    codes = grid.quantize_codes(w, scales, 4, 4)
    assert codes[0, 1] == 8  # zero is exact
    assert grid.symmetric_scales(np.zeros((2, 4)), 4, 4).tolist() == [[1.0], [1.0]]


def test_bad_shapes_are_refused() -> None:
    with pytest.raises(ValueError, match="multiple of group_size"):
        grid.symmetric_scales(np.zeros((2, 10)), 4, 4)
    with pytest.raises(ValueError, match="bits"):
        grid.check_shape(8, 3, 4)


@pytest.mark.parametrize("bits", [4, 8])
def test_packing_round_trips_and_uses_the_gptq_layout(bits: int) -> None:
    codes = np.random.default_rng(1).integers(0, 1 << bits, size=(5, 32), dtype=np.uint8)
    packed = grid.pack_codes(codes, bits)
    assert packed.dtype == np.int32 and packed.shape == (32 * bits // 32, 5)
    assert np.array_equal(grid.unpack_codes(packed, bits, 32), codes)
    # the first column sits in the lowest bits of the first word
    word = int(packed.view(np.uint32)[0, 2])
    assert word & grid.max_code(bits) == codes[2, 0]
    assert (word >> bits) & grid.max_code(bits) == codes[2, 1]
    with pytest.raises(ValueError, match="fit"):
        grid.pack_codes(np.full((1, 32), 1 << bits, dtype=np.uint32), bits)


def test_packed_bytes() -> None:
    # 4 bit: half a byte per weight plus one fp16 scale per row and group
    assert grid.packed_bytes(8, 256, 4, 128) == 8 * 256 // 2 + 8 * 2 * 2


# ---------------------------------------------------------------------- Hessian


def test_the_hessian_does_not_depend_on_how_tokens_are_batched() -> None:
    x = np.random.default_rng(2).normal(size=(30, 8))
    one, two = HessianAccumulator(8), HessianAccumulator(8)
    one.add(x)
    two.add(x[:7])
    two.add(x[7:])
    two.add(np.zeros((0, 8)))
    assert np.allclose(one.hessian, two.hessian)
    assert np.allclose(one.hessian, 2 * x.T @ x / 30) and one.tokens == 30


# ------------------------------------------------------------------------- GPTQ


@pytest.mark.parametrize("bits", [4, 8])
def test_gptq_with_uncorrelated_inputs_is_plain_rounding(bits: int) -> None:
    w = _layer(3)
    result = gptq_numpy(w, np.eye(64), bits=bits, group_size=16)
    codes, scales = grid.round_to_nearest(w, bits, 16)
    assert np.array_equal(result.codes, codes) and np.allclose(result.scales, scales)


@pytest.mark.parametrize("bits", [4, 8])
def test_gptq_beats_plain_rounding_on_correlated_inputs(bits: int) -> None:
    rng = np.random.default_rng(4)
    w = _layer(5)
    mix = rng.normal(size=(64, 64))
    x = rng.normal(size=(400, 64)) @ mix  # strongly correlated input features
    acc = HessianAccumulator(64)
    acc.add(x)
    result = gptq_numpy(w, acc.hessian, bits=bits, group_size=16, block_size=24)
    codes, scales = grid.round_to_nearest(w, bits, 16)
    rtn = grid.dequantize(codes, scales, bits, 16)
    gptq_err = output_error(w, result.dequantized(), acc.hessian)
    rtn_err = output_error(w, rtn, acc.hessian)
    assert gptq_err < 0.9 * rtn_err
    assert result.codes.max() <= grid.max_code(bits) and result.output_error > 0
    # the outputs on the calibration inputs move less than with plain rounding
    assert np.linalg.norm(x @ (w - result.dequantized()).T) < np.linalg.norm(x @ (w - rtn).T)


def test_gptq_does_not_depend_on_the_block_size() -> None:
    rng = np.random.default_rng(6)
    w, x = _layer(7), rng.normal(size=(300, 64)) @ rng.normal(size=(64, 64))
    acc = HessianAccumulator(64)
    acc.add(x)
    small = gptq_numpy(w, acc.hessian, bits=4, group_size=64, block_size=8)
    large = gptq_numpy(w, acc.hessian, bits=4, group_size=64, block_size=64)
    # with one group the scales are taken before any update, so the blocking is exact
    assert np.array_equal(small.codes, large.codes)


def test_gptq_zeroes_inputs_that_never_fire() -> None:
    rng = np.random.default_rng(8)
    x = rng.normal(size=(200, 64))
    x[:, 5] = 0.0
    acc = HessianAccumulator(64)
    acc.add(x)
    result = gptq_numpy(_layer(9), acc.hessian, bits=4, group_size=16)
    assert np.all(result.dequantized()[:, 5] == 0.0)


# ------------------------------------------------------------------ PTQ settings


def _ptq_config(**recipe: object) -> dict:
    from medcompress.config import load_config

    config = load_config("experiments/E3.yaml")
    config["quantization"].update(recipe)
    return config


def test_recipes_resolve_per_arm() -> None:
    from medcompress.compression.quantization.ptq import PTQRecipe
    from medcompress.config import load_config

    w4 = PTQRecipe.from_config(_ptq_config(), "w4")
    assert (w4.backend, w4.weight_bits, w4.group_size, w4.calibration_size) == ("gptq", 4, 128, 128)
    assert w4.targets == (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    w8a8 = PTQRecipe.from_config(_ptq_config(), "w8a8")
    assert w8a8.backend == "llm_int8" and w8a8.group_size is None and w8a8.calibration_size is None
    for exp in ("E8", "E9"):
        recipe = PTQRecipe.from_config(load_config(f"experiments/{exp}.yaml"))
        assert recipe.name == "w4" and recipe.backend == "gptq"


def test_recipe_errors_name_the_open_setting() -> None:
    from medcompress.compression.quantization.ptq import PTQRecipe, kernel_class_from_config
    from medcompress.config import ConfigError

    with pytest.raises(ConfigError, match="unknown quantization recipe"):
        PTQRecipe.from_config(_ptq_config(), "w2")
    config = _ptq_config()
    config["quantization"]["calibration"]["size"] = "TODO"
    with pytest.raises(ConfigError, match="calibration.size"):
        PTQRecipe.from_config(config, "w4")
    PTQRecipe.from_config(config, "w8a8")  # LLM.int8 needs no calibration
    with pytest.raises(ConfigError, match="kernel_class"):
        kernel_class_from_config(_ptq_config(kernel_class="TODO"))
    assert kernel_class_from_config(_ptq_config(kernel_class="reference")) == "reference"


def test_rc_q0_set_the_kernel_class_and_the_library_versions() -> None:
    from medcompress.compression.quantization.kernels import TORCH_KERNEL_CLASS
    from medcompress.compression.quantization.ptq import (
        kernel_class_from_config,
        library_versions_from_config,
    )
    from medcompress.config import ConfigError, load_config

    for exp in ("E3", "E4", "E8", "E9"):  # every quantized arm reads the same block
        config = load_config(f"experiments/{exp}.yaml")
        assert kernel_class_from_config(config) == TORCH_KERNEL_CLASS
        versions = library_versions_from_config(config)
        assert versions["gptqmodel"] == "7.6.0" and versions["bitsandbytes"] == "0.50.2"
        assert all(isinstance(v, str) for v in versions.values())  # "5.19" must not be a float
    with pytest.raises(ConfigError, match="library_versions"):
        library_versions_from_config(_ptq_config(library_versions="TODO"))
    with pytest.raises(ConfigError, match="map a package"):
        library_versions_from_config(_ptq_config(library_versions=["gptqmodel==7.6.0"]))
    with pytest.raises(ConfigError, match="bitsandbytes"):
        library_versions_from_config(_ptq_config(library_versions={"gptqmodel": "7.6.0"}))


def test_installed_versions_name_missing_packages_none() -> None:
    from medcompress.compression.quantization.ptq import installed_versions

    found = installed_versions(["numpy", "no-such-package-for-sure"])
    assert found["numpy"] == np.__version__ and found["no-such-package-for-sure"] is None


def test_quantized_checkpoints_round_trip(tmp_path) -> None:  # noqa: ANN001
    from medcompress.compression.quantization.ptq import (
        PTQRecipe,
        QuantizedMatrix,
        parent_of,
        quantized_dir,
        read_quantized,
        save_quantized,
    )
    from medcompress.config import RunContext

    recipe = PTQRecipe.from_config(_ptq_config(), "w4")
    rng = np.random.default_rng(10)
    records = {}
    for key, shape in (("layer0.self_attn.q_proj", (8, 256)), ("layer1.mlp.down_proj", (4, 128))):
        codes, scales = grid.round_to_nearest(rng.normal(size=shape), 4, 128)
        records[key] = QuantizedMatrix(key, codes, scales, 4, 128)
    run = RunContext("E8", seed=2, budget="B2")
    config = {"paths": {"checkpoint_root": str(tmp_path)}}
    folder = save_quantized(
        records, recipe, quantized_dir(config, run), start_from=parent_of(run), extra={"x": 1}
    )
    assert folder == tmp_path / "E8" / "B2" / "seed2" / "quantized"
    manifest, back = read_quantized(folder)
    assert manifest["start_from"] == {"experiment": "E7", "budget": "B2", "seed": 2}
    assert manifest["x"] == 1 and manifest["quantized_weight_bytes"] == sum(
        grid.packed_bytes(*r.codes.shape, 4, 128) for r in records.values()
    )
    for key, rec in records.items():
        assert np.array_equal(back[key].codes, rec.codes)
        # scales are stored in 16 bit, as the kernels use them
        assert np.allclose(back[key].scales, rec.scales, rtol=1e-3)
    empty = save_quantized(
        {},
        PTQRecipe.from_config(_ptq_config(), "w8a8"),
        tmp_path / "int8",
        start_from=parent_of(RunContext("E3", seed=0, variant="w8a8")),
    )
    assert read_quantized(empty)[1] == {}
    assert parent_of(RunContext("E9", seed=0))["experiment"] == "E0"
