"""Model loading rules, yes/no tokens and adapter key remapping (no torch needed; runs in CI)."""

from __future__ import annotations

import pytest

from medcompress.config import config_root, load_config
from medcompress.models.loading import (
    ModelConfigError,
    ModelSpec,
    check_loadable,
    load_model,
    remap_adapter_key,
)
from medcompress.models.tokens import (
    fingerprint_vocabulary,
    single_token_ids,
    yes_no_token_ids,
)


class _WordTokenizer:
    """Tokenizer stub: a space-free string is one token if it is in the vocabulary."""

    def __init__(self, vocab: dict[str, int]) -> None:
        self.vocab = vocab

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [self.vocab[text]] if text in self.vocab else [0, 0]


def test_single_token_ids_skips_multi_token_spellings() -> None:
    vocab = {"Yes": 5, "yes": 6, "No": 7}
    ids = single_token_ids(
        lambda s: [vocab[s]] if s in vocab else [1, 2], ["Yes", "yes", "YES", "No"]
    )
    assert ids == (5, 6, 7)
    assert single_token_ids(lambda s: [5], ["a", "b"]) == (5,)  # duplicates collapse


def test_yes_no_token_ids() -> None:
    tok = _WordTokenizer({"Yes": 3, "yes": 4, "No": 5, "no": 6})
    assert yes_no_token_ids(tok) == ((3, 4), (5, 6))
    with pytest.raises(ValueError, match="single-token"):
        yes_no_token_ids(_WordTokenizer({"Yes": 3}))
    with pytest.raises(ValueError, match="shared"):
        yes_no_token_ids(_WordTokenizer({"Yes": 3, "No": 3}))


def test_fingerprint_vocabulary_is_order_free_and_sensitive() -> None:
    a = fingerprint_vocabulary({"a": 0, "b": 1}, ["<s>", "</s>"])
    assert a == fingerprint_vocabulary({"b": 1, "a": 0}, ["</s>", "<s>"])
    assert a != fingerprint_vocabulary({"a": 0, "b": 2}, ["<s>", "</s>"])
    assert a != fingerprint_vocabulary({"a": 0, "b": 1}, ["<s>"])


MODULES = [
    "model.language_model.layers.0.self_attn.q_proj",
    "model.language_model.layers.1.self_attn.q_proj",
    "model.visual.blocks.0.mlp.gate_proj",
]


def test_remap_adapter_key_maps_old_names_onto_the_current_model() -> None:
    old = "base_model.model.model.layers.1.self_attn.q_proj.lora_A.weight"
    assert remap_adapter_key(old, MODULES) == (
        "base_model.model.model.language_model.layers.1.self_attn.q_proj.lora_A.weight"
    )
    vision = "base_model.model.visual.blocks.0.mlp.gate_proj.lora_B.weight"
    assert remap_adapter_key(vision, MODULES) == (
        "base_model.model.model.visual.blocks.0.mlp.gate_proj.lora_B.weight"
    )
    current = "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_B.weight"
    assert remap_adapter_key(current, MODULES) == current


def test_remap_adapter_key_refuses_to_guess() -> None:
    with pytest.raises(ValueError, match="no module"):
        remap_adapter_key("base_model.model.model.layers.0.self_attn.z_proj.lora_A.weight", MODULES)
    # a layer that does not exist must not be silently mapped onto another layer
    with pytest.raises(ValueError, match="2 modules"):
        remap_adapter_key("base_model.model.model.layers.7.self_attn.q_proj.lora_A.weight", MODULES)
    with pytest.raises(ValueError, match="not a LoRA key"):
        remap_adapter_key("lm_head.weight", MODULES)


def _spec(name: str) -> ModelSpec:
    return ModelSpec.from_config(load_config(config_root() / "models" / name))


def test_spec_from_decided_teacher_config() -> None:
    teacher = _spec("qwen2_5_vl_7b_teacher.yaml")
    assert teacher.role == "teacher" and teacher.family == "qwen2_5_vl"
    assert teacher.source == "Qwen/Qwen2.5-VL-7B-Instruct"
    assert teacher.revision and teacher.adapter_revision is None  # our adapter (RC-TT1)
    assert teacher.dtype == "float16" and not teacher.is_candidate
    assert teacher.resolved_revision == teacher.revision
    published = _spec("qwen2_5_vl_7b_teacher_simulamet.yaml")  # the appendix item
    assert published.revision == teacher.revision and published.adapter_revision
    check_loadable(published)  # decided and pinned


def test_candidate_student_needs_explicit_permission() -> None:
    student = _spec("qwen2_5_vl_3b_student.yaml")
    # pinned 2026-10-16 to the revision downloaded on Kaggle, still a candidate until the freeze
    assert student.is_candidate and student.revision == student.sha_seen
    assert student.resolved_revision == "66285546d2b821cf421d4f5eb2576359d3770cd3"
    fallback = _spec("qwen3_vl_2b_student.yaml")
    assert fallback.revision is None and fallback.resolved_revision == fallback.sha_seen
    assert student.dtype == "float16"  # TODO in the config falls back to the T4 default
    with pytest.raises(ModelConfigError, match="candidate"):
        check_loadable(student)
    check_loadable(student, allow_candidate=True)


def test_unpinned_and_unknown_models_are_refused() -> None:
    teacher = _spec("qwen2_5_vl_7b_teacher.yaml")
    unpinned = ModelSpec(**{**teacher.__dict__, "revision": None, "sha_seen": None})
    with pytest.raises(ModelConfigError, match="pinned"):
        check_loadable(unpinned)
    with pytest.raises(ModelConfigError, match="no value for"):
        ModelSpec.from_config(
            {"model": {"name": "x", "role": "student", "family": "TODO", "source": "TODO"}}
        )
    with pytest.raises(ModelConfigError, match="role"):
        ModelSpec.from_config({"name": "x", "role": "judge", "family": "f", "source": "s"})


def test_remote_code_families_need_opt_in() -> None:
    ovis = _spec("ovis2_5_2b_student.yaml")
    assert ovis.needs_remote_code
    with pytest.raises(ModelConfigError, match="remote_code"):
        load_model(ovis, allow_candidate=True)


def test_every_student_config_parses() -> None:
    names = sorted(p.name for p in (config_root() / "models").glob("*_student.yaml"))
    assert len(names) >= 10
    for name in names:
        spec = _spec(name)
        assert spec.role == "student" and spec.source and spec.family
