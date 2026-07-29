"""Tests for the good/better/best model presets."""

from __future__ import annotations

import pytest

from isaac.llm.presets import (
    PRESETS,
    apply_preset,
    describe_presets,
    get_preset,
    preset_dicts,
    recommend_preset,
)


class TestPresetCatalogue:
    def test_the_documented_rungs_exist(self) -> None:
        assert {"minimal", "small", "good", "better", "best"} <= set(PRESETS)

    @pytest.mark.parametrize("name", list(PRESETS))
    def test_every_preset_is_self_consistent(self, name: str) -> None:
        preset = PRESETS[name]
        assert preset.name == name, "dict key and preset name must agree"
        assert preset.model, "a preset without a model is unusable"
        assert preset.provider
        assert preset.notes, "each rung must explain its trade-off"
        assert preset.test_time_samples >= 1

    @pytest.mark.parametrize("name", list(PRESETS))
    def test_sampling_temperature_is_non_zero(self, name: str) -> None:
        # Self-consistency over greedy samples is wasted compute: every draw
        # would be identical.
        preset = PRESETS[name]
        if preset.test_time_samples > 1:
            assert preset.sampling_temperature > 0

    def test_vram_increases_across_the_local_ladder(self) -> None:
        ladder = [PRESETS[n].vram_gb for n in ("minimal", "small", "good", "better")]
        assert ladder == sorted(ladder), "local rungs must be ordered by memory need"

    def test_only_the_api_rung_has_no_vram_figure(self) -> None:
        assert PRESETS["best"].vram_gb == 0.0
        assert all(PRESETS[n].vram_gb > 0 for n in ("minimal", "small", "good", "better"))

    def test_constrained_decoding_is_on_where_tools_are_unsupported(self) -> None:
        # gemma3:1b reports no tool capability, so the envelope is mandatory.
        assert PRESETS["minimal"].constrained_decoding is True

    def test_larger_models_do_not_pay_for_constraint(self) -> None:
        assert PRESETS["better"].constrained_decoding is False
        assert PRESETS["best"].constrained_decoding is False

    def test_repair_stays_on_everywhere_as_a_safety_net(self) -> None:
        assert all(p.repair_tool_calls for p in PRESETS.values())


class TestGetPreset:
    def test_lookup_is_case_insensitive(self) -> None:
        assert get_preset("GOOD").name == "good"

    def test_surrounding_whitespace_is_tolerated(self) -> None:
        assert get_preset("  best  ").name == "best"

    def test_unknown_name_lists_the_valid_ones(self) -> None:
        with pytest.raises(KeyError) as excinfo:
            get_preset("gigantic")
        assert "good" in str(excinfo.value)


class TestAsEnv:
    def test_renders_core_variables(self) -> None:
        env = PRESETS["good"].as_env()
        assert env["ISAAC_MODEL_NAME"] == "nemotron-3-nano:4b"
        assert env["ISAAC_LLM_PROVIDER"] == "ollama"

    def test_boolean_flags_render_as_zero_or_one(self) -> None:
        assert PRESETS["minimal"].as_env()["ISAAC_CONSTRAINED_DECODING"] == "1"
        assert PRESETS["good"].as_env()["ISAAC_CONSTRAINED_DECODING"] == "0"

    def test_tier_models_are_emitted_only_when_set(self) -> None:
        assert "ISAAC_STRONG_MODEL" in PRESETS["best"].as_env()
        assert "ISAAC_STRONG_MODEL" not in PRESETS["good"].as_env()

    def test_all_values_are_strings(self) -> None:
        # These go straight into os.environ, which rejects non-strings.
        for preset in PRESETS.values():
            assert all(isinstance(v, str) for v in preset.as_env().values())


class TestApplyPreset:
    def test_writes_into_a_supplied_mapping(self) -> None:
        env: dict[str, str] = {}
        apply_preset("good", env=env)
        assert env["ISAAC_MODEL_NAME"] == "nemotron-3-nano:4b"

    def test_does_not_touch_os_environ_when_given_a_mapping(self, monkeypatch) -> None:
        monkeypatch.delenv("ISAAC_MODEL_NAME", raising=False)
        apply_preset("good", env={})
        import os

        assert "ISAAC_MODEL_NAME" not in os.environ

    def test_mutates_os_environ_by_default(self, monkeypatch) -> None:
        monkeypatch.delenv("ISAAC_MODEL_NAME", raising=False)
        apply_preset("better")
        import os

        assert os.environ["ISAAC_MODEL_NAME"] == PRESETS["better"].model


class TestRecommend:
    @pytest.mark.parametrize(
        ("vram", "expected"),
        [(24.0, "better"), (12.0, "better"), (6.0, "good"), (4.0, "small"), (2.0, "minimal")],
    )
    def test_picks_the_rung_that_fits(self, vram: float, expected: str) -> None:
        assert recommend_preset(vram_gb=vram).name == expected

    def test_no_gpu_with_a_key_suggests_the_api_rung(self) -> None:
        assert recommend_preset(vram_gb=0.0, has_api_key=True).name == "best"

    def test_no_gpu_without_a_key_stays_local(self) -> None:
        # ISAAC is local-first: never recommend an API the user cannot use.
        assert recommend_preset(vram_gb=0.0, has_api_key=False).name == "minimal"

    def test_a_capable_gpu_is_not_overridden_by_an_api_key(self) -> None:
        assert recommend_preset(vram_gb=12.0, has_api_key=True).name == "better"


class TestRendering:
    def test_table_lists_every_preset(self) -> None:
        table = describe_presets()
        assert all(name in table for name in PRESETS)

    def test_dicts_are_json_serialisable(self) -> None:
        import json

        json.dumps(preset_dicts())
