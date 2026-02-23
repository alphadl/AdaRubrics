"""Tests for configuration management."""

from __future__ import annotations

import json

from adarubric.config import AdaRubricConfig


class TestConfigFromFile:
    def test_from_json(self, tmp_path):
        config_data = {
            "llm": {"model": "gpt-4o-mini", "provider": "openai"},
            "generator": {"num_dimensions": 3},
            "filter": {"strategy": "percentile", "percentile": 80.0},
        }
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config_data))

        config = AdaRubricConfig.from_json(path)
        assert config.llm.model == "gpt-4o-mini"
        assert config.generator.num_dimensions == 3
        assert config.filter.strategy == "percentile"

    def test_to_json_roundtrip(self, tmp_path):
        original = AdaRubricConfig()
        original.llm.model = "test-model"

        path = tmp_path / "out.json"
        original.to_json(path)

        loaded = AdaRubricConfig.from_json(path)
        assert loaded.llm.model == "test-model"

    def test_defaults(self):
        config = AdaRubricConfig()
        assert config.llm.provider == "openai"
        assert config.llm.model == "gpt-4o"
        assert config.evaluator.aggregation_strategy == "weighted_mean"
        assert config.filter.min_score == 3.0

    def test_env_override(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        config_data = {"llm": {"model": "gpt-4o"}}
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config_data))

        config = AdaRubricConfig.from_json(path)
        assert config.llm.api_key == "sk-test-key"
