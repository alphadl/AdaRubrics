"""Tests for configuration management."""

from __future__ import annotations

import json

from adarubric.config import AdaRubricConfig
from adarubric.pipeline import _default_eval_max_tokens, _default_rubric_max_tokens


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

    def test_to_json_excludes_api_key_by_default(self, tmp_path):
        cfg = AdaRubricConfig()
        cfg.llm.api_key = "sk-secret"
        path = tmp_path / "out.json"
        cfg.to_json(path)
        data = json.loads(path.read_text())
        assert "api_key" not in data.get("llm", {})

    def test_to_json_include_secrets(self, tmp_path):
        cfg = AdaRubricConfig()
        cfg.llm.api_key = "sk-secret"
        path = tmp_path / "secret.json"
        cfg.to_json(path, include_secrets=True)
        data = json.loads(path.read_text())
        assert data["llm"]["api_key"] == "sk-secret"

    def test_defaults(self):
        config = AdaRubricConfig()
        assert config.llm.provider == "openai"
        assert config.llm.model == "gpt-4o"
        assert config.evaluator.aggregation_strategy == "weighted_mean"
        assert config.filter.min_score == 3.0

    def test_default_rubric_max_tokens_resolution(self):
        c = AdaRubricConfig()
        c.llm.max_tokens = 2048
        assert _default_rubric_max_tokens(c) == 2048
        c.generator.max_tokens = 512
        assert _default_rubric_max_tokens(c) == 512

    def test_default_eval_max_tokens_resolution(self):
        c = AdaRubricConfig()
        c.llm.max_tokens = 4096
        assert _default_eval_max_tokens(c) == 8192
        c.evaluator.max_tokens = 12_000
        assert _default_eval_max_tokens(c) == 12_000

    def test_default_token_helpers_without_config(self):
        assert _default_rubric_max_tokens(None) == 4096
        assert _default_eval_max_tokens(None) == 8192

    def test_env_override(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        config_data = {"llm": {"model": "gpt-4o"}}
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config_data))

        config = AdaRubricConfig.from_json(path)
        assert config.llm.api_key == "sk-test-key"
