"""Tests for JSON substring extraction from LLM outputs."""

from __future__ import annotations

from adarubric.llm.json_extract import extract_json_substring


def test_fenced_json_block():
    raw = 'Here you go:\n```json\n{"a": 1}\n```\nThanks'
    assert extract_json_substring(raw) == '{"a": 1}'


def test_plain_object():
    assert extract_json_substring('prefix {"x": true} suffix') == '{"x": true}'


def test_nested_braces():
    raw = '{"outer": {"inner": 2}}'
    assert extract_json_substring(raw) == raw
