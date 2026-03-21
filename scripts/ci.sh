#!/usr/bin/env bash
# Mirror .github/workflows/ci.yml locally (install dev deps first: pip install -e ".[dev]")
set -euo pipefail
cd "$(dirname "$0")/.."
ruff check adarubric/ tests/
ruff format --check adarubric/ tests/
pytest tests/ -v --tb=short --cov=adarubric --cov-report=term-missing
# Same flags as CI type-check job (strict mypy catches ndarray typing on all Pythons)
mypy adarubric/ --ignore-missing-imports
echo "OK: all CI checks passed"
