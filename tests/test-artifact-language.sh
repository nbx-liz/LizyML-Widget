#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
python3 "$ROOT/tests/artifact-language-probe.py"
python3 "$ROOT/.githooks/artifact-language.py" self-test
