#!/usr/bin/env python3
"""Verify one repository carries the canonical Issue #22 artifact bundle."""

from __future__ import annotations

import argparse
import hashlib
import os
import pathlib
import stat
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parent.parent
FILES = {
    ".githooks/artifact-language.py": 0o755,
    ".githooks/commit-msg": 0o755,
    ".github/workflows/artifact-language.yml": 0o644,
}


def digest(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=pathlib.Path, required=True)
    args = parser.parse_args()
    target = args.target.resolve()
    failures: list[str] = []
    bundle = hashlib.sha256()
    for relative, expected_mode in FILES.items():
        canonical = ROOT / relative
        candidate = target / relative
        if not canonical.is_file() or canonical.is_symlink():
            failures.append(f"canonical file is missing or unsafe: {relative}")
            continue
        if not candidate.is_file() or candidate.is_symlink():
            failures.append(f"target file is missing or unsafe: {relative}")
            continue
        tracked = subprocess.run(
            ["git", "-C", os.fspath(target), "ls-files", "--error-unmatch", "--", relative],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if tracked.returncode != 0:
            failures.append(f"target file is not tracked: {relative}")
        canonical_digest = digest(canonical)
        candidate_digest = digest(candidate)
        if canonical_digest != candidate_digest:
            failures.append(f"target bytes differ: {relative}")
        mode = stat.S_IMODE(candidate.stat().st_mode)
        if mode != expected_mode:
            failures.append(
                f"target mode differs: {relative} is {mode:04o}, expected {expected_mode:04o}"
            )
        bundle.update(relative.encode("utf-8") + b"\0")
        bundle.update(bytes.fromhex(canonical_digest))
    if failures:
        for failure in failures:
            print(f"BLOCKED: {failure}", file=sys.stderr)
        return 1
    print(f"artifact-language bundle sha256={bundle.hexdigest()} target={target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
