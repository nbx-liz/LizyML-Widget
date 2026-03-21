#!/usr/bin/env python3
"""Release automation script for LizyML-Widget.

Usage:
    uv run python scripts/release.py v0.4.0

What it does:
    1. Validates version format and semver increment
    2. Extracts release notes from CHANGELOG.md
    3. Commits CHANGELOG (if uncommitted changes exist)
    4. Pushes develop to origin
    5. Creates a PR from develop → main with release title and notes

After the PR is merged on GitHub, the auto-release.yml workflow will:
    - Create a git tag
    - Create a GitHub Release with notes from CHANGELOG.md
    - Trigger release.yml (PyPI publish)
"""

from __future__ import annotations

import re
import subprocess
import sys


def run(cmd: str, *, check: bool = True) -> str:
    """Run a shell command and return stdout."""
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, check=False
    )
    if check and result.returncode != 0:
        print(f"ERROR: {cmd}", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(1)
    return result.stdout.strip()


def get_current_version() -> str:
    """Get the latest git tag version."""
    tags = run("git tag --sort=-v:refname")
    for line in tags.splitlines():
        if re.match(r"^v\d+\.\d+\.\d+$", line):
            return line
    return "v0.0.0"


def validate_version(new: str, current: str) -> str:
    """Validate version format and that it's an increment."""
    if not re.match(r"^v\d+\.\d+\.\d+$", new):
        print(
            f"ERROR: Invalid version format: {new} (expected vX.Y.Z)",
            file=sys.stderr,
        )
        sys.exit(1)

    def parse(v: str) -> tuple[int, ...]:
        return tuple(int(x) for x in v.lstrip("v").split("."))

    new_parts = parse(new)
    cur_parts = parse(current)

    if new_parts <= cur_parts:
        print(
            f"ERROR: New version {new} must be greater than current {current}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Determine type
    if new_parts[0] > cur_parts[0]:
        return "MAJOR"
    if new_parts[1] > cur_parts[1]:
        return "MINOR"
    return "PATCH"


def extract_changelog_section(version: str) -> str:
    """Extract the CHANGELOG.md section for a given version."""
    ver = version.lstrip("v")
    try:
        with open("CHANGELOG.md") as f:
            content = f.read()
    except FileNotFoundError:
        return f"Release {ver}"

    pattern = rf"## \[{re.escape(ver)}\].*?\n(.*?)(?=\n## \[|\Z)"
    match = re.search(pattern, content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return f"Release {ver}"


def check_branch() -> None:
    """Ensure we're on develop."""
    branch = run("git branch --show-current")
    if branch != "develop":
        print(
            f"ERROR: Must be on 'develop' branch, currently on '{branch}'",
            file=sys.stderr,
        )
        sys.exit(1)


def check_changelog_has_version(version: str) -> None:
    """Ensure CHANGELOG.md has an entry for this version."""
    ver = version.lstrip("v")
    with open("CHANGELOG.md") as f:
        content = f.read()
    if f"## [{ver}]" not in content:
        print(
            f"ERROR: CHANGELOG.md has no entry for [{ver}]. "
            f"Add it before running this script.",
            file=sys.stderr,
        )
        sys.exit(1)


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: uv run python scripts/release.py v0.4.0", file=sys.stderr)
        sys.exit(1)

    new_version = sys.argv[1]

    # Validate
    check_branch()
    current = get_current_version()
    bump_type = validate_version(new_version, current)
    check_changelog_has_version(new_version)
    notes = extract_changelog_section(new_version)

    print(f"Current version: {current}")
    print(f"New version:     {new_version} ({bump_type})")
    print(f"Release notes:\n{notes}\n")

    # Commit CHANGELOG if there are uncommitted changes
    status = run("git status --porcelain CHANGELOG.md", check=False)
    if status:
        ver = new_version.lstrip("v")
        msg = f"docs: update CHANGELOG for v{ver} release"
        run(f'git add CHANGELOG.md && git commit -m "{msg}"')
        print("Committed CHANGELOG.md")

    # Push develop
    run("git push origin develop")
    print("Pushed develop to origin")

    # Create PR
    ver = new_version.lstrip("v")
    pr_title = f"release: {new_version} — LizyML-Widget {ver}"

    pr_body = f"""## Summary
{notes}

## Version
- Previous: {current}
- New: {new_version}
- Type: {bump_type}
"""

    # Write body to temp file to avoid shell escaping issues
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(pr_body)
        body_file = f.name

    pr_url = run(
        f"gh pr create --base main --head develop "
        f'--title "{pr_title}" '
        f'--body-file "{body_file}"'
    )
    print(f"\nPR created: {pr_url}")
    print("\nNext: Merge the PR on GitHub.")
    print("Tag + release will be created automatically.")


if __name__ == "__main__":
    main()
