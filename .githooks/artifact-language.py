#!/usr/bin/env python3
"""Validate English-only commit and GitHub artifact boundaries for Issue #22."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


# managed-by: claude-code-config (.githooks/install.sh installs this copy)
MAX_INPUT_BYTES = 2 * 1024 * 1024
JAPANESE = re.compile(
    r"[\u3000-\u303f\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff\uff00-\uffef]"
)
CONVENTIONAL_SUBJECT = re.compile(
    r"[a-z][a-z0-9-]*(?:\([^()\s]+\))?!?: [^\r\n]+"
)
PULL_REQUEST_ACTIONS = {"opened", "edited", "reopened", "synchronize"}
ISSUE_ACTIONS = {"opened", "edited", "reopened"}
COMMENT_ACTIONS = {"created", "edited"}
FULL_SHA = re.compile(r"[0-9a-fA-F]{40}")


class PolicyError(RuntimeError):
    """One selected artifact boundary is malformed or violates policy."""


def require_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PolicyError(f"{label} must be a JSON object")
    return value


def require_string(value: Any, label: str, *, optional: bool = False) -> str:
    if optional and value is None:
        return ""
    if not isinstance(value, str):
        raise PolicyError(f"{label} must be a string")
    return value


def read_utf8(path: pathlib.Path, label: str) -> str:
    try:
        if not path.is_file():
            raise PolicyError(f"{label} is missing or not a regular file")
        if path.stat().st_size > MAX_INPUT_BYTES:
            raise PolicyError(f"{label} exceeds {MAX_INPUT_BYTES} bytes")
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise PolicyError(f"{label} is not UTF-8") from exc
    except OSError as exc:
        raise PolicyError(f"cannot read {label}: {exc}") from exc


def validate_english(label: str, text: str) -> None:
    match = JAPANESE.search(text)
    if match:
        raise PolicyError(
            f"{label} contains a disallowed non-English character "
            f"U+{ord(match.group(0)):04X}"
        )


def validate_commit_message(message: str, label: str = "commit message") -> None:
    # The hook receives a proposed message while GitHub supplies the persisted
    # message. Explicit `-m '# ...'` paragraphs are persisted even though they
    # look like editor-template comments, so both language and subject checks
    # must cover the raw message rather than a guessed cleanup post-image.
    validate_english(label, message)
    payload = message.strip()
    if not payload:
        raise PolicyError(f"{label} is empty")
    subject = payload.splitlines()[0]
    if subject.startswith("Merge ") or subject.startswith('Revert "'):
        return
    if CONVENTIONAL_SUBJECT.fullmatch(subject) is None:
        raise PolicyError(
            f"{label} subject is not Conventional Commits form: "
            "<type>(<optional-scope>): <subject>"
        )


def validate_field(container: dict[str, Any], key: str, label: str,
                   *, optional: bool = False) -> None:
    if key not in container:
        raise PolicyError(f"{label} is missing")
    text = require_string(container.get(key), label, optional=optional)
    validate_english(label, text)


def commit_messages(value: Any, label: str) -> list[str]:
    if not isinstance(value, list):
        raise PolicyError(f"{label} must be a JSON array")
    messages: list[str] = []
    for index, row in enumerate(value):
        commit = require_object(row, f"{label}[{index}]").get("commit")
        commit_object = require_object(commit, f"{label}[{index}].commit")
        messages.append(
            require_string(
                commit_object.get("message"), f"{label}[{index}].commit.message"
            )
        )
    return messages


def fetch_pull_request_commits(event: dict[str, Any]) -> list[dict[str, Any]]:
    repository = require_object(event.get("repository"), "repository")
    full_name = require_string(repository.get("full_name"), "repository.full_name")
    pull_request = require_object(event.get("pull_request"), "pull_request")
    number = pull_request.get("number")
    if isinstance(number, bool) or not isinstance(number, int) or number < 1:
        raise PolicyError("pull_request.number must be a positive integer")
    token = os.environ.get("GITHUB_TOKEN", "")
    api = os.environ.get("GITHUB_API_URL", "https://api.github.com").rstrip("/")
    if not token:
        raise PolicyError("GITHUB_TOKEN is required to inspect pull request commits")
    rows: list[dict[str, Any]] = []
    for page in range(1, 101):
        encoded_repo = urllib.parse.quote(full_name, safe="/")
        url = (
            f"{api}/repos/{encoded_repo}/pulls/{number}/commits"
            f"?per_page=100&page={page}"
        )
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {token}",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                raw = response.read(MAX_INPUT_BYTES + 1)
        except (OSError, urllib.error.URLError) as exc:
            raise PolicyError(f"cannot fetch pull request commits: {exc}") from exc
        if len(raw) > MAX_INPUT_BYTES:
            raise PolicyError("pull request commit response exceeds the input cap")
        try:
            page_rows = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PolicyError("pull request commit response is not valid UTF-8 JSON") from exc
        if not isinstance(page_rows, list):
            raise PolicyError("pull request commit response must be a JSON array")
        rows.extend(page_rows)
        if len(page_rows) < 100:
            return rows
    raise PolicyError("pull request commit pagination exceeds 100 pages")


def fetch_push_commits(event: dict[str, Any]) -> list[dict[str, Any]]:
    """Fetch the complete pushed range from the webhook's before/after SHAs."""
    repository = require_object(event.get("repository"), "repository")
    full_name = require_string(repository.get("full_name"), "repository.full_name")
    before = require_string(event.get("before"), "before")
    after = require_string(event.get("after"), "after")
    for label, value in (("before", before), ("after", after)):
        if FULL_SHA.fullmatch(value) is None:
            raise PolicyError(f"{label} must be a 40-character commit SHA")
    if event.get("created") is not False or event.get("deleted") is not False:
        raise PolicyError("default-branch creation or deletion requires manual review")
    token = os.environ.get("GITHUB_TOKEN", "")
    api = os.environ.get("GITHUB_API_URL", "https://api.github.com").rstrip("/")
    if not token:
        raise PolicyError("GITHUB_TOKEN is required to inspect pushed commits")
    encoded_repo = urllib.parse.quote(full_name, safe="/")
    encoded_before = urllib.parse.quote(before, safe="")
    encoded_after = urllib.parse.quote(after, safe="")
    rows: list[dict[str, Any]] = []
    expected_total: int | None = None
    seen: set[str] = set()
    for page in range(1, 101):
        url = (
            f"{api}/repos/{encoded_repo}/compare/{encoded_before}...{encoded_after}"
            f"?per_page=100&page={page}"
        )
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {token}",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                raw = response.read(MAX_INPUT_BYTES + 1)
        except (OSError, urllib.error.URLError) as exc:
            raise PolicyError(f"cannot fetch pushed commits: {exc}") from exc
        if len(raw) > MAX_INPUT_BYTES:
            raise PolicyError("push comparison response exceeds the input cap")
        try:
            comparison = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PolicyError("push comparison response is not valid UTF-8 JSON") from exc
        comparison_object = require_object(comparison, "push comparison response")
        total = comparison_object.get("total_commits")
        if isinstance(total, bool) or not isinstance(total, int) or total < 1:
            raise PolicyError(
                "push comparison total_commits must be a positive integer"
            )
        if expected_total is None:
            expected_total = total
        elif total != expected_total:
            raise PolicyError("push comparison total changed during pagination")
        page_rows = comparison_object.get("commits")
        if not isinstance(page_rows, list):
            raise PolicyError("push comparison commits must be a JSON array")
        if not page_rows:
            raise PolicyError(
                f"push comparison is incomplete: total={expected_total}, rows={len(rows)}"
            )
        for index, row in enumerate(page_rows):
            commit = require_object(row, f"push comparison commits[{index}]")
            sha = require_string(
                commit.get("sha"), f"push comparison commits[{index}].sha"
            )
            if FULL_SHA.fullmatch(sha) is None:
                raise PolicyError(
                    f"push comparison commits[{index}].sha must be a 40-character SHA"
                )
            if sha in seen:
                raise PolicyError(f"push comparison repeats commit {sha}")
            seen.add(sha)
            rows.append(commit)
        if len(rows) == expected_total:
            final_sha = require_string(
                rows[-1].get("sha"), "push comparison final commit.sha"
            )
            if final_sha.lower() != after.lower():
                raise PolicyError(
                    "push comparison does not terminate at the webhook after SHA"
                )
            return rows
        if len(rows) > expected_total:
            raise PolicyError(
                f"push comparison exceeds total: total={expected_total}, rows={len(rows)}"
            )
    raise PolicyError("push comparison pagination exceeds 100 pages")


def validate_event_value(
    event_name: str,
    event: dict[str, Any],
    *,
    pull_request_commits: Any | None = None,
    push_commits: Any | None = None,
) -> bool:
    """Validate one selected event; return False only for a non-default push."""
    action = event.get("action")
    if event_name == "pull_request":
        if action not in PULL_REQUEST_ACTIONS:
            raise PolicyError(f"unsupported pull_request action: {action!r}")
        pull_request = require_object(event.get("pull_request"), "pull_request")
        validate_field(pull_request, "title", "pull_request.title")
        validate_field(pull_request, "body", "pull_request.body", optional=True)
        rows = (
            fetch_pull_request_commits(event)
            if pull_request_commits is None else pull_request_commits
        )
        for index, message in enumerate(commit_messages(rows, "pull_request.commits")):
            validate_commit_message(message, f"pull_request.commits[{index}].message")
        return True
    if event_name == "issues":
        if action not in ISSUE_ACTIONS:
            raise PolicyError(f"unsupported issues action: {action!r}")
        issue = require_object(event.get("issue"), "issue")
        validate_field(issue, "title", "issue.title")
        validate_field(issue, "body", "issue.body", optional=True)
        return True
    if event_name == "issue_comment":
        if action not in COMMENT_ACTIONS:
            raise PolicyError(f"unsupported issue_comment action: {action!r}")
        comment = require_object(event.get("comment"), "comment")
        validate_field(comment, "body", "comment.body")
        return True
    if event_name == "push":
        repository = require_object(event.get("repository"), "repository")
        default_branch = require_string(
            repository.get("default_branch"), "repository.default_branch"
        )
        ref = require_string(event.get("ref"), "ref")
        if ref != f"refs/heads/{default_branch}":
            return False
        rows = fetch_push_commits(event) if push_commits is None else push_commits
        for index, message in enumerate(commit_messages(rows, "push.commits")):
            validate_commit_message(message, f"push.commits[{index}].message")
        return True
    raise PolicyError(f"unsupported GitHub event: {event_name!r}")


def self_test() -> None:
    validate_commit_message("fix(test): valid English message\n")
    validate_commit_message("Merge branch 'feature/example'\n")
    validate_commit_message('Revert "fix(test): example"\n\nThis reverts commit abc.\n')

    def rejected(callable_object: Any, contains: str) -> None:
        try:
            callable_object()
        except PolicyError as exc:
            if contains not in str(exc):
                raise AssertionError(f"{contains!r} absent from {str(exc)!r}") from exc
        else:
            raise AssertionError(f"expected PolicyError containing {contains!r}")

    rejected(lambda: validate_commit_message("fix(test): 日本語\n"), "commit message")
    rejected(lambda: validate_commit_message("not conventional\n"), "Conventional")
    pr = {
        "action": "opened",
        "pull_request": {"number": 1, "title": "English PR", "body": "English body"},
        "repository": {"full_name": "o/r"},
    }
    commits = [{"commit": {"message": "fix(test): English commit"}}]
    validate_event_value("pull_request", pr, pull_request_commits=commits)
    for field in ("title", "body"):
        mutated = json.loads(json.dumps(pr))
        mutated["pull_request"][field] = "日本語"
        rejected(
            lambda value=mutated: validate_event_value(
                "pull_request", value, pull_request_commits=commits
            ),
            f"pull_request.{field}",
        )
    rejected(
        lambda: validate_event_value(
            "pull_request", pr,
            pull_request_commits=[{"commit": {"message": "fix(test): 日本語"}}],
        ),
        "pull_request.commits[0].message",
    )
    issue = {"action": "opened", "issue": {"title": "English", "body": None}}
    validate_event_value("issues", issue)
    issue["issue"]["body"] = "日本語"
    rejected(lambda: validate_event_value("issues", issue), "issue.body")
    comment = {"action": "created", "comment": {"body": "English"}}
    validate_event_value("issue_comment", comment)
    comment["comment"]["body"] = "日本語"
    rejected(lambda: validate_event_value("issue_comment", comment), "comment.body")
    push = {
        "ref": "refs/heads/main",
        "repository": {"default_branch": "main"},
    }
    push_rows = [{"commit": {"message": "fix(test): English"}}]
    validate_event_value("push", push, push_commits=push_rows)
    push_rows[0]["commit"]["message"] = "fix(test): 日本語"
    rejected(
        lambda: validate_event_value("push", push, push_commits=push_rows),
        "push.commits[0].message",
    )
    push["ref"] = "refs/heads/feature/example"
    if validate_event_value("push", push, push_commits=push_rows):
        raise AssertionError("a non-default push must be non-applicable")
    print("artifact-language: self-test passed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    commit_parser = subparsers.add_parser("commit-message")
    commit_parser.add_argument("--file", type=pathlib.Path, required=True)
    event_parser = subparsers.add_parser("github-event")
    event_parser.add_argument("--event-name", required=True)
    event_parser.add_argument("--event-path", type=pathlib.Path, required=True)
    event_parser.add_argument("--pr-commits-file", type=pathlib.Path)
    event_parser.add_argument("--push-commits-file", type=pathlib.Path)
    subparsers.add_parser("self-test")
    args = parser.parse_args()
    try:
        if args.command == "commit-message":
            validate_commit_message(read_utf8(args.file, "commit message file"))
            print("artifact-language: commit message accepted")
        elif args.command == "github-event":
            try:
                event = json.loads(read_utf8(args.event_path, "GitHub event payload"))
            except json.JSONDecodeError as exc:
                raise PolicyError(f"GitHub event payload is invalid JSON: {exc}") from exc
            event_object = require_object(event, "GitHub event payload")
            pr_fixture = None
            if args.pr_commits_file:
                try:
                    pr_fixture = json.loads(
                        read_utf8(args.pr_commits_file, "pull request commits fixture")
                    )
                except json.JSONDecodeError as exc:
                    raise PolicyError(
                        f"pull request commits fixture is invalid JSON: {exc}"
                    ) from exc
            push_fixture = None
            if args.push_commits_file:
                try:
                    push_fixture = json.loads(
                        read_utf8(args.push_commits_file, "push commits fixture")
                    )
                except json.JSONDecodeError as exc:
                    raise PolicyError(
                        f"push commits fixture is invalid JSON: {exc}"
                    ) from exc
            applicable = validate_event_value(
                args.event_name,
                event_object,
                pull_request_commits=pr_fixture,
                push_commits=push_fixture,
            )
            print(
                "artifact-language: accepted"
                if applicable else "artifact-language: not-applicable"
            )
        else:
            self_test()
        return 0
    except PolicyError as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
