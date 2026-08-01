#!/usr/bin/env python3
"""Production-shaped offline probes for the Issue #22 artifact boundaries."""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / ".githooks" / "artifact-language.py"
WRAPPER = ROOT / ".githooks" / "commit-msg"
VERIFIER = ROOT / "scripts" / "verify-artifact-language-bundle.py"

spec = importlib.util.spec_from_file_location("artifact_language", VALIDATOR)
assert spec and spec.loader
artifact_language = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = artifact_language
spec.loader.exec_module(artifact_language)


class ArtifactLanguageTests(unittest.TestCase):
    def assert_blocked(self, callable_object, label: str) -> None:
        with self.assertRaisesRegex(artifact_language.PolicyError, label):
            callable_object()

    @staticmethod
    def push_event() -> dict:
        return {
            "ref": "refs/heads/main",
            "before": "1" * 40,
            "after": "2" * 40,
            "created": False,
            "deleted": False,
            "repository": {"default_branch": "main", "full_name": "o/r"},
        }

    @staticmethod
    def push_row(
        index: int,
        message: str = "fix(test): English",
        *,
        sha: str | None = None,
    ) -> dict:
        return {"sha": sha or f"{index:040x}", "commit": {"message": message}}

    def test_commit_boundary(self) -> None:
        for message in (
            "fix(test): valid English message\n",
            "feat(parser)!: describe a breaking change\n\nBREAKING CHANGE: details\n",
            "Merge branch 'feature/example'\n",
            'Revert "fix(test): example"\n\nThis reverts commit abc.\n',
            "fix(test): message\n# English template comment\n",
            "fix(test): message\n# ------------------------ >8 ------------------------\nEnglish discarded\n",
        ):
            artifact_language.validate_commit_message(message)
        self.assert_blocked(
            lambda: artifact_language.validate_commit_message("fix(test): 日本語\n"),
            "commit message",
        )
        self.assert_blocked(
            lambda: artifact_language.validate_commit_message(
                "fix(test): English\n\n日本語 body\n"
            ),
            "commit message",
        )
        self.assert_blocked(
            lambda: artifact_language.validate_commit_message(
                "fix(test): English\n\n# 日本語 retained body\n"
            ),
            "commit message",
        )
        self.assert_blocked(
            lambda: artifact_language.validate_commit_message(
                "fix(test): English\n"
                "# ------------------------ >8 ------------------------\n"
                "日本語 retained by explicit message input\n"
            ),
            "commit message",
        )
        self.assert_blocked(
            lambda: artifact_language.validate_commit_message(
                "# plain English invalid subject\n\n"
                "fix(test): conventional second paragraph\n"
            ),
            "Conventional",
        )
        self.assert_blocked(
            lambda: artifact_language.validate_commit_message("plain English\n"),
            "Conventional",
        )

    @staticmethod
    def pr_event() -> tuple[dict, list[dict]]:
        return (
            {
                "action": "opened",
                "pull_request": {
                    "number": 7,
                    "title": "English title",
                    "body": "English body",
                },
                "repository": {"full_name": "o/r"},
            },
            [{"commit": {"message": "fix(test): English commit"}}],
        )

    def test_pull_request_matrix(self) -> None:
        event, commits = self.pr_event()
        for action in ("opened", "edited", "reopened", "synchronize"):
            event["action"] = action
            artifact_language.validate_event_value(
                "pull_request", event, pull_request_commits=commits
            )
            for field in ("title", "body"):
                mutated = json.loads(json.dumps(event))
                mutated["pull_request"][field] = "日本語"
                self.assert_blocked(
                    lambda value=mutated: artifact_language.validate_event_value(
                        "pull_request", value, pull_request_commits=commits
                    ),
                    f"pull_request.{field}",
                )
            self.assert_blocked(
                lambda: artifact_language.validate_event_value(
                    "pull_request",
                    event,
                    pull_request_commits=[
                        {
                            "commit": {
                                "message": (
                                    "fix(test): English\n\n# 日本語 retained body"
                                )
                            }
                        }
                    ],
                ),
                "pull_request.commits",
            )
            self.assert_blocked(
                lambda: artifact_language.validate_event_value(
                    "pull_request",
                    event,
                    pull_request_commits=[
                        {
                            "commit": {
                                "message": (
                                    "# plain English invalid subject\n\n"
                                    "fix(test): conventional second paragraph"
                                )
                            }
                        }
                    ],
                ),
                "Conventional",
            )
        event["action"] = "closed"
        self.assert_blocked(
            lambda: artifact_language.validate_event_value(
                "pull_request", event, pull_request_commits=commits
            ),
            "unsupported",
        )

    def test_issue_and_comment_matrix(self) -> None:
        for action in ("opened", "edited", "reopened"):
            artifact_language.validate_event_value(
                "issues",
                {"action": action, "issue": {"title": "English", "body": None}},
            )
            for field in ("title", "body"):
                issue = {
                    "action": action,
                    "issue": {"title": "English", "body": "English"},
                }
                issue["issue"][field] = "日本語"
                self.assert_blocked(
                    lambda value=issue: artifact_language.validate_event_value(
                        "issues", value
                    ),
                    f"issue.{field}",
                )
        for action in ("created", "edited"):
            artifact_language.validate_event_value(
                "issue_comment",
                {"action": action, "comment": {"body": "English"}},
            )
            self.assert_blocked(
                lambda value=action: artifact_language.validate_event_value(
                    "issue_comment",
                    {"action": value, "comment": {"body": "日本語"}},
                ),
                "comment.body",
            )

    def test_malformed_fields_fail_closed(self) -> None:
        malformed = (
            ("pull_request", {"action": "opened", "pull_request": []}),
            (
                "pull_request",
                {
                    "action": "opened",
                    "pull_request": {"number": 1, "title": "English"},
                },
            ),
            ("issues", {"action": "opened", "issue": {"title": 7, "body": None}}),
            ("issues", {"action": "opened", "issue": {"title": "English"}}),
            (
                "issue_comment",
                {"action": "created", "comment": {"body": ["English"]}},
            ),
            (
                "push",
                {
                    "ref": "refs/heads/main",
                    "created": True,
                    "deleted": False,
                    "repository": {
                        "default_branch": "main",
                        "full_name": "o/r",
                    },
                    "before": "0" * 40,
                    "after": "1" * 40,
                },
            ),
        )
        for event_name, event in malformed:
            with self.subTest(event_name=event_name):
                self.assertRaises(
                    artifact_language.PolicyError,
                    artifact_language.validate_event_value,
                    event_name,
                    event,
                )

    def test_default_push_and_non_artifact_scope(self) -> None:
        push = self.push_event()
        push["irrelevant_source_text"] = "日本語 outside the artifact fields"
        rows = [
            self.push_row(1),
            self.push_row(
                2,
                "Merge branch 'feature/example'",
                sha=push["after"],
            ),
        ]
        self.assertNotIn("size", push)
        self.assertTrue(
            artifact_language.validate_event_value("push", push, push_commits=rows)
        )
        rows[1]["commit"]["message"] = "fix(test): 日本語"
        self.assert_blocked(
            lambda: artifact_language.validate_event_value(
                "push", push, push_commits=rows
            ),
            r"push.commits\[1\].message",
        )
        rows[1]["commit"]["message"] = (
            "fix(test): English\n\n# 日本語 retained body"
        )
        self.assert_blocked(
            lambda: artifact_language.validate_event_value(
                "push", push, push_commits=rows
            ),
            r"push.commits\[1\].message",
        )
        rows[1]["commit"]["message"] = (
            "# plain English invalid subject\n\n"
            "fix(test): conventional second paragraph"
        )
        self.assert_blocked(
            lambda: artifact_language.validate_event_value(
                "push", push, push_commits=rows
            ),
            "Conventional",
        )
        rows[1]["commit"]["message"] = "Merge branch 'feature/example'"
        push["ref"] = "refs/heads/feature/example"
        self.assertFalse(
            artifact_language.validate_event_value("push", push, push_commits=rows)
        )

    def test_default_push_fetches_complete_compare_range(self) -> None:
        first_page = [self.push_row(index) for index in range(1, 101)]
        second_page = [self.push_row(101, sha=self.push_event()["after"])]

        class Response:
            def __init__(self, payload: dict) -> None:
                self.payload = json.dumps(payload).encode()

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self, _limit: int) -> bytes:
                return self.payload

        responses = [
            Response({"total_commits": 101, "commits": first_page}),
            Response({"total_commits": 101, "commits": second_page}),
        ]
        with mock.patch.dict(
            os.environ, {"GITHUB_TOKEN": "test-token"}
        ), mock.patch.object(
            artifact_language.urllib.request,
            "urlopen",
            side_effect=responses,
        ) as opened:
            rows = artifact_language.fetch_push_commits(self.push_event())
        self.assertEqual(len(rows), 101)
        comparison = (
            "https://api.github.com/repos/o/r/compare/"
            f"{self.push_event()['before']}...{self.push_event()['after']}"
        )
        self.assertEqual(
            [call.args[0].full_url for call in opened.call_args_list],
            [
                f"{comparison}?per_page=100&page=1",
                f"{comparison}?per_page=100&page=2",
            ],
        )

        incomplete = [
            Response({"total_commits": 101, "commits": first_page}),
            Response({"total_commits": 101, "commits": []}),
        ]
        with mock.patch.dict(
            os.environ, {"GITHUB_TOKEN": "test-token"}
        ), mock.patch.object(
            artifact_language.urllib.request,
            "urlopen",
            side_effect=incomplete,
        ):
            self.assert_blocked(
                lambda: artifact_language.fetch_push_commits(self.push_event()),
                "incomplete",
            )

        wrong_endpoint = [
            Response(
                {
                    "total_commits": 1,
                    "commits": [self.push_row(1, sha="3" * 40)],
                }
            )
        ]
        with mock.patch.dict(
            os.environ, {"GITHUB_TOKEN": "test-token"}
        ), mock.patch.object(
            artifact_language.urllib.request,
            "urlopen",
            side_effect=wrong_endpoint,
        ):
            self.assert_blocked(
                lambda: artifact_language.fetch_push_commits(self.push_event()),
                "does not terminate",
            )

        duplicate = [
            Response({"total_commits": 101, "commits": first_page}),
            Response({"total_commits": 101, "commits": [first_page[-1]]}),
        ]
        with mock.patch.dict(
            os.environ, {"GITHUB_TOKEN": "test-token"}
        ), mock.patch.object(
            artifact_language.urllib.request,
            "urlopen",
            side_effect=duplicate,
        ):
            self.assert_blocked(
                lambda: artifact_language.fetch_push_commits(self.push_event()),
                "repeats commit",
            )

        changed_total = [
            Response({"total_commits": 101, "commits": first_page}),
            Response({"total_commits": 102, "commits": second_page}),
        ]
        with mock.patch.dict(
            os.environ, {"GITHUB_TOKEN": "test-token"}
        ), mock.patch.object(
            artifact_language.urllib.request,
            "urlopen",
            side_effect=changed_total,
        ):
            self.assert_blocked(
                lambda: artifact_language.fetch_push_commits(self.push_event()),
                "total changed",
            )

        capped = []
        for page in range(100):
            start = page * 100 + 1
            capped.append(
                Response(
                    {
                        "total_commits": 10001,
                        "commits": [
                            self.push_row(index)
                            for index in range(start, start + 100)
                        ],
                    }
                )
            )
        with mock.patch.dict(
            os.environ, {"GITHUB_TOKEN": "test-token"}
        ), mock.patch.object(
            artifact_language.urllib.request,
            "urlopen",
            side_effect=capped,
        ):
            self.assert_blocked(
                lambda: artifact_language.fetch_push_commits(self.push_event()),
                "pagination exceeds",
            )

    def test_cli_wrapper_and_fixture_event(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            hook_dir = root / "hooks"
            hook_dir.mkdir()
            shutil.copy2(VALIDATOR, hook_dir / VALIDATOR.name)
            shutil.copy2(WRAPPER, hook_dir / WRAPPER.name)
            (hook_dir / WRAPPER.name).chmod(0o755)
            message = root / "message"
            message.write_text("fix(test): English\n", encoding="utf-8")
            accepted = subprocess.run(
                [os.fspath(hook_dir / WRAPPER.name), os.fspath(message)],
                text=True, capture_output=True, check=False,
            )
            self.assertEqual(accepted.returncode, 0, accepted.stderr)
            message.write_text("fix(test): 日本語\n", encoding="utf-8")
            blocked = subprocess.run(
                [os.fspath(hook_dir / WRAPPER.name), os.fspath(message)],
                text=True, capture_output=True, check=False,
            )
            self.assertNotEqual(blocked.returncode, 0)
            self.assertIn("BLOCKED:", blocked.stderr)

            event, commits = self.pr_event()
            event_path = root / "event.json"
            commits_path = root / "commits.json"
            event_path.write_text(json.dumps(event), encoding="utf-8")
            commits_path.write_text(json.dumps(commits), encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable, os.fspath(VALIDATOR), "github-event",
                    "--event-name", "pull_request", "--event-path", os.fspath(event_path),
                    "--pr-commits-file", os.fspath(commits_path),
                ],
                text=True, capture_output=True, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

            push_event_path = root / "push-event.json"
            push_commits_path = root / "push-commits.json"
            push_event_path.write_text(json.dumps(self.push_event()), encoding="utf-8")
            push_commits_path.write_text(
                json.dumps(
                    [self.push_row(1, sha=self.push_event()["after"])]
                ),
                encoding="utf-8",
            )
            push_result = subprocess.run(
                [
                    sys.executable,
                    os.fspath(VALIDATOR),
                    "github-event",
                    "--event-name",
                    "push",
                    "--event-path",
                    os.fspath(push_event_path),
                    "--push-commits-file",
                    os.fspath(push_commits_path),
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(push_result.returncode, 0, push_result.stderr)

            invalid_utf8 = root / "invalid-utf8"
            invalid_utf8.write_bytes(b"fix(test): invalid \xff\n")
            invalid = subprocess.run(
                [
                    sys.executable,
                    os.fspath(VALIDATOR),
                    "commit-message",
                    "--file",
                    os.fspath(invalid_utf8),
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(invalid.returncode, 0)
            self.assertIn("not UTF-8", invalid.stderr)

    def test_installed_hook_controls_real_git_commits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = pathlib.Path(temporary) / "repo"
            target.mkdir()
            subprocess.run(["git", "init", "-q", os.fspath(target)], check=True)
            subprocess.run(
                ["git", "-C", os.fspath(target), "config", "user.name", "Test User"],
                check=True,
            )
            subprocess.run(
                [
                    "git",
                    "-C",
                    os.fspath(target),
                    "config",
                    "user.email",
                    "test@example.com",
                ],
                check=True,
            )
            subprocess.run(
                ["git", "-C", os.fspath(target), "checkout", "-q", "-b", "feature/test"],
                check=True,
            )
            installed = subprocess.run(
                ["bash", os.fspath(ROOT / ".githooks" / "install.sh"), os.fspath(target)],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(installed.returncode, 0, installed.stderr)

            accepted = subprocess.run(
                [
                    "git",
                    "-C",
                    os.fspath(target),
                    "commit",
                    "--allow-empty",
                    "-m",
                    "fix(test): English control",
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(accepted.returncode, 0, accepted.stderr)
            accepted_head = subprocess.run(
                ["git", "-C", os.fspath(target), "rev-parse", "HEAD"],
                text=True,
                capture_output=True,
                check=True,
            ).stdout.strip()

            blocked_messages = (
                (
                    "fix(test): English subject",
                    "# Japanese control \u65e5\u672c\u8a9e retained body",
                ),
                (
                    "# plain English invalid subject",
                    "fix(test): conventional second paragraph",
                ),
            )
            for subject, body in blocked_messages:
                blocked = subprocess.run(
                    [
                        "git",
                        "-C",
                        os.fspath(target),
                        "commit",
                        "--allow-empty",
                        "-m",
                        subject,
                        "-m",
                        body,
                    ],
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertNotEqual(blocked.returncode, 0)
                self.assertIn("BLOCKED:", blocked.stderr)
                blocked_head = subprocess.run(
                    ["git", "-C", os.fspath(target), "rev-parse", "HEAD"],
                    text=True,
                    capture_output=True,
                    check=True,
                ).stdout.strip()
                self.assertEqual(blocked_head, accepted_head)

    def test_framework_preserving_install_keeps_pre_commit_owner(self) -> None:
        marker = (
            "# managed-by: claude-code-config "
            "(.githooks/pre-commit runs through pre-commit)\n"
        )
        generated = """#!/usr/bin/env bash
# File generated by pre-commit: https://pre-commit.com
exit 0
"""
        managed_names = (
            "commit-msg",
            "pre-merge-commit",
            "pre-push",
            "artifact-language.py",
            "protected-refs.sh",
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            fake_modules = root / "modules"
            fake_package = fake_modules / "pre_commit"
            fake_package.mkdir(parents=True)
            (fake_package / "__init__.py").write_text("", encoding="utf-8")
            (fake_package / "clientlib.py").write_text(
                "from __future__ import annotations\n"
                "\n"
                "import pathlib\n"
                "import re\n"
                "\n"
                "def load_config(path: str) -> dict[str, object]:\n"
                "    text = pathlib.Path(path).read_text(encoding='utf-8')\n"
                "    if 'repos: []' in text:\n"
                "        return {'repos': []}\n"
                "    def value(name: str, default: str = '') -> str:\n"
                "        match = re.search(\n"
                "            rf'^[ \\t]+(?:-[ \\t]+)?{name}:[ \\t]*(.+?)[ \\t]*$',\n"
                "            text,\n"
                "            re.MULTILINE,\n"
                "        )\n"
                "        return match.group(1) if match else default\n"
                "    hook = {\n"
                "        'id': value('id'),\n"
                "        'name': value('name'),\n"
                "        'entry': value('entry'),\n"
                "        'language': 'unsupported' if value('language') == 'system' "
                "else value('language'),\n"
                "        'pass_filenames': value('pass_filenames') == 'true',\n"
                "        'always_run': value('always_run') == 'true',\n"
                "        'stages': ['pre-commit'] if value('stages') == '[pre-commit]' else [],\n"
                "        'args': [],\n"
                "        'additional_dependencies': [],\n"
                "    }\n"
                "    return {'repos': [{'repo': 'local', 'hooks': [hook]}]}\n",
                encoding="utf-8",
            )
            fake_bin = root / "bin"
            fake_bin.mkdir()
            fake_pre_commit = fake_bin / "pre-commit"
            fake_pre_commit.write_text(
                "#!/usr/bin/env python3\n"
                "import os\n"
                "import pathlib\n"
                "import subprocess\n"
                "import sys\n"
                f"GENERATED = {generated!r}\n"
                "if sys.argv[1:] == ['--version']:\n"
                "    print('pre-commit 4.5.1-test')\n"
                "    raise SystemExit(0)\n"
                "if len(sys.argv) > 1 and sys.argv[1] == 'install':\n"
                "    hook = pathlib.Path('.git/hooks/pre-commit')\n"
                "    hook.write_text(GENERATED, encoding='utf-8')\n"
                "    hook.chmod(0o755)\n"
                "    raise SystemExit(0)\n"
                "if len(sys.argv) > 2 and sys.argv[1:3] == "
                "['run', 'claude-code-config-protected-commit']:\n"
                "    config = pathlib.Path('.pre-commit-config.yaml').read_text(encoding='utf-8')\n"
                "    required = (\n"
                "        'id: claude-code-config-protected-commit',\n"
                "        'entry: .githooks/pre-commit',\n"
                "        'language: system',\n"
                "        'pass_filenames: false',\n"
                "        'always_run: true',\n"
                "    )\n"
                "    if not all(value in config for value in required):\n"
                "        print('protected local commit (claude-code-config)........Skipped')\n"
                "        raise SystemExit(0)\n"
                "    touch_relative = os.environ.get('FAKE_PRE_COMMIT_TOUCH_RELATIVE')\n"
                "    if touch_relative:\n"
                "        pathlib.Path(touch_relative).write_text('changed\\n', encoding='utf-8')\n"
                "    result = subprocess.run(['.githooks/pre-commit'], check=False)\n"
                "    mutate = os.environ.get('FAKE_PRE_COMMIT_MUTATE_TARGET')\n"
                "    if mutate:\n"
                "        with open(mutate, 'a', encoding='utf-8') as stream:\n"
                "            stream.write('# changed during probe\\n')\n"
                "    raise SystemExit(result.returncode)\n"
                "raise SystemExit(2)\n",
                encoding="utf-8",
            )
            fake_pre_commit.chmod(0o755)

            def fixture(
                name: str,
                *,
                config_marker: bool = True,
                generated_hook: str = generated,
                always_run: bool = True,
                empty_repos: bool = False,
                entry: str = ".githooks/pre-commit",
            ) -> pathlib.Path:
                target = root / name
                target.mkdir()
                subprocess.run(["git", "init", "-q", os.fspath(target)], check=True)
                subprocess.run(
                    ["git", "-C", os.fspath(target), "checkout", "-q", "-b", "feature/test"],
                    check=True,
                )
                config = target / ".pre-commit-config.yaml"
                config_text = marker if config_marker else ""
                if empty_repos:
                    config_text += "repos: []\n"
                else:
                    config_text += (
                        "repos:\n"
                        "  - repo: local\n"
                        "    hooks:\n"
                        "      - id: claude-code-config-protected-commit\n"
                        "        name: protected local commit (claude-code-config)\n"
                        f"        entry: {entry}\n"
                        "        language: system\n"
                        "        stages: [pre-commit]\n"
                        "        pass_filenames: false\n"
                    )
                    if always_run:
                        config_text += "        always_run: true\n"
                config.write_text(config_text, encoding="utf-8")
                source_hook = target / ".githooks" / "pre-commit"
                source_hook.parent.mkdir()
                shutil.copy2(ROOT / ".githooks" / "pre-commit", source_hook)
                shutil.copy2(
                    ROOT / ".githooks" / "protected-refs.sh",
                    target / ".githooks" / "protected-refs.sh",
                )
                if entry != ".githooks/pre-commit":
                    wrong_entry = target / entry.removeprefix("./")
                    wrong_entry.write_text(
                        "#!/usr/bin/env bash\n"
                        "printf 'claude-code-config-pre-commit-reached\\n' "
                        ">\"$CLAUDE_CODE_CONFIG_PRE_COMMIT_PROBE\"\n",
                        encoding="utf-8",
                    )
                    wrong_entry.chmod(0o755)
                subprocess.run(
                    ["git", "-C", os.fspath(target), "add", "."],
                    check=True,
                )
                subprocess.run(
                    [
                        "git",
                        "-C",
                        os.fspath(target),
                        "-c",
                        "user.name=Artifact Probe",
                        "-c",
                        "user.email=artifact@example.invalid",
                        "-c",
                        "core.hooksPath=/dev/null",
                        "commit",
                        "-qm",
                        "test: commit framework fixture",
                    ],
                    check=True,
                )
                hook = target / ".git" / "hooks" / "pre-commit"
                hook.write_text(generated_hook, encoding="utf-8")
                hook.chmod(0o755)
                return target

            def install(
                target: pathlib.Path,
                *,
                extra_env: dict[str, str] | None = None,
            ) -> subprocess.CompletedProcess[str]:
                environment = os.environ.copy()
                environment["PATH"] = f"{fake_bin}{os.pathsep}{environment['PATH']}"
                environment["PYTHONPATH"] = (
                    f"{fake_modules}{os.pathsep}{environment.get('PYTHONPATH', '')}"
                )
                if extra_env:
                    environment.update(extra_env)
                return subprocess.run(
                    [
                        "bash",
                        os.fspath(ROOT / ".githooks" / "install.sh"),
                        "--preserve-pre-commit",
                        os.fspath(target),
                    ],
                    text=True,
                    capture_output=True,
                    check=False,
                    env=environment,
                )

            def assert_native_slots_absent(target: pathlib.Path) -> None:
                for name in managed_names:
                    self.assertFalse((target / ".git" / "hooks" / name).exists(), name)

            target = fixture("accepted")
            framework_hook = target / ".git" / "hooks" / "pre-commit"
            before = framework_hook.read_bytes()
            installed = install(target)
            self.assertEqual(installed.returncode, 0, installed.stderr)
            self.assertEqual(framework_hook.read_bytes(), before)
            self.assertNotEqual(
                framework_hook.read_bytes(), (ROOT / ".githooks" / "pre-commit").read_bytes()
            )
            for name in managed_names:
                self.assertEqual(
                    (target / ".git" / "hooks" / name).read_bytes(),
                    (ROOT / ".githooks" / name).read_bytes(),
                )

            isolated = fixture("isolated-side-effects")
            isolated_user = isolated / "user.txt"
            isolated_user.write_text("committed\n", encoding="utf-8")
            subprocess.run(["git", "-C", os.fspath(isolated), "add", "user.txt"], check=True)
            subprocess.run(
                [
                    "git",
                    "-C",
                    os.fspath(isolated),
                    "-c",
                    "user.name=Artifact Probe",
                    "-c",
                    "user.email=artifact@example.invalid",
                    "-c",
                    "core.hooksPath=/dev/null",
                    "commit",
                    "-qm",
                    "test: add user content",
                ],
                check=True,
            )
            before_user = isolated_user.read_bytes()
            installed = install(
                isolated,
                extra_env={"FAKE_PRE_COMMIT_TOUCH_RELATIVE": "user.txt"},
            )
            self.assertEqual(installed.returncode, 0, installed.stderr)
            self.assertEqual(isolated_user.read_bytes(), before_user)

            missing = fixture("missing-marker", config_marker=False)
            refused = install(missing)
            self.assertNotEqual(refused.returncode, 0)
            self.assertIn("integration marker", refused.stderr)
            assert_native_slots_absent(missing)

            staged_only = fixture("staged-only-marker", config_marker=False)
            staged_config = staged_only / ".pre-commit-config.yaml"
            staged_config.write_text(marker + staged_config.read_text(encoding="utf-8"))
            subprocess.run(
                ["git", "-C", os.fspath(staged_only), "add", ".pre-commit-config.yaml"],
                check=True,
            )
            refused = install(staged_only)
            self.assertNotEqual(refused.returncode, 0)
            self.assertIn("index must be clean", refused.stderr)
            assert_native_slots_absent(staged_only)

            staged_content = fixture("staged-user-content")
            user_file = staged_content / "user.txt"
            user_file.write_text("original\n", encoding="utf-8")
            subprocess.run(["git", "-C", os.fspath(staged_content), "add", "user.txt"], check=True)
            before_user = user_file.read_bytes()
            mutating_hook = staged_content / ".git" / "hooks" / "pre-commit"
            mutating_hook.write_text(
                "#!/usr/bin/env bash\n"
                "# File generated by pre-commit: https://pre-commit.com\n"
                f"printf 'changed\\n' >>{shlex.quote(os.fspath(user_file))}\n"
                "exec .githooks/pre-commit\n",
                encoding="utf-8",
            )
            mutating_hook.chmod(0o755)
            refused = install(staged_content)
            self.assertNotEqual(refused.returncode, 0)
            self.assertIn("index must be clean", refused.stderr)
            self.assertEqual(user_file.read_bytes(), before_user)
            assert_native_slots_absent(staged_content)

            foreign = fixture("foreign-owner")
            foreign_hook = foreign / ".git" / "hooks" / "pre-commit"
            foreign_hook.write_text(
                "#!/usr/bin/env bash\n# another framework\nexit 0\n", encoding="utf-8"
            )
            foreign_hook.chmod(0o755)
            refused = install(foreign)
            self.assertNotEqual(refused.returncode, 0)
            self.assertIn("not owned by the pre-commit framework", refused.stderr)
            assert_native_slots_absent(foreign)

            silent = fixture("missing-runtime-entry", always_run=False)
            silent_hook = silent / ".git" / "hooks" / "pre-commit"
            before_silent = silent_hook.read_bytes()
            refused = install(silent)
            self.assertNotEqual(refused.returncode, 0)
            self.assertIn("always_run must be true", refused.stderr)
            self.assertEqual(silent_hook.read_bytes(), before_silent)
            assert_native_slots_absent(silent)

            spoofed = fixture(
                "spoofed-generated-hook",
                generated_hook=(
                    "#!/usr/bin/env bash\n"
                    "# File generated by pre-commit: https://pre-commit.com\n"
                    "echo 'protected local commit (claude-code-config)........Passed'\n"
                ),
            )
            refused = install(spoofed)
            self.assertNotEqual(refused.returncode, 0)
            self.assertIn("does not match the hook generated", refused.stderr)
            assert_native_slots_absent(spoofed)

            sentinel_fake = fixture(
                "sentinel-writing-foreign-hook",
                empty_repos=True,
                generated_hook=(
                    "#!/usr/bin/env bash\n"
                    "# File generated by pre-commit: https://pre-commit.com\n"
                    "if [ -n \"${CLAUDE_CODE_CONFIG_PRE_COMMIT_PROBE:-}\" ]; then\n"
                    "  printf 'claude-code-config-pre-commit-reached\\n' "
                    ">\"$CLAUDE_CODE_CONFIG_PRE_COMMIT_PROBE\"\n"
                    "fi\n"
                    "exit 0\n"
                ),
            )
            refused = install(sentinel_fake)
            self.assertNotEqual(refused.returncode, 0)
            self.assertIn("expected exactly one", refused.stderr)
            assert_native_slots_absent(sentinel_fake)

            wrong_entry = fixture(
                "wrong-semantic-entry",
                entry="./wrong-entry.sh",
            )
            refused = install(wrong_entry)
            self.assertNotEqual(refused.returncode, 0)
            self.assertIn("entry must be .githooks/pre-commit", refused.stderr)
            assert_native_slots_absent(wrong_entry)

            changing = fixture("self-modifying-framework")
            changing_hook = changing / ".git" / "hooks" / "pre-commit"
            refused = install(
                changing,
                extra_env={
                    "FAKE_PRE_COMMIT_MUTATE_TARGET": os.fspath(changing_hook),
                },
            )
            self.assertNotEqual(refused.returncode, 0)
            self.assertIn("changed during framework-preserving installation", refused.stderr)
            assert_native_slots_absent(changing)

            linked = fixture("symlinked-framework")
            linked_hook = linked / ".git" / "hooks" / "pre-commit"
            linked_target = root / "generated-pre-commit"
            linked_target.write_text(generated, encoding="utf-8")
            linked_target.chmod(0o755)
            linked_hook.unlink()
            linked_hook.symlink_to(linked_target)
            refused = install(linked)
            self.assertNotEqual(refused.returncode, 0)
            self.assertIn("plain executable", refused.stderr)
            assert_native_slots_absent(linked)

    def test_bundle_verifier_accepts_exact_tracked_copy_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = pathlib.Path(temporary) / "repo"
            target.mkdir()
            subprocess.run(["git", "init", "-q", os.fspath(target)], check=True)
            for relative in (
                ".githooks/artifact-language.py",
                ".githooks/commit-msg",
                ".github/workflows/artifact-language.yml",
            ):
                destination = target / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / relative, destination)
            subprocess.run(["git", "-C", os.fspath(target), "add", "."], check=True)
            accepted = subprocess.run(
                [sys.executable, os.fspath(VERIFIER), "--target", os.fspath(target)],
                text=True, capture_output=True, check=False,
            )
            self.assertEqual(accepted.returncode, 0, accepted.stderr)
            (target / ".github/workflows/artifact-language.yml").write_text(
                "name: mutated\n", encoding="utf-8"
            )
            blocked = subprocess.run(
                [sys.executable, os.fspath(VERIFIER), "--target", os.fspath(target)],
                text=True, capture_output=True, check=False,
            )
            self.assertNotEqual(blocked.returncode, 0)
            self.assertIn("target bytes differ", blocked.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
