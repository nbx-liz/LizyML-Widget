#!/usr/bin/env bash
# managed-by: claude-code-config (.githooks/install.sh installs this copy)
# install.sh — install this repository's branch-protection hooks into a clone's
# own hooks directory, then prove that Git actually runs them.
#
# Usage: bash .githooks/install.sh [--preserve-pre-commit] [<target repository>]
#
# Why a copy instead of `git config core.hooksPath .githooks`: that setting points
# Git at a directory **inside the worktree**, so the gate is content rather than
# configuration and disappears the moment the worktree does not contain it — a
# `git sparse-checkout` set that excludes it, a checkout of any branch predating
# it, or a detached checkout of an older commit. Git runs no hook and prints no
# diagnostic in that state, so protection vanishes without a word (DC1). The
# hooks directory under `.git` is not touched by checkout, which is the whole
# reason Git puts hooks there.
#
# Why this script lives in `.githooks/` rather than `scripts/`: `scripts` is a
# managed root deployed wholesale to the live `~/.claude` tree. This installer is
# repository tooling with no business being deployed, and #122 pins the exact
# production delta it would otherwise perturb.
#
# **This script accepts one state and refuses every other.** It does not migrate
# settings, merge with a foreign hook, or repair a destination it does not
# recognise. Three adversarial rounds produced escapes exclusively from branches
# that tried to be accommodating: a `--unset` that cleared one scope while
# another still overrode it and reported success; an `rm -rf` that deleted the
# tracked source when `.git/hooks` was a symlink to it; a backup that a second
# run overwrote. The input here is a whole repository's configuration and layout,
# which is an open-ended space — so the way to be safe in it is to keep the set
# of accepted states small and say precisely what to do about the rest.
set -euo pipefail

SOURCE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
SOURCE_ROOT="$(dirname "$SOURCE")"
MODE="native"
if [ "${1:-}" = "--preserve-pre-commit" ]; then
  MODE="framework"
  shift
fi
if [ "$#" -gt 1 ]; then
  echo "BLOCKED: usage: bash .githooks/install.sh [--preserve-pre-commit] [<target repository>]" >&2
  exit 1
fi
TARGET_REPO="${1:-$SOURCE_ROOT}"

HOOK_FILES="commit-msg pre-commit pre-merge-commit pre-push"
INSTALL_HOOK_FILES="$HOOK_FILES"
if [ "$MODE" = "framework" ]; then
  INSTALL_HOOK_FILES="commit-msg pre-merge-commit pre-push"
fi
LIB_FILES="artifact-language.py protected-refs.sh"
MARKER="managed-by: claude-code-config"
FRAMEWORK_CONFIG_MARKER="managed-by: claude-code-config (.githooks/pre-commit runs through pre-commit)"
FRAMEWORK_PROBE_NAME="protected local commit (claude-code-config)"
FRAMEWORK_PROBE_ID="claude-code-config-protected-commit"

die() { echo "BLOCKED: $*" >&2; exit 1; }

for name in $HOOK_FILES $LIB_FILES; do
  [ -f "$SOURCE/$name" ] || die "missing source hook: $SOURCE/$name"
  # The marker is how an installed file is recognised as ours on the next run, so
  # a source file that has lost it installs once and is then refused forever as a
  # stranger's hook — with remediation advice that makes no sense for a file this
  # script wrote. Check the source side too; the boundary has to be symmetric.
  grep -q "$MARKER" "$SOURCE/$name" \
    || die "$SOURCE/$name no longer carries the '$MARKER' marker line; restore it before installing"
done

git -C "$TARGET_REPO" rev-parse --git-dir >/dev/null 2>&1 \
  || die "not a git repository: $TARGET_REPO"

DEST="$(git -C "$TARGET_REPO" rev-parse --path-format=absolute --git-common-dir)/hooks"

# --- core.hooksPath must not point away from the destination -----------------
# It overrides the hooks directory completely, so a value pointing elsewhere
# makes this install inert (DC4). Presence is decided by `git config --get`'s
# exit status, never by the emptiness of its output: `core.hooksPath = ""` is
# *set*, prints nothing, and disables every hook — a state in which a string test
# reports "unset" while no gate runs at all.
#
# A value that resolves to the destination itself is fine and is left alone:
# refusing it would block the reinstall that CONTRIBUTING.md tells everyone to
# run after editing a hook, in a clone where the gate demonstrably works.
#
# The origins are reported by `git config --show-origin`, which names the file
# that actually supplies each value. Probing the four file scopes instead gets
# this wrong in both directions: `--worktree` falls back to the local file when
# `extensions.worktreeConfig` is off (so an ordinary local value is reported as
# living in two scopes, and the `--worktree --unset-all` remedy edits a file the
# operator was not told about), and a value arriving through `include` /
# `includeIf` or `GIT_CONFIG_COUNT` lives in no scope at all.
if git -C "$TARGET_REPO" config --get core.hooksPath >/dev/null 2>&1; then
  effective_now="$(git -C "$TARGET_REPO" rev-parse --path-format=absolute --git-path hooks 2>/dev/null || true)"
  if [ "$effective_now" != "$DEST" ]; then
    echo "BLOCKED: core.hooksPath sends Git to ${effective_now:-an unusable path}, not $DEST, so this install would not run." >&2
    echo "It is set here:" >&2
    git -C "$TARGET_REPO" config --show-origin --get-all core.hooksPath 2>/dev/null |
      while IFS="$(printf '\t')" read -r origin value; do
        case "$value" in
          *[![:space:]]*) ;;
          *) value="(the empty string, which disables every hook)" ;;
        esac
        case "$origin" in
          file:*) printf '  %s = %s\n' "${origin#file:}" "$value" >&2 ;;
          *)      printf '  %s = %s (not a config file)\n' "$origin" "$value" >&2 ;;
        esac
      done
    # The first remedy is local and reversible. Clearing the value is offered
    # second and with a warning, because a global `core.hooksPath` is how husky
    # and the pre-commit framework install themselves: unsetting it here disables
    # the gate in every other repository on the machine, silently.
    echo "Point this clone at its own hooks directory — this leaves every other repository alone:" >&2
    echo "  git -C '$TARGET_REPO' config core.hooksPath '$DEST'" >&2
    echo "then re-run this installer." >&2
    echo "Clearing the entries above works too, but a value in a shared config file is what your" >&2
    echo "other repositories use (husky, pre-commit): git config --file <path> --unset-all core.hooksPath" >&2
    exit 1
  fi
fi

# --- refuse a destination that is not plainly ours --------------------------
# `.git/hooks` symlinked to the worktree's `.githooks` is the pre-`core.hooksPath`
# idiom for exactly this job, and it makes destination and source the same files:
# a copy through it would rewrite the tracked source, and any cleanup would delete
# it.
if [ -L "$DEST" ]; then
  die "$DEST is a symbolic link. Replace it with a real directory (the hooks are copied, not linked), then re-run."
fi
if [ -d "$DEST" ] && [ "$(cd "$DEST" && pwd -P)" = "$SOURCE" ]; then
  die "$DEST resolves to the source directory itself. Nothing to install; remove that arrangement first."
fi

# In framework-preserving mode, `.git/hooks/pre-commit` belongs to pre-commit.
# Validate the complete accepted state before writing any native slot: the
# configuration is tracked and explicitly names our integration, while the
# effective hook is a plain executable generated by pre-commit. The generated
# hook's digest is pinned across the install below so this mode cannot quietly
# replace or rewrite the framework it promises to preserve.
framework_digest=""
if [ "$MODE" = "framework" ]; then
  config="$TARGET_REPO/.pre-commit-config.yaml"
  git -C "$TARGET_REPO" cat-file -e HEAD:.pre-commit-config.yaml 2>/dev/null \
    || die "$config must be tracked in HEAD before framework-preserving installation"
  [ -f "$config" ] && [ ! -L "$config" ] \
    || die "$config must be a plain regular file"
  rc=0
  git -C "$TARGET_REPO" diff --cached --quiet -- || rc=$?
  case "$rc" in
    0) ;;
    1) die "the index must be clean before the pre-commit framework probe" ;;
    *) die "Git could not verify that the index is clean (exit $rc)" ;;
  esac
  rc=0
  git -C "$TARGET_REPO" diff --quiet -- .pre-commit-config.yaml || rc=$?
  case "$rc" in
    0) ;;
    1) die "$config must match its indexed and HEAD version" ;;
    *) die "Git could not verify $config against the index (exit $rc)" ;;
  esac
  grep -Fq "$FRAMEWORK_CONFIG_MARKER" "$config" \
    || die "$config does not carry the reviewed framework integration marker"
  pre_commit_command="$(command -v pre-commit 2>/dev/null)" \
    || die "pre-commit must be on PATH for framework-preserving installation; run this installer through the repository's environment"
  case "$pre_commit_command" in
    /*) ;;
    *) die "pre-commit must resolve to an absolute executable path, not $pre_commit_command" ;;
  esac
  [ -x "$pre_commit_command" ] \
    || die "$pre_commit_command is not an executable pre-commit command"
  pre_commit_version="$("$pre_commit_command" --version 2>/dev/null)" \
    || die "$pre_commit_command could not report its version"
  case "$pre_commit_version" in
    "pre-commit "*) ;;
    *) die "$pre_commit_command did not identify itself as pre-commit" ;;
  esac
  pre_commit_python="$(command -v python3 2>/dev/null)" \
    || die "python3 from the repository environment is required to inspect the pre-commit configuration"
  case "$pre_commit_python" in
    /*) ;;
    *) die "python3 must resolve to an absolute executable path, not $pre_commit_python" ;;
  esac
  semantic_rc=0
  semantic_out="$(
    "$pre_commit_python" - "$config" "$FRAMEWORK_PROBE_ID" 2>&1 <<'PY'
from __future__ import annotations

import sys

from pre_commit.clientlib import load_config


config_path, required_id = sys.argv[1:]
try:
    config = load_config(config_path)
except Exception as error:
    print(f"pre-commit could not load the committed configuration: {error}")
    raise SystemExit(1)

matches = []
for repository in config["repos"]:
    for hook in repository.get("hooks", ()):
        if hook.get("id") == required_id:
            matches.append((repository.get("repo"), hook))

if len(matches) != 1:
    print(f"expected exactly one {required_id!r} hook, found {len(matches)}")
    raise SystemExit(1)

repository, hook = matches[0]
problems = []
if repository != "local":
    problems.append("repo must be local")
if hook.get("name") != "protected local commit (claude-code-config)":
    problems.append("name does not match")
if hook.get("entry") != ".githooks/pre-commit":
    problems.append("entry must be .githooks/pre-commit")
# pre-commit 4.x normalises the legacy `system` spelling to `unsupported`.
if hook.get("language") not in {"system", "unsupported"}:
    problems.append("language must be system")
if hook.get("pass_filenames") is not False:
    problems.append("pass_filenames must be false")
if hook.get("always_run") is not True:
    problems.append("always_run must be true")
if hook.get("stages") != ["pre-commit"]:
    problems.append("stages must contain only pre-commit")
if hook.get("args") != []:
    problems.append("args must be empty")
if hook.get("additional_dependencies") != []:
    problems.append("additional_dependencies must be empty")

if problems:
    print("; ".join(problems))
    raise SystemExit(1)
PY
  )" || semantic_rc=$?
  [ "$semantic_rc" -eq 0 ] \
    || die "the committed pre-commit integration is not the reviewed definition: ${semantic_out:-no diagnostic}"
  framework_hook="$DEST/pre-commit"
  [ -f "$framework_hook" ] && [ ! -L "$framework_hook" ] && [ -x "$framework_hook" ] \
    || die "$framework_hook must be a plain executable pre-commit framework hook"
  grep -Fq "File generated by pre-commit" "$framework_hook" \
    || die "$framework_hook is not owned by the pre-commit framework"
  framework_digest="$(sha256sum <"$framework_hook" | cut -d' ' -f1)"
  framework_probe_dir="$(mktemp -d)"
  framework_probe_repo="$framework_probe_dir/repo"
  framework_probe_file="$framework_probe_dir/reached"
  target_head="$(git -C "$TARGET_REPO" rev-parse HEAD)"
  # Never run the repository's complete pre-commit chain against the operator's
  # worktree. A no-checkout clone plus plumbing commands materialises exactly
  # HEAD without firing checkout hooks. Generate the framework hook again with
  # the command in this environment and require byte parity: a comment-shaped
  # foreign script is not evidence that pre-commit owns the target slot.
  if ! (
    set -e
    export GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_SYSTEM=/dev/null GIT_CONFIG_COUNT=0
    git clone --quiet --no-hardlinks --no-checkout "$TARGET_REPO" "$framework_probe_repo"
    git -C "$framework_probe_repo" update-ref refs/heads/claude-code-config-probe "$target_head"
    git -C "$framework_probe_repo" symbolic-ref HEAD refs/heads/claude-code-config-probe
    git -C "$framework_probe_repo" read-tree "$target_head"
    git -C "$framework_probe_repo" checkout-index -a
    cd "$framework_probe_repo"
    "$pre_commit_command" install --config .pre-commit-config.yaml >/dev/null
  ); then
    rm -rf "$framework_probe_dir"
    die "could not prepare an isolated clone for the pre-commit framework probe"
  fi
  if ! cmp -s "$framework_hook" "$framework_probe_repo/.git/hooks/pre-commit"; then
    rm -rf "$framework_probe_dir"
    die "$framework_hook does not match the hook generated by $pre_commit_version in the repository environment; re-run pre-commit install through that environment"
  fi
  # Export the sentinel only to a targeted `pre-commit run <id>`, never to the
  # generated hook chain. An unrelated hook therefore cannot answer for the
  # reviewed entry, and an absent or skipped id leaves no sentinel.
  framework_rc=0
  framework_out="$(
    cd "$framework_probe_repo" &&
      env GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_SYSTEM=/dev/null GIT_CONFIG_COUNT=0 \
        CLAUDE_CODE_CONFIG_PRE_COMMIT_PROBE="$framework_probe_file" \
        "$pre_commit_command" run "$FRAMEWORK_PROBE_ID" \
          --hook-stage pre-commit 2>&1
  )" || framework_rc=$?
  if [ "$framework_rc" -ne 0 ]; then
    rm -rf "$framework_probe_dir"
    die "the preserved pre-commit framework hook failed its integration probe: ${framework_out:-no output}"
  fi
  if [ ! -f "$framework_probe_file" ] || [ -L "$framework_probe_file" ]; then
    rm -rf "$framework_probe_dir"
    die "the preserved pre-commit framework hook did not execute '$FRAMEWORK_PROBE_NAME'"
  fi
  if [ "$(cat "$framework_probe_file")" != "claude-code-config-pre-commit-reached" ]; then
    rm -rf "$framework_probe_dir"
    die "the preserved pre-commit framework hook published an invalid integration probe"
  fi
  rm -rf "$framework_probe_dir"
fi

mkdir -p "$DEST"

for name in $INSTALL_HOOK_FILES $LIB_FILES; do
  target="$DEST/$name"
  [ -e "$target" ] || [ -L "$target" ] || continue
  # A symlink is never something this script wrote, and writing through one is
  # how the gate silently becomes worktree content again: `cp` follows it, and so
  # does every check that comes after — `-f`, the hash, `-x`, and `git hook run`
  # all report a healthy install while the real file lives in the worktree and
  # disappears on the next checkout. It is also how a copy can overwrite a
  # tracked file. Refuse the shape rather than try to see through it.
  if [ -L "$target" ]; then
    die "$target is a symbolic link. The hooks are copied, not linked — remove it and re-run."
  fi
  if [ "$target" -ef "$SOURCE/$name" ]; then
    die "$target is the same file as $SOURCE/$name. Remove that arrangement and re-run."
  fi
  # A hook that is not ours belongs to someone else — a pre-commit framework,
  # husky, an operator's own script. Replacing it would disable their gate to
  # enable ours, and keeping a backup only moves the problem (a second run
  # overwrites the first backup). Refuse and let a human decide.
  if [ -d "$target" ] || ! grep -q "$MARKER" "$target" 2>/dev/null; then
    die "$target already exists and was not installed by this script. Move it aside (or fold it into .githooks/) and re-run."
  fi
done

# Stage every post-image and preserve every pre-image before the first managed
# slot changes. The EXIT handler restores only paths that still match the bytes
# this transaction installed; if a concurrent writer changes one, its work is
# preserved and the retained transaction directory identifies the incomplete
# recovery instead of silently overwriting it.
transaction="$(mktemp -d "$DEST/.claude-code-config-install.XXXXXX")"
if ! (
  set -e
  mkdir "$transaction/stage" "$transaction/backup" "$transaction/installed"
  for name in $INSTALL_HOOK_FILES; do
    cp "$SOURCE/$name" "$transaction/stage/$name"
    chmod 755 "$transaction/stage/$name"
  done
  for name in $LIB_FILES; do
    cp "$SOURCE/$name" "$transaction/stage/$name"
    chmod 644 "$transaction/stage/$name"
  done
  for name in $INSTALL_HOOK_FILES $LIB_FILES; do
    if [ -e "$DEST/$name" ]; then
      cp -p "$DEST/$name" "$transaction/backup/$name"
    fi
  done
); then
  rm -rf "$transaction"
  die "could not stage the managed hook transaction"
fi

transaction_active=0
probe=""
message_probe=""
on_exit() {  # <original status>
  local original_status="$1" name target tmp rollback_failed=0
  trap - EXIT HUP INT TERM
  set +e
  [ -z "$probe" ] || rm -f "$probe"
  [ -z "$message_probe" ] || rm -f "$message_probe"
  if [ "$transaction_active" -eq 1 ]; then
    for name in $INSTALL_HOOK_FILES $LIB_FILES; do
      [ -e "$transaction/installed/$name" ] || continue
      target="$DEST/$name"
      if [ ! -f "$target" ] || [ -L "$target" ] \
        || ! cmp -s "$transaction/stage/$name" "$target" \
        || [ "$(stat -c '%a' "$transaction/stage/$name" 2>/dev/null)" \
          != "$(stat -c '%a' "$target" 2>/dev/null)" ]; then
        echo "BLOCKED: rollback preserved a concurrently changed path: $target" >&2
        rollback_failed=1
        continue
      fi
      if [ -f "$transaction/backup/$name" ]; then
        tmp="$DEST/.$name.rollback.$$"
        cp -p "$transaction/backup/$name" "$tmp" && mv -f "$tmp" "$target" \
          || rollback_failed=1
      else
        rm -f "$target" || rollback_failed=1
      fi
    done
    if [ "$rollback_failed" -eq 0 ]; then
      rm -rf "$transaction"
    else
      echo "BLOCKED: rollback was incomplete; pre-images remain in $transaction" >&2
    fi
  elif [ -n "$transaction" ]; then
    rm -rf "$transaction"
  fi
  exit "$original_status"
}
trap 'on_exit $?' EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

# Each staged file is copied to a fresh name and then renamed over the target, so
# the write never follows a destination link. Keep the stage copy for exact
# post-image matching if a later verification step needs rollback.
install_one() {  # <name>
  local name="$1" tmp="$DEST/.$1.install.$$"
  cp -p "$transaction/stage/$name" "$tmp"
  mv -f "$tmp" "$DEST/$name"
  : >"$transaction/installed/$name"
}
transaction_active=1
for name in $INSTALL_HOOK_FILES; do install_one "$name"; done
for name in $LIB_FILES;  do install_one "$name"; done

# --- verify the copy --------------------------------------------------------
for name in $INSTALL_HOOK_FILES $LIB_FILES; do
  [ -f "$DEST/$name" ] && [ ! -L "$DEST/$name" ] \
    || die "$DEST/$name is not a plain regular file after the copy"
  if [ "$(sha256sum <"$SOURCE/$name" | cut -d' ' -f1)" \
     != "$(sha256sum <"$DEST/$name" | cut -d' ' -f1)" ]; then
    die "$DEST/$name does not match $SOURCE/$name after the copy"
  fi
done
for name in $INSTALL_HOOK_FILES; do
  [ -x "$DEST/$name" ] || die "$DEST/$name is not executable — Git would ignore it silently"
done
if [ "$MODE" = "framework" ]; then
  [ "$(sha256sum <"$DEST/pre-commit" | cut -d' ' -f1)" = "$framework_digest" ] \
    || die "$DEST/pre-commit changed during framework-preserving installation"
fi

# --- prove Git runs THIS copy ----------------------------------------------
# Two questions, because either alone can be satisfied by the wrong file: where
# Git looks (it must be where we installed — an override pointing elsewhere would
# otherwise let an identical worktree copy answer the probe; re-checked here
# because config can change between the check above and now), and whether the
# hook there actually refuses. `git hook run` resolves the hook the way a real
# operation does; "cannot find a hook" also exits non-zero, which is why the
# refusal message is what is asserted.
effective=""
effective="$(git -C "$TARGET_REPO" rev-parse --path-format=absolute --git-path hooks 2>/dev/null)" || true
[ -n "$effective" ] || die "Git could not resolve a hooks directory in $TARGET_REPO"
if [ "$effective" != "$DEST" ]; then
  die "Git resolves hooks to $effective, not $DEST — the installed copy would never run."
fi

probe="$(mktemp)"
message_probe="$(mktemp)"
printf 'refs/heads/master %040d refs/heads/master %040d\n' 0 0 >"$probe"

probe_how="git hook run"
use_git_hook_run=1
probe_rc=0
probe_out="$(git -C "$TARGET_REPO" hook run --to-stdin="$probe" pre-push 2>&1)" || probe_rc=$?
case "$probe_out" in
  *"is not a git command"* | *"unknown subcommand"*)
    # `git hook run` arrived in git 2.36; 2.34 is still current on Ubuntu 22.04
    # LTS. The hooks themselves work fine there, so refusing the install would
    # deny the gate to the clone over a missing verification verb. Invoke the
    # installed file the way Git would instead — the destination has already been
    # proven to be where Git resolves hooks.
    probe_how="the installed hook, directly (this git has no 'git hook run')"
    use_git_hook_run=0
    probe_rc=0
    probe_out="$(bash "$DEST/pre-push" origin "$TARGET_REPO" <"$probe" 2>&1)" || probe_rc=$?
    ;;
esac

case "$probe_out" in
  *"BLOCKED:"*)
    [ "$probe_rc" -ne 0 ] || die "the installed pre-push printed a refusal but exited 0"
    ;;
  *)
    die "the installed pre-push did not refuse a protected record in $TARGET_REPO (exit $probe_rc: ${probe_out:-no output}). The hooks are on disk but inert."
    ;;
esac

# The message boundary has one positive and one negative control. A non-zero
# status alone is not proof of policy execution (a missing hook also fails), so
# the Japanese probe must carry the validator's structured refusal.
printf 'fix(test): installer English control\n' >"$message_probe"
if [ "$use_git_hook_run" -eq 1 ]; then
  git -C "$TARGET_REPO" hook run commit-msg -- "$message_probe" >/dev/null \
    || die "the installed commit-msg rejected the English control"
else
  bash "$DEST/commit-msg" "$message_probe" >/dev/null \
    || die "the installed commit-msg rejected the English control"
fi
printf 'fix(test): Japanese control \346\227\245\346\234\254\350\252\236\n' >"$message_probe"
message_rc=0
if [ "$use_git_hook_run" -eq 1 ]; then
  message_out="$(git -C "$TARGET_REPO" hook run commit-msg -- "$message_probe" 2>&1)" \
    || message_rc=$?
else
  message_out="$(bash "$DEST/commit-msg" "$message_probe" 2>&1)" || message_rc=$?
fi
case "$message_out" in
  *"BLOCKED:"*) [ "$message_rc" -ne 0 ] \
    || die "the installed commit-msg printed a refusal but exited 0" ;;
  *) die "the installed commit-msg did not reject the Japanese control" ;;
esac

transaction_active=0
rm -rf "$transaction"
transaction=""

if [ "$MODE" = "framework" ]; then
  echo "installed $INSTALL_HOOK_FILES into $DEST, preserved pre-commit, verified with $probe_how"
else
  echo "installed $INSTALL_HOOK_FILES into $DEST, verified with $probe_how"
fi
