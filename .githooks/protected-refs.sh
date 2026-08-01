#!/usr/bin/env bash
# managed-by: claude-code-config (.githooks/install.sh installs this copy)
# protected-refs.sh — the branches this repository refuses to commit onto or
# push to, and the predicates the hooks in this directory share.
#
# Sourced by `pre-commit`, `pre-merge-commit` and `pre-push`; it is never
# executed on its own, so it carries no exec bit.
#
# This list is the SSOT for the repository-local layer. `hooks/validate-*.sh`
# keep their own copies because they are deployed to a live `~/.claude` that has
# no access to this checkout, so the duplication is structural rather than
# accidental (DC3). `tests/test-githooks-protection.sh` compares the three lists
# and fails when they drift.
PROTECTED_BRANCHES="main master develop production"

# protected_ref <full ref, e.g. refs/heads/master>
# True when the ref names a protected branch. Every caller compares the
# fully-qualified form, so there is deliberately no short-name predicate to reach
# for: Git hands `pre-push` the resolved destination ref — `HEAD`, `@`,
# `HEAD:master`, `+master` and a bare push under `push.default` all arrive as
# `refs/heads/master`, the resolution `hooks/validate-push.sh` fails to perform
# on the command string — and the commit side asks `symbolic-ref` for the same
# form. The comparison is exact, so `feature/master` and `master-2` are ordinary
# branches (DC2).
protected_ref() {
  local candidate="$1" name
  for name in $PROTECTED_BRANCHES; do
    if [ "$candidate" = "refs/heads/$name" ]; then
      return 0
    fi
  done
  return 1
}

# assert_not_on_protected_branch <operation label>
# Refuses the operation when HEAD is on a protected branch. Used by the commit
# and merge hooks, which — unlike `pre-push` — receive no ref on stdin and must
# ask Git where HEAD points.
#
# The question is asked in fully-qualified form. `--short` runs
# `shorten_unambiguous_ref`, which returns `heads/master` — with exit status 0 —
# whenever a second ref named `master` exists (a tag, or `refs/master`), so an
# exact comparison against `master` silently answers "not protected" while the
# status-based fail-loud arm below never fires. Tags arrive from any remote via
# `git fetch --tags`, so one pushed tag would disarm every clone that fetched it.
# `pre-push` never had this hole because Git hands it the full ref; asking for
# the same authoritative form here removes the ambiguous derived one.
assert_not_on_protected_branch() {
  local operation="$1" branch status

  set +e
  branch="$(git symbolic-ref --quiet HEAD)"
  status=$?
  set -e

  case "$status" in
    0) ;;
    # 1 = HEAD is not a symbolic ref (detached). There is no branch name to
    # protect; the destination is still covered by `pre-push`.
    1) return 0 ;;
    # Anything else is Git failing, not Git answering. Refuse rather than read
    # the silence as "not on a protected branch" (DC1).
    *)
      echo "BLOCKED: cannot determine the current branch" \
           "(git symbolic-ref exited $status); refusing the $operation." >&2
      return 1
      ;;
  esac

  if protected_ref "$branch"; then
    echo "BLOCKED: $operation on protected branch '${branch#refs/heads/}'." >&2
    echo "Create a feature branch first: git checkout -b <type>/<name>" >&2
    return 1
  fi
  return 0
}
