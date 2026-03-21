# Contributing to LizyML-Widget

Thank you for your interest in contributing!

## Development Setup

```bash
git clone https://github.com/nbx-liz/LizyML-Widget.git
cd LizyML-Widget
uv sync --frozen --dev
pre-commit install

# JS frontend
cd js && pnpm install && pnpm build && cd ..
```

## Workflow

1. **Branch** from `develop`: `feat/`, `fix/`, `docs/`, `refactor/`
2. **Write tests first** (TDD) — see `skills/testing/SKILL.md`
3. **Run quality gates** before pushing:
   ```bash
   make ci
   ```
4. **Create PR** to `develop` (squash merge)
5. **Conventional Commits**: `<type>(<scope>): <description>`

   Types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `perf`, `ci`

## Quality Gates

All checks must pass before merge:

| Check | Command | Threshold |
|-------|---------|-----------|
| Lint | `uv run ruff check .` | No errors |
| Format | `uv run ruff format --check .` | No diffs |
| Type check | `uv run mypy src/lizyml_widget/` | Strict, no errors |
| Tests | `uv run pytest --cov-fail-under=80` | 80%+ coverage |
| JS build | `cd js && pnpm build` | No errors |
| JS lint | `cd js && pnpm lint` | No errors |

## Specification Changes

Changes to traitlets, BackendAdapter Protocol, common types, data flow, or
external dependencies require a **HISTORY.md Proposal** before implementation.
See `CLAUDE.md §2` for the full change gate list.

Pure UI adjustments, test additions, and documentation fixes do not require a Proposal.

## Documentation Hierarchy

When in doubt, specifications take precedence in this order:

1. `BLUEPRINT.md` — Architecture and design
2. `HISTORY.md` — Change decisions
3. `PLAN.md` — Implementation roadmap
4. `skills/*` — Implementation procedures
5. Source code

## Language Convention

- BLUEPRINT / HISTORY / PLAN / CLAUDE.md: Japanese
- Code comments, docstrings, commits, PRs: English
