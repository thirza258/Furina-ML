# Furina-ML Subagents

Specialized agents configured for token-efficient development in this repository.
Definitions live in `.claude/agents/` (a local, gitignored folder); this file is
the committed index.

## Available Subagents

### Repository-wide

| Agent | Role | Model | Scope |
| --- | --- | --- | --- |
| `code-investigator` | Locates relevant code and dependencies without full scans | `haiku` | Read-only symbol and file discovery |
| `security-reviewer` | Audits auth, permissions, API exposure, and OWASP risks | `sonnet` | Changed files, DRF views, queries, configs |
| `test-engineer` | Writes reproduction tests and validates fixes | `sonnet` | Targeted Django/React test execution |

### ML lifecycle platform

These four cover the `lifecycle` app and its frontend. See `docs/lifecycle.md`
for the architecture they work within.

| Agent | Role | Scope |
| --- | --- | --- |
| `lifecycle-backend` | Implements and maintains the lifecycle app | `backend/lifecycle/**`, settings app registration |
| `frontend-integration` | Builds the lifecycle workbench UI | `frontend/src/{interface,services,pages,components}/**` |
| `ml-test-engineer` | Authors and runs lifecycle tests, guards legacy regressions | `backend/lifecycle/tests.py`; regression on `core`/`academy` tests |
| `migration-specialist` | Backfills legacy runs into the lifecycle platform | `backend/lifecycle/management/commands/migrate_legacy_runs.py` |

## Usage Guidelines
- **Avoid Over-Delegation**: Solve trivial edits (1-2 files) directly without launching subagents.
- **Specific Tasks**: Provide subagents with explicit file paths or feature scopes.
- **Concise Summaries**: Subagents report findings directly without dumping raw log streams or entire files.
- **Preserve Contracts**: `/api/datasets/`, `/api/train/`, `/api/runs/` response shapes are
  frozen (asserted by `core/tests.py`, consumed by the shipping frontend). New work goes
  under `/api/lifecycle/`.
- **Verify Before Reporting Done**: `manage.py test lifecycle core academy` and
  `npm run build --prefix frontend` must both pass.
