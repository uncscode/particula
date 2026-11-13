# Retired repository development system — 2026-11-13

Original source revision: `ec58e2546720310dfd1dde9d04524d467695953f`.

The repository no longer uses ADW or the root OpenCode development system.
The entire original `.opencode/` tree is retained here, including guides,
architecture decisions, plans, agents, wrappers and configuration. This is
historical material, not active agent instructions or current API evidence.
Original tracked contents are byte-preserved; `tracked-sha256.json` records
all 1,693 original tracked paths and their hashes. Ignored local dependencies
remain local and are not newly published by the move.

The four retired `.github/ISSUE_TEMPLATE/adw-*.md` templates are also retained
under their original relative paths. The ordinary planning issue template
remains active with retired automation labels/tools removed.

## Maintained replacements

The repository's `AGENTS.md` routes to canonical skills under
`plugins/particula-helper/skills/`: `particula-plan`, `particula-develop`,
`particula-test`, `particula-docs`, and `particula-review`. Their shared
references carry repository conventions, validation, docstrings and native/GPU
scientific boundaries. All host packages contain the same skills; the plugin's
OpenCode adapter is independent of the retired repository workflow.

`guide-inventory.json` accounts for all 87 guide files. Converted entries
retain useful Particula decisions, not obsolete tool calls or mandatory agent
orchestration. Historical architecture records remain unchanged; current
`docs/guide/` contracts supersede obsolete facade and checkpoint-schema prose.

## Conflicts resolved

- Direct pytest/Ruff/mypy commands replace wrapper calls. Direct `-Werror`
  matches source CI; the old wrapper-specific flag ban is retired.
- Full-package coverage keeps the 80% gate. Focused assertions are not evidence
  of full-package coverage. Repeated `--cov` options specify multiple modules.
- Maintained documentation uses Astro/Starlight, percent-cell Python and saved
  results. Notebook pairing, MkDocs commands and stale example counts retire.
- Native 0.3.0 containers and schema-v5 checkpoints replace stale facade/schema
  guidance. Concrete-only boundaries and explicit transfers remain intact.
- Planning uses meaningful dependencies and observable completion criteria,
  without fixed phase counts, old IDs, workflow state, or automatic posting.
- Google-style docstrings describe actual domains/errors; conflicting legacy
  template headings and speculative requirements are not copied.
- Requested plans now use `plans/`; public scientific evidence remains in
  `docs/contribute/validation/`. No old plan is marked completed by this move.
- Qualification command updates supply no new measured evidence. Release,
  scientific and performance approval gates remain distinct.

## Guide disposition

| Original path below `.opencode/guides/` | Maintained destination or disposition |
| --- | --- |
| `README.md` | AGENTS.md routing; plugin README |
| `agent_fallback_system.md` | Retired workflow/tool manual; no active skill dependency |
| `agents/QUICK_REFERENCE.md` | Retired workflow/tool manual; no active skill dependency |
| `agents/README.md` | Retired workflow/tool manual; no active skill dependency |
| `agents/adw-build-family.md` | particula-develop (useful decisions only; automation retired) |
| `agents/adw-default.md` | Retired workflow/tool manual; no active skill dependency |
| `agents/adw-review-system.md` | particula-review (useful decisions only; automation retired) |
| `agents/adw-setup-maintainer.md` | Retired workflow/tool manual; no active skill dependency |
| `agents/adw-setup.md` | Retired workflow/tool manual; no active skill dependency |
| `agents/agent-creator.md` | Retired workflow/tool manual; no active skill dependency |
| `agents/agent-permissions.md` | Retired workflow/tool manual; no active skill dependency |
| `agents/auto-mode-bootstrap.md` | Retired workflow/tool manual; no active skill dependency |
| `agents/dev-plan-manager.md` | particula-plan (portable planning decisions only; orchestration retired) |
| `agents/documentation-system.md` | particula-docs (useful decisions only; automation retired) |
| `agents/epic-to-issues.md` | Retired workflow/tool manual; no active skill dependency |
| `agents/execute-plan.md` | particula-develop (useful decisions only; automation retired) |
| `agents/git-commit.md` | particula-review (useful decisions only; automation retired) |
| `agents/investigate-adw.md` | Retired workflow/tool manual; no active skill dependency |
| `agents/linter.md` | particula-test (useful decisions only; automation retired) |
| `agents/plan-epic-drafter.md` | particula-plan (portable planning decisions only; orchestration retired) |
| `agents/plan-feature-drafter.md` | particula-plan (portable planning decisions only; orchestration retired) |
| `agents/plan-maintenance-drafter.md` | particula-plan (portable planning decisions only; orchestration retired) |
| `agents/plan-orchestrator.md` | particula-plan (portable planning decisions only; orchestration retired) |
| `agents/plan-phase-splitter.md` | particula-plan (portable planning decisions only; orchestration retired) |
| `agents/plan-research-drafter.md` | particula-plan (portable planning decisions only; orchestration retired) |
| `agents/plan-review-architecture.md` | particula-plan (portable planning decisions only; orchestration retired) |
| `agents/plan-type-designer.md` | Retired workflow/tool manual; no active skill dependency |
| `agents/plan-work.md` | particula-plan (portable planning decisions only; orchestration retired) |
| `agents/planner-intake.md` | particula-plan (portable planning decisions only; orchestration retired) |
| `agents/planning-system.md` | particula-plan (portable planning decisions only; orchestration retired) |
| `agents/pr-comment-analyzer.md` | particula-review (useful decisions only; automation retired) |
| `agents/setup-adw.md` | Retired workflow/tool manual; no active skill dependency |
| `agents/setup-platform-dev.md` | Retired workflow/tool manual; no active skill dependency |
| `agents/setup-update.md` | Retired workflow/tool manual; no active skill dependency |
| `agents/shipper.md` | particula-review (useful decisions only; automation retired) |
| `architecture/architecture_guide.md` | scientific-boundaries.md; current docs/guide contracts |
| `architecture/architecture_outline.md` | repository-conventions.md; current architecture docs |
| `architecture/decisions/ADR-001-neutral-gpu-wall-loss-boundary.md` | Historical ADR/reference retained; current native/GPU docs own the contract |
| `architecture/decisions/ADR-002-gpu-fixed-slot-activation-boundary.md` | Historical ADR/reference retained; current native/GPU docs own the contract |
| `architecture/decisions/ADR-003-dependency-neutral-execution-capabilities.md` | Historical ADR/reference retained; current native/GPU docs own the contract |
| `architecture/decisions/ADR-004-concrete-gpu-resident-session-boundary.md` | Historical ADR/reference retained; current native/GPU docs own the contract |
| `architecture/decisions/ADR-005-one-time-gpu-resident-session-setup.md` | Historical ADR/reference retained; current native/GPU docs own the contract |
| `architecture/decisions/ADR-006-resident-gpu-step-lifecycle-guard.md` | Historical ADR/reference retained; current native/GPU docs own the contract |
| `architecture/decisions/ADR-007-resident-session-checkpoint-finalize-restart.md` | Historical ADR/reference retained; current native/GPU docs own the contract |
| `architecture/decisions/ADR-008-resident-session-failure-close-semantics.md` | Historical ADR/reference retained; current native/GPU docs own the contract |
| `architecture/decisions/ADR-009-resident-process-delegation-adapters.md` | Historical ADR/reference retained; current native/GPU docs own the contract |
| `architecture/decisions/ADR-010-resident-state-update-boundary.md` | Historical ADR/reference retained; current native/GPU docs own the contract |
| `architecture/decisions/ADR-011-resident-thermodynamic-freshness-coordinator.md` | Historical ADR/reference retained; current native/GPU docs own the contract |
| `architecture/decisions/ADR-012-resident-complete-loop-and-diagnostics.md` | Historical ADR/reference retained; current native/GPU docs own the contract |
| `architecture/decisions/ADR-013-pre-execution-availability-resolution.md` | Historical ADR/reference retained; current native/GPU docs own the contract |
| `architecture/decisions/ADR-014-opt-in-cpu-fallback-boundary.md` | Historical ADR/reference retained; current native/GPU docs own the contract |
| `architecture/decisions/ADR-015-execution-public-surface-and-experimental-gpu-policy.md` | Historical ADR/reference retained; current native/GPU docs own the contract |
| `architecture/decisions/ADR-016-direct-gpu-volume-evolution-boundary.md` | Historical ADR/reference retained; current native/GPU docs own the contract |
| `architecture/decisions/ADR-017-direct-gpu-particle-transport-boundary.md` | Historical ADR/reference retained; current native/GPU docs own the contract |
| `architecture/decisions/ADR-018-resident-communication-integration.md` | Historical ADR/reference retained; current native/GPU docs own the contract |
| `architecture/decisions/ADR-019-prepared-resident-graph-capture-qualification.md` | Historical ADR/reference retained; current native/GPU docs own the contract |
| `architecture/decisions/ADR-020-authenticated-native-resident-graph-replay.md` | Historical ADR/reference retained; current native/GPU docs own the contract |
| `architecture/decisions/ADR-021-graph-capture-teardown-ownership.md` | Historical ADR/reference retained; current native/GPU docs own the contract |
| `architecture/decisions/README.md` | Historical ADR/reference retained; current native/GPU docs own the contract |
| `architecture/decisions/template.md` | Historical ADR/reference retained; current native/GPU docs own the contract |
| `architecture_reference.md` | scientific-boundaries.md; current docs/guide contracts |
| `backend_configuration.md` | Retired workflow/tool manual; no active skill dependency |
| `code_culture.md` | particula-develop; repository-conventions.md |
| `code_style.md` | particula-develop; repository-conventions.md |
| `commit_conventions.md` | particula-review; repository-conventions.md |
| `conditional_docs.md` | AGENTS.md task-to-skill routing |
| `docstring_class.md` | particula-docs; docstrings.md (actual contracts only) |
| `docstring_function.md` | particula-docs; docstrings.md (Google-style sections) |
| `docstring_guide.md` | particula-docs; docstrings.md |
| `documentation_guide.md` | particula-docs; maintained documentation/authoring guides |
| `index.md` | AGENTS.md routing; plugin README |
| `linting_guide.md` | particula-test; repository-validation.md |
| `notebook_validation_guide.md` | particula-docs; source/result workflow |
| `opencode-advanced-features.md` | Retired workflow/tool manual; no active skill dependency |
| `phase-sizing-rules.md` | particula-plan (coherent steps; retired rigid parser/size rules) |
| `planner_issue_format.md` | particula-plan; ordinary planning issue template |
| `pr_conventions.md` | particula-review; repository-conventions.md |
| `review_guide.md` | particula-review |
| `security/container-isolation.md` | Retired workflow/tool manual; no active skill dependency |
| `setup_guide.md` | Retired workflow/tool manual; no active skill dependency |
| `testing_guide.md` | particula-test; repository-validation.md |
| `tool-wrapper-exceptions.json` | Retired workflow/tool manual; no active skill dependency |
| `troubleshooting_setup.md` | Retired workflow/tool manual; no active skill dependency |
| `workflow-conditionals.md` | Retired workflow/tool manual; no active skill dependency |
| `workflow-engine.md` | Retired workflow/tool manual; no active skill dependency |
| `workflow-examples.md` | Retired workflow/tool manual; no active skill dependency |
| `workflow-json-schema.md` | Retired workflow/tool manual; no active skill dependency |
