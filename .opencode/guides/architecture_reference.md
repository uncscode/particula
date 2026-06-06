# Architecture Reference

**Version:** 2.1.0
**Last Updated:** 2026-06-05

## Overview

This guide serves as the entry point to the adw architecture documentation. It provides references to detailed architectural resources and guidance on when to consult each.

## Research Artifact Storage Boundary

ADforge uses a two-tier storage model for research artifacts and keeps that
model separate from mutable OpenCode sidecar output.

- Small, reviewable research artifacts belong in committed
  `.opencode/plans/sections/research/R{n}/...` paths.
- Large or generated research artifacts belong in the Git-ignored local output
  tree `adforge_local/outputs/research/R{n}/`.
- General OpenCode runtime/generated sidecar output remains under
  `adforge_local/opencode/output/`.

The canonical helper boundaries match that split: use
`get_adforge_local_outputs_dir()` for the shared local outputs root,
`ensure_research_outputs_dir()` for per-research local artifact directories,
and `get_adforge_local_opencode_dir()` for the separate OpenCode sidecar root.

Example:

- Committed, reviewable artifact:
  `.opencode/plans/sections/research/R1/reports/summary.md`
- Local generated artifact:
  `adforge_local/outputs/research/R1/large-run.csv`

This phase documents storage boundaries only and does not reopen helper
implementation work in `adw/utils/paths.py`.

Local research outputs are Git-ignored and must not store secrets,
credentials, or private datasets.

## Shared Structured Summary Boundary

Use the `adforge_core` structured-summary seam when you need compact, deterministic plan or PR summaries without depending on broader workflow internals.

- Schema and fallback authority: `adforge_core/agents/subagents/summarizer.py`
- Thin plan adapter: `adforge_core/tools/plans.py`
- Thin PR adapter: `adforge_core/tools/platform.py`
- Stable public exports: `adforge_core/tools/__init__.py`

Boundary expectations:

- Raw plan/platform summary helpers remain the canonical lookup and aggregation authorities.
- Structured-summary adapters only map bounded allowlisted fields into the summarizer seam.
- Returned data preserves deterministic wrapper envelopes and serialized summary payloads.
- `adforge_voice` consumes this seam through existing read-only tool registration for structured plan and PR summaries rather than introducing a separate summary API surface.

See [Architecture Outline](architecture/architecture_outline.md#shared-structured-summary-adapters) for the module boundary and [Architecture Guide](architecture/architecture_guide.md#adforgecore-structured-summary-hooks-and-native-tool-metadata-boundary) for the design rule.

## `adforge_core.tools` Metadata and Catalog Foundation

`adforge_core.tools` now includes a small internal foundation for native
tool-definition metadata, ordered registration, metadata-only built-in
seeding, deterministic schema export, and import-pure pre-dispatch policy
evaluation, while keeping the existing root helper surface stable.

- Metadata contract module: `adforge_core/tools/models.py`
- Ordered registry module: `adforge_core/tools/catalog.py`
- Metadata-only built-in registry: `adforge_core/tools/builtins.py`
- Read-only git metadata module: `adforge_core/tools/git.py`
- Read-only ADW metadata module: `adforge_core/tools/adw_read.py`
- Schema export helpers: `adforge_core/tools/export.py`
- Native read-only workspace handlers: `adforge_core/tools/files.py`
- Metadata-only mutating platform definitions: `adforge_core/tools/platform.py`
- Native tool policy resolver: `adforge_core/tools/policy.py`
- Workspace-aware catalog composition: `adforge_core/context/resource_loader.py`
- Stable root helper surface: `adforge_core/tools/__init__.py`

Boundary expectations:

- `adforge_core.tools.models` is import-pure and owns the canonical metadata
  contract for tool definitions and argument definitions.
- `adforge_core.tools.catalog` is import-pure and preserves deterministic
  registration order for those definitions.
- `adforge_core.tools.builtins` is import-pure and declares the built-in
  metadata set for the existing helper surface, including deterministic
  `ToolCatalog` seeding behind submodule-only accessors.
- `adforge_core.tools.git` is import-pure and owns the canonical read-only git
  inspection metadata surface (`git_status`, `git_diff`, `git_log`,
  `git_show`), including required-argument metadata such as `git_show.ref`.
- `adforge_core.tools.adw_read` is import-pure and owns the canonical read-only
  ADW workflow-state and plan-read metadata surface (`adw_spec_read`,
  `adw_plans_show`, `adw_plans_list_sections`).
- `adforge_core.tools.platform` combines the existing thin PR-summary adapter
  with metadata-only mutating platform tool definitions whose confirmation and
  policy traits are exported for downstream dispatch layers without adding new
  live execution wiring in this slice.
- `adforge_core.tools.export` is import-pure and serializes ordered catalog
  metadata plus OpenAI-style function schemas without widening the root
  `adforge_core.tools` public API.
- `adforge_core.tools.files` is the Python-native runtime handler layer for
  repository-scoped read-only workspace access; it owns bounded
  `read_workspace_file`, `find_workspace_files`, and
  `search_workspace_content` behavior and reuses `resolve_under_root(...)` for
  confinement.
- `adforge_core.tools.policy` is import-pure and evaluates
  `ToolPolicyRequest` inputs against `ProjectContext` / `PolicyContext` plus
  catalog metadata before any future tool dispatch; it reuses
  `adforge_core.context.confinement` for path-root enforcement and may lazily
  cache the built-in catalog for repeated evaluation.
- `adforge_core.context.resource_loader` is the workspace-aware composition seam
  for effective catalogs. It layers built-in definitions, package entry points
  in `adforge.tools`, and repo-local `.adforge/extensions/` providers in
  deterministic order, with fail-closed validation for malformed payloads,
  duplicate tool names, built-in replacement attempts, and out-of-root local
  sources.
- In this P1 slice, tool identity is intentionally global-by-name; duplicate
  `ToolDefinition.name` values fail closed regardless of namespace metadata.
- `ToolDefinition.namespace` is descriptive metadata in this phase, not part of
  the uniqueness key.
- The existing `adforge_core.tools` root export surface remains intentionally
  stable; callers should import the new foundation from submodules rather than
  assuming new root-level exports.
- The existing `adforge_core.tools` root helper surface also stays
  intentionally narrow after the native workspace handlers land; callers should
  import `adforge_core.tools.files` directly instead of expecting file/search
  helpers to be added to `adforge_core.tools.__all__`.
- Tool-policy consumers should import from `adforge_core.tools.policy`; the
  root `adforge_core.tools` surface remains intentionally unchanged.
- This is a boundary refinement within the already accepted `adforge_core`
  package direction from
  [ADR-021](architecture/decisions/ADR-021-adforge-core-shared-platform-primitives-package.md),
  so no new ADR is required for this slice by itself.

## `adforge_core.hooks` Foundation

`adforge_core.hooks` is the shared hook boundary for typed hook contracts,
deterministic registry behavior, and the first active runtime execution seam
for tool-call hook events.

- Hook models and contracts: `adforge_core/hooks/models.py`
- Registry and ordered loading: `adforge_core/hooks/registry.py`
- Runtime execution seam: `adforge_core/hooks/executor.py`
- Stable namespace entrypoint: `adforge_core/hooks/__init__.py`

Boundary expectations:

- Hook models are the canonical typed contract for downstream hook metadata and
  registration payloads.
- Registry behavior is deterministic and source-aware across built-in,
  installed package, and repo-local hook providers.
- Source ordering is preserved during registration/loading so higher-level
  consumers get stable, repeatable hook resolution.
- The namespace entrypoint remains import-pure and side-effect free, exporting
  only typed contracts plus registry helpers, while the internal executor
  provides deterministic `before_tool_call` and `after_tool_call` execution.
- Hook-driven mutations are in-memory only. Changed mutable
  ``BEFORE_TOOL_CALL`` payloads are revalidated against exported tool schema
  plus tool policy before downstream execution continues, while
  ``AFTER_TOOL_CALL`` remains post-dispatch observability-only.
- Non-observability contract violations remain fail-closed, while
  observability-hook failures are warning-only and do not block the call path.
- Execution consumers should import `adforge_core.hooks.executor` directly;
  the stable `adforge_core.hooks` root entrypoint is intentionally unchanged.

This does not warrant a new ADR by itself because it activates a runtime seam
within the already documented `adforge_core` package direction rather than
changing the package boundary strategy.

## `adforge_core.context` Namespace Boundary

`adforge_core` exposes a deliberately small root surface: `context` and
`tools`. Within that package, the shipped `adforge_core.context` boundary is a
compact, import-pure foundation for downstream consumers. Its public namespace
entrypoint continues to expose seven active modules covering schema contracts in
`models.py`, narrowing-only policy helpers in `policy.py`, workspace
normalization in `workspace.py`, in-root confinement in `confinement.py`,
explicit-input context resolution in `registry.py`, child-context delegation in
`delegation.py`, and redaction-ready audit payload shaping in `audit.py`.
This foundation now also includes an internal Python-native tool resource
loader in `resource_loader.py`.

- Root export surface: `adforge_core/__init__.py`
- Namespace entrypoint: `adforge_core/context/__init__.py`
- Active schema-contract module: `adforge_core/context/models.py`
- Active normalization helper module: `adforge_core/context/workspace.py`
- Active path-confinement helper module: `adforge_core/context/confinement.py`
- Active explicit-input resolver module: `adforge_core/context/registry.py`
- Active policy helper module: `adforge_core/context/policy.py`
- Active delegation helper module: `adforge_core/context/delegation.py`
- Active audit payload module: `adforge_core/context/audit.py`
- Internal tool resource loader module: `adforge_core/context/resource_loader.py`

Boundary expectations:

- `adforge_core.context.models` exports the canonical shared schema
  contracts: `ProjectContext`, `PolicyContext`, `WorkspaceTrust`,
  `LocalInternetAccess`, and `FolderAccessPolicy`.
- `PolicyContext` is the concrete shared authority contract for tool access,
  folder/path-root access, data-label scoping, and local-internet narrowing.
- `adforge_core.context.policy` exports import-pure helpers for a closed
  baseline policy plus fail-closed narrowing merges across tools, paths, data
  labels, and local internet access. Treat this as an active helper layer,
  not a placeholder namespace.
- Canonical policy composition helpers are `build_closed_policy` and
  `merge_policy_context`; both preserve deny-by-default behavior and only
  narrow authority relative to the baseline/current context.
- `adforge_core.context.workspace` exports deterministic, fail-closed
  normalization helpers: `normalize_workspace_id`, `normalize_agent_name`,
  `normalize_workspace_root`, and `normalize_config_path`.
- `adforge_core.context.confinement` exports the import-pure
  `resolve_under_root` helper for canonical in-root resolution under trusted
  roots, rejecting absolute candidates, traversal patterns, and symlinked
  segments.
- `adforge_core.context.registry` exports `resolve_project_context` for
  deterministic `ProjectContext` construction from explicit `workspace_id`,
  `root`, `config_path`, and `agent_name` inputs. The shipped default path is
  explicit-input only: it normalizes the inputs, confines `config_path` under
  the trusted root, and returns internal trust plus a deny-by-default policy
  that allowlists only the canonical workspace root for folder access.
- `adforge_core.context.delegation` exports `derive_child_context` for
  import-pure child-context derivation that preserves parent workspace
  metadata, allows delegated policy/workflow authority to be inherited or
  narrowed only, reuses `merge_policy_context(...)`, and fail-closes on
  workflow-step broadening.
- `adforge_core.context.resource_loader` is an internal, import-pure loading
  seam for Python-native tool resources. It preserves deterministic source
  order (`built-in` -> package entry points in `adforge.tools` -> repo-local
  modules under `.adforge/extensions/`), reuses `ProjectContext` plus
  `resolve_under_root(...)` for repo-local confinement, and fails closed on
  malformed payloads, load/import exceptions, built-in replacement attempts,
  out-of-root or wrong-subtree paths, and duplicate tool names.
- All active modules stay import-pure and side-effect free so downstream
  consumers can depend on them without pulling in `adw` runtime systems.
- `workspace` remains normalization-only, while `confinement` is limited to
  boundary enforcement and canonical in-root resolution; neither layer widens
  into registry loading, policy evaluation, subprocess execution, or
  filesystem mutation.
- The root and namespace export surfaces remain intentionally unchanged for
  this slice; consumers should not infer any new root-level
  `adforge_core.context` re-exports from the internal loader addition.
- The default resolver path does not consult shared registry state.
  `allow_registry=True` is an explicit future seam and currently fails closed
  rather than performing any registry reads.
- `adforge_core.context.audit` exports import-pure payload models for stable
  allow/deny decision serialization of precomputed outcomes, deterministic
  reason fields, and redaction-ready persisted payloads; it does not evaluate
  policy, recompute delegation, resolve registry/workspace state, or write
  runtime audit logs.
- Consumer flow summary: explicit inputs -> normalized `ProjectContext` ->
  narrowing-only `derive_child_context(...)` when delegation is needed ->
  redaction-ready audit payloads.
- Downstream E26 follow-on work should consume or extend this foundation in
  higher-level callers; it is not, by itself, a shipped broad runtime API
  family.
- Smoke tests lock the small root export contract, the active module exports,
  and the import-purity expectations so downstream consumers can safely depend
  on the namespace.

This does not warrant a new ADR by itself because it refines the internal/public package surface within the already accepted `adforge_core` package direction from [ADR-021](architecture/decisions/ADR-021-adforge-core-shared-platform-primitives-package.md) rather than introducing a new architectural strategy.

## Voice Service Boundary (P4)

`adforge_voice` is a dedicated voice-service package with deterministic route
auth behavior and tool-dispatch boundaries:

- Registry/dispatch module: `adforge_voice/tools.py`
- Read handlers: `adforge_voice/handlers/status.py`
- Review handlers: `adforge_voice/handlers/review.py`
- Mutating handlers: `adforge_voice/handlers/intake.py`
- Plan handlers: `adforge_voice/handlers/plans.py`
- Endpoint integration: `adforge_voice/server.py` delegates `/tools/execute`
  dispatch through the registry path.
- Auth primitives: `adforge_voice/auth.py` exposes local TOTP/JWT helpers.
- Auth settings loader: `adforge_voice/config.py` validates
  `VOICE_AUTH_MODE` and mode-specific contracts.

Route auth contract for protected HTTP routes (`/session`, `/tools/execute`):

- `token`: requires `x-adforge-tool-token` matching `VOICE_TOOL_EXECUTE_TOKEN`.
- `totp_jwt`: requires `Authorization: Bearer <session-jwt>` for `/session`
  and treats bearer-authenticated `/tools/execute` calls as read-only.
  Mutating `/tools/execute` requests still require `x-adforge-tool-token`, and
  bearer-only mutating requests fail with `403 forbidden`. Legacy route-token
  auth may still be accepted for read-only compatibility paths.
- `passkey`: passkey config validation is active, but route-level HTTP auth is
  unavailable in MVP and protected routes fail closed with
  `503 configuration_error`.

Structured plan and PR summaries are exposed through the existing read-only registry and `/tools/execute` boundary. Registry-driven schema export and server status mapping remain generic, so these capabilities do not add new routes or voice-specific summary protocols.

Persisted voice-session storage under `adforge_local/state/voice/sessions/` is
bounded by fixed retention policy: expired records (older than 7 days) are
pruned during session writes, malformed session files are removed during
cleanup, and the repository retains at most 100 persisted session files.

## Label Registry Architecture

Workflow label ownership is registry-backed and reconciled by platform-specific sync layers.

- Canonical registry path: `.opencode/workflow/labels.json` (propagated to downstream
  repos via `adw setup pull-opencode`)
- Registry loader entrypoint: `adw/github/labels.py` via `load_label_registry()`
- GitHub reconciliation path: `adw/github/operations.py` (`sync_all_labels`)
- GitLab reconciliation path: `adw/platforms/gitlab.py` (`sync_gitlab_labels`)
- Wrapper filtering behavior: best-effort defense in depth; unknown labels warn and drop
  in wrapper filtering when possible, while router/platform validation remains
  authoritative (canonical enforcement).

`adw setup labels` executes routing-aware sync orchestration, including stale `workflow:*`
label cleanup during reconciliation.

## Auto-Mode Architecture

Auto-mode orchestration lives in `adw/automode/` and coordinates dependency-ordered issue execution via a persisted manifest.
Cron-driven dispatch iterates every manifest in `AUTO_MODE_ACTIVE_STATUSES`, ensuring each
active manifest advances independently rather than only advancing the next runnable entry.

**Package structure:**
- `adw/automode/manifest.py` — `AutoModeManifest` schema + `ManifestStore` persistence.
- `adw/automode/graph.py` — `DependencyGraph` construction, dependency validation, and deterministic ordering.
- `adw/automode/scheduler.py` — scheduler decision engine for auto-mode actions.
- `adw/automode/fix_trigger.py` — CI-failure detection and fix-tracking metadata for in-flight issues.

**ManifestStore API (manifest.py):**
- `load()` reads, validates, and migrates the manifest (returns `None` if missing).
- `save()` writes the manifest using atomic replace (temp file + `os.replace`).
- `lock()` acquires a sibling `.lock` file to serialize readers/writers.
- Error handling: invalid JSON or schema raises `ManifestLoadError`; newer schema versions raise `AutoModeManifestVersionError`.

**DependencyGraph responsibilities (graph.py):**
- Validate `depends_on` references exist in the manifest.
- Detect cycles and surface a readable cycle path.
- Produce a deterministic topological ordering for `execution_order`.

**Scheduler actions (scheduler.py):**
- `DispatchIssue(number)` — start the next runnable issue.
- `WaitForCI(number)` — hold until CI gating completes.
- `AllComplete` — no runnable issues remain.
- `Blocked(reason)` — dependencies or manual pause prevent progress.

**Guardrails & recovery:**
- Auto-mode refuses to run on protected branches (`main`/`master`) and in detached HEAD scenarios.
- Manifest location: `adforge_local/state/auto_mode_manifest.json` (global per repo).
- Recovery step: if the manifest is corrupted, delete `adforge_local/state/auto_mode_manifest.json` and re-run `adw auto-mode init` to rebuild.
- Label gating is enforced only when the platform client exposes `Operation.ISSUE_READ`; if
  issue label reads are unavailable, the scheduler falls back to the first ready issue to
preserve prior automation behavior.

## Symlink-Mode Destination Safety (E23-F3)

`adw setup pull-opencode` and worktree bootstrap support a trusted destination
symlink mode for `.opencode` roots while preserving fail-closed path safety.

Canonical behavior boundaries:

- Destination root may be a symlink **only** when the resolved target is an
  in-repository directory.
- Validation fails closed when any destination ancestor is a symlink.
- Validation fails closed for unresolved/dangling targets.
- Validation fails closed for non-directory targets.
- Validation fails closed when resolved targets escape repository root.

Source-of-truth implementation paths:

- `adw/commands/pull_opencode.py` (`_validate_destination_root`)
- `adw/git/worktree.py` (`ensure_adw_directories`, `_ensure_symlink_mode_opencode_directory`)

Destination naming policy is explicit and fail-closed:

- `--dest` must resolve to a path with basename `.opencode`.
- If `--dest` itself is a symlink, its resolved target must also have basename
  `.opencode`.

Examples:

- Valid: `.opencode`, `./.opencode`, `/repo/.opencode` (in-repo),
  `/repo/.opencode -> /repo/shared/.opencode`.
- Invalid: `tools/opencode`, `/repo/config`,
  `/repo/.opencode -> /repo/shared/opencode`.

Verification surfaces for this behavior:

- `adw/git/tests/worktree_test.py`
- `adw/commands/tests/pull_opencode_test.py`
- `.opencode/tools/__tests__/run_pytest_basic.test.ts`

## Git Tooling Boundary Architecture

Agent-facing git tooling now uses atomic wrapper boundaries in `.opencode/tools/`:

- `git_diff` (read-only inspection)
- `git_stage` (staging)
- `git_commit` (commit-only)
- `git_branch` (branch-pointer operations)
- `git_merge` (merge/rebase lifecycle)
- `git_worktree` (worktree lifecycle)

Shared wrapper-contract helpers (for example, `normalizeSparseOptions`,
`validateCommandMatrix`, `selectDiagnostic`, and `buildErrorEnvelope`) live in
`.opencode/tools/wrapper_contract.ts`. Adoption is incremental: `git_shared.ts`
delegates compatible sparse normalization to the shared helper, while wrappers
such as `git_stage.ts`, `adw_plans.ts`, and `adw_issues_spec.ts` may retain
local validation or diagnostic glue until a behavior-preserving migration is
covered by regression tests.

Safety/policy checks that enforce wrapper-specific intent (for example,
forbidden-argument rules, protected-branch restrictions, and path/ref
guardrails) remain local to each wrapper. Shared-helper migrations must keep
fail-closed validation and deterministic `ERROR:` envelope behavior while
preserving public behavior parity unless an explicit contract change is
approved.

Legacy monolithic `git_operations` wrapper assets were archived under `.trash/.opencode/tools/` and should be treated as historical/deprecated migration context only.

See [ADR-019](architecture/decisions/ADR-019-atomic-git-tool-wrapper-boundaries.md) for the git-wrapper decision record.

## Repository Wrapper Policy Governance

Repository-level wrapper policy extends beyond git tooling and defines how all
wrapper surfaces should be documented and consumed:

- **Canonical active usage:** split/atomic wrappers are the preferred interface
  for new and updated workflows, docs, and agent guidance.
- **Compatibility-window framing:** retained broad wrappers are
  compatibility-only and should be treated as historical/migration context,
  not preferred integration paths.
- **Exception-governance path:** broad-wrapper retention requires
  exception-approved policy metadata and bounded follow-up.

Exception-approved entries must include explicit owner,
compatibility window, and replacement guidance so maintainers can evaluate
retention decisions deterministically.

Repository policy authority and precedent:

- [ADR-020: Repository Wrapper Policy Boundaries](architecture/decisions/ADR-020-repository-wrapper-policy-boundaries.md)
- [ADR-019: Atomic Git Tool Wrapper Boundaries](architecture/decisions/ADR-019-atomic-git-tool-wrapper-boundaries.md)

## Architecture Documentation Structure

The architecture documentation is organized as follows:

```
.opencode/guides/
├── architecture/
│   ├── architecture_guide.md           # Detailed architectural documentation
│   ├── architecture_outline.md         # High-level system overview
│   └── decisions/                      # Architecture Decision Records (ADRs)
│       ├── README.md                   # ADR index and guidelines
│       ├── template.md                 # Template for new ADRs
│       ├── 001-centralized-workflow-state.md      # Individual ADRs
│       ├── 002-error-handling-pattern.md
│       └── ...
└── ../plans/                           # Canonical plan records and section content
    ├── epics/
    ├── features/
    ├── maintenance/
    ├── sections/
    ├── templates/
    └── generated/
```

ADRs live under `.opencode/guides/architecture/decisions/` and link back to the
plan artifacts in `.opencode/plans/` (records and sections) so you
can trace every decision to its roadmap context before writing or updating
architecture guidance.

## Quick Navigation

### For New Contributors

Start here to understand the system:

1. **[Architecture Outline](architecture/architecture_outline.md)**: Quick overview of components and structure
2. **[Architecture Guide](architecture/architecture_guide.md)**: Detailed patterns and principles
3. **[Decision Records](architecture/decisions/README.md)**: Historical context for key decisions

### For Implementing Features

When implementing new features, consult:

1. **[Architecture Guide](architecture/architecture_guide.md)**: Ensure your design follows established patterns
2. **[Architecture Outline](architecture/architecture_outline.md)**: Understand module boundaries
3. **[Code Style Guide](code_style.md)**: Follow coding conventions
4. **Development Plans (`.opencode/plans/`)**: Confirm the feature aligns with
   its epic/maintenance context and update the appropriate plan JSON and
   section files under `.opencode/plans/`

### For Making Architectural Decisions

When making significant architectural decisions:

1. Review **[Architecture Guide](architecture/architecture_guide.md)** for alignment with principles
2. Review **[Decision Records](architecture/decisions/README.md)** for related past decisions
3. Create a new **[ADR](architecture/decisions/template.md)** to document your decision
4. Request architectural review using `/architecture_review`

### For Code Reviews

When reviewing code for architectural concerns:

1. Check alignment with **[Design Principles](architecture/architecture_guide.md#design-principles)**
2. Verify adherence to **[Common Patterns](architecture/architecture_guide.md#common-patterns)**
3. Ensure avoidance of **[Anti-Patterns](architecture/architecture_guide.md#anti-patterns)**
4. Reference **[Review Guide](review_guide.md)** for review criteria

## Primary Documentation

### [Architecture Guide](architecture/architecture_guide.md)

The comprehensive architectural documentation covering:

- **Architectural Principles**: Core design principles guiding the system
- **System Architecture**: High-level structure and component organization
- **Design Patterns**: Standard patterns used throughout the codebase
- **Anti-Patterns**: Approaches to avoid
- **Data Flow**: How data moves through the system
- **Error Handling**: Exception hierarchy and error strategies
- **Testing Architecture**: Test organization and strategies
- **Performance & Security**: Key considerations

**When to Read:**
- Designing new modules or major features
- Understanding system-wide patterns
- Making architectural decisions
- Conducting architecture reviews

### [Architecture Outline](architecture/architecture_outline.md)

A high-level overview providing:

- **System Overview**: What the system does
- **Core Components**: Main building blocks and their responsibilities
- **Module Structure**: Directory organization
- **Technology Stack**: Languages, frameworks, and key dependencies
- **Quick Reference**: Design principles and common patterns
- **Extension Points**: Areas designed for customization

**When to Read:**
- First exploring the codebase
- Understanding component responsibilities
- Finding where to add new features
- Getting oriented quickly

### [Architecture Decision Records (ADRs)](architecture/decisions/README.md)

Historical record of significant architectural decisions:

- **Context**: Why the decision was needed
- **Decision**: What was chosen
- **Alternatives**: What else was considered
- **Consequences**: Trade-offs and outcomes

**When to Read:**
- Understanding why things work the way they do
- Reconsidering past decisions in new contexts
- Learning from past trade-offs
- Creating similar decisions

**When to Create:**
- Making significant architectural changes
- Choosing technologies or frameworks
- Establishing new patterns
- Changing system boundaries

See [ADR README](architecture/decisions/README.md) for guidelines on creating ADRs.

## Integration with ADW

ADW commands reference these architecture documents to:

- **Understand Structure**: Know where code belongs
- **Follow Patterns**: Use established approaches
- **Respect Boundaries**: Maintain module separation
- **Make Decisions**: Create ADRs for significant changes

### Relevant ADW Commands

- `/architecture_review`: Review code for architectural consistency
- `/feature`: Plan features using architectural patterns
- `/implement`: Implement following architectural guidelines
- `/review`: Check adherence to architecture

## Related Documentation

- **[Code Style Guide](code_style.md)**: Coding conventions and standards
- **[Testing Guide](testing_guide.md)**: Test organization and patterns
- **[Review Guide](review_guide.md)**: Code review criteria including architecture
- **[Documentation Guide](documentation_guide.md)**: How to document architectural changes

## Maintaining Architecture Documentation

### When to Update

Update architecture documentation when:

- **Adding Major Features**: Update guide and outline with new patterns
- **Changing Module Structure**: Update outline with new organization
- **Making Architectural Decisions**: Create ADR, update guide
- **Deprecating Components**: Update guide, create deprecation ADR
- **Introducing New Patterns**: Add to design patterns section

### How to Update

1. **Make Changes**: Update relevant documentation files
2. **Create ADR**: For significant decisions, create an ADR in `decisions/`
3. **Update Index**: Add new ADRs to [decisions/README.md](architecture/decisions/README.md)
4. **Cross-Reference**: Link related documents
5. **Review**: Get architecture review before finalizing

### Review Process

Architecture documentation changes should be reviewed by:
- Technical leads
- Senior engineers familiar with the system
- Anyone who will be affected by the changes

Use `/architecture_review` to request review.

## Questions?

If you're unsure about:
- **Where something belongs**: Check [Architecture Outline](architecture/architecture_outline.md)
- **What pattern to use**: Check [Architecture Guide](architecture/architecture_guide.md)
- **Why something was done**: Check [Decision Records](architecture/decisions/README.md)
- **Whether to create an ADR**: Check [ADR Guidelines](architecture/decisions/README.md)

When in doubt, ask for an architecture review or consult with technical leads.
