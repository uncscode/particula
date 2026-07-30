# Documentation Updates

## Implemented Documentation

Issue #1500 intentionally made no user-documentation changes. The P1 taxonomy
is direct-import-only and has no package or top-level export; public imports,
availability behavior, fallback examples, and stability documentation remain
P4--P6 work.

Issue #1502 added an execution-boundary subsection to
`docs/Features/data-containers-and-gpu-foundations.md`. It documents the
direct `particula.execution.fallback` import, default-deny policy, five allowed
availability/support reasons, CPU-authoritative `PRE_UPLOAD` and caller-asserted
`RESTORED` boundaries, separate provenance metadata, unchanged native metadata,
and the absence of implicit fallback, movement, lifecycle operations, retry, or
rollback.

- Create or update `docs/Features/backend_selection.md` with the public imports,
  exception taxonomy, availability decision table, and fallback examples.
- Update `docs/Features/Roadmap/data-oriented-gpu.md` Track T6 status and retain
  the no-silent-movement guardrail.
- Further update `docs/Features/data-containers-and-gpu-foundations.md` to
  distinguish stable high-level execution APIs from experimental low-level GPU
  APIs when P4 exports are settled.
- Add public fallback examples only after P4 establishes supported import paths;
  do not imply resident/direct-GPU integration.
- Document that kernel/runtime failures propagate and never trigger retry.
- Document approved imports and concrete-module-only internals in API reference
  material and `AGENTS.md` if its quick-reference contract changes.
- Update E7-F1 and E7 epic plan cross-references if final names differ, then mark
  E7-F6 phases/status as shipped through normal plan-update workflow.
- Validate links and snippets with `mkdocs build --strict`; examples must remain
  runnable in CPU-only environments unless explicitly Warp-gated.
