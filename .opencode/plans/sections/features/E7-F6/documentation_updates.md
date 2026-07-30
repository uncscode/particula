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

- [x] Added `docs/Features/backend_selection.md` with stable values, typed
  reason outcomes, resolver order, a selection-only CPU fence, and guarded
  resident-boundary pseudocode.
- [x] Updated the roadmap and foundation guide with the shipped Track T6 status,
  stable-versus-experimental distinction, and no-silent-movement guardrail.
- [x] Documented that kernel/runtime/adapter failures propagate without retry;
  the guide does not imply resident/direct-GPU integration.
- Document approved imports and concrete-module-only internals in API reference
  material and `AGENTS.md` if its quick-reference contract changes.
- Update E7-F1 and E7 epic plan cross-references if final names differ, then mark
  E7-F6 phases/status as shipped through normal plan-update workflow.
- [x] Validated links and snippets with `mkdocs build --strict`; the executable
  selection fence is CPU-only and resident pseudocode is explicitly Warp-gated.
