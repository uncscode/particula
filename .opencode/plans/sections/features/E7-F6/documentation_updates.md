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
- [x] Validated links and snippets with `mkdocs build --strict`; the executable
  selection fence is CPU-only and resident pseudocode is explicitly Warp-gated.

## Deferred Documentation

- Broader transport, detailed RNG/restart policy, orchestration, and
  direct-kernel follow-up documentation remain downstream work. They do not
  change the shipped E7-F6 selection, availability, or explicit-fallback
  boundary.
- E7-F1 records were not changed: no demonstrated stale E7-F1 public-name or
  path assertion was identified. Historical dependency ordering is retained.
