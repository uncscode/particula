# Documentation Updates

## P1 Status

Issue #1500 intentionally made no user-documentation changes. The P1 taxonomy
is direct-import-only and has no package or top-level export; public imports,
availability behavior, fallback examples, and stability documentation remain
P4--P6 work.

- Create or update `docs/Features/backend_selection.md` with the public imports,
  exception taxonomy, availability decision table, and fallback examples.
- Update `docs/Features/Roadmap/data-oriented-gpu.md` Track T6 status and retain
  the no-silent-movement guardrail.
- Update `docs/Features/data-containers-and-gpu-foundations.md` to distinguish
  stable high-level execution APIs from experimental low-level GPU APIs.
- Add an explicit fallback example showing pre-execution CPU selection and a
  resident-state example requiring checkpoint/finalize before CPU execution.
- Document that kernel/runtime failures propagate and never trigger retry.
- Document approved imports and concrete-module-only internals in API reference
  material and `AGENTS.md` if its quick-reference contract changes.
- Update E7-F1 and E7 epic plan cross-references if final names differ, then mark
  E7-F6 phases/status as shipped through normal plan-update workflow.
- Validate links and snippets with `mkdocs build --strict`; examples must remain
  runnable in CPU-only environments unless explicitly Warp-gated.
