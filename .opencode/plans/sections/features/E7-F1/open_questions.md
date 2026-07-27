# Open Questions

- [x] Where should backend selection live?
  - Resolved 2026-07-26: Use a separate `particula.execution` context layer.
  - Rationale: `RunnableABC` is typed to CPU `Aerosol`, while strategies and
    builders describe physics/configuration. A separate context can remain
    dependency-neutral and later own adapter/session selection.
  - Evidence: `particula/runnable.py:36-218` and issue #1451 Track T1.

- [x] Should an adapter failure trigger another backend automatically?
  - Resolved 2026-07-26: No. Validate first, dispatch once, and propagate failure.
  - Rationale: issue #1451 explicitly excludes silent fallback and hidden
    movement; E7-F6 owns any explicit transition policy.

- [x] Should device requests use a closed enum or preserve a validated native
  device string such as `"cuda:0"`?
  - Resolved 2026-07-27: Keep backend identity closed, but preserve the device as
    an opaque native string validated by the selected adapter.
  - Rationale: Warp already supports indexed and evolving device syntax; the
    dependency-neutral execution layer should not duplicate its parser.
  - Evidence:
    - `particula/gpu/conversion.py:54` - the existing boundary accepts a string
      including `"cuda:0"` and `"cpu"`.
    - `particula/gpu/conversion.py:67` - validation delegates the complete value
      to Warp's native device resolver.
  - Resolved by: plan-question-resolver

- [x] Should adapter registration be public in E7-F1 or remain internal until
  E7-F2/E7-F3 exercise real GPU adapters?
  - Resolved 2026-07-27: Publish the adapter protocol and selection contracts,
    but keep mutable registration and registry storage private in E7-F1.
  - Rationale: No plugin requirement exists, while public mutation would freeze
    replacement, compatibility, and lifecycle rules before real adapters land.
  - Evidence:
    - `.opencode/plans/sections/features/E7-F1/architecture_design.md:48` - the
      accepted architecture keeps concrete adapters and registries module-local.
    - `.opencode/guides/architecture_reference.md:28` - public APIs must be
      exported deliberately.
  - Resolved by: plan-question-resolver
