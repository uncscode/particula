# Open Questions

- [x] What should deterministic byte accounting measure?
  - Resolved 2026-08-30: Report logical manifest bytes computed from checked
    shape products and dtype item sizes, grouped by canonical role and family.
  - Rationale: This is deterministic across devices and shares the registry's
    allocation formulas; allocator pools and future tape remain separate evidence.
  - Evidence:
    - `particula/execution/gpu_resources.py:488` - The registry defines supported
      manifest item sizes.
    - `particula/execution/gpu_resources.py:1229` - Registry acquisition defines
      checked shape and allocation formulas.
  - Resolved by: plan-question-resolver

- [x] Should compatible capture-resource preparation create fresh views?
  - Resolved 2026-08-30: No. Return the exact published outer set, native
    records, arrays, capacities, and report by identity.
  - Rationale: Stable capture bindings require identity-preserving repeated
    acquisition rather than equivalent replacement objects.
  - Evidence:
    - `particula/execution/gpu_resources.py:497` - Registry publication pins
      reusable arrays by role and identity.
    - `particula/execution/tests/gpu_resources_test.py:184` - Existing tests
      assert stable repeated acquisition.
  - Resolved by: plan-question-resolver

- [x] Must every E8-F2 validation/status temporary be registry-owned, or may a
  prepared adapter retain a separately owned fixed array?
  - Resolved 2026-08-30: Every reusable capture-lifetime array must be published
    and identity-pinned by the registry; discarded preparation-only temporaries
    need not be registered.
  - Rationale: This keeps compatibility signatures and byte reports complete
    while still permitting caller allocation before registry publication.
  - Evidence:
    - `particula/execution/gpu_resources.py:497` - Publication may pin caller- or
      registry-allocated arrays and validates identity, not allocator provenance.
    - `.opencode/plans/sections/features/E8-F3/architecture_design.md:5` - The
      registry is the sole capture-lifetime storage authority.
  - Resolved by: plan-question-resolver

- [x] Should capture-selected diagnostic outputs be checkpoint continuation
  payloads?
  - Resolved 2026-08-30: Pin diagnostic outputs for capture identity but exclude
    them from checkpoint continuation unless an output is proven to affect the
    next timestep.
  - Rationale: Capture stability does not make caller-owned observation outputs
    canonical recovery state.
  - Evidence:
    - `particula/execution/checkpoint.py:121` - Arbitrary caller-owned outputs are
      never checkpoint authority.
    - `particula/execution/gpu_resources.py:766` - Diagnostic outputs remain
      caller-owned arrays validated against resident storage.
  - Resolved by: plan-question-resolver
