# Open Questions

- [x] May fallback occur automatically after a GPU adapter or kernel raises?
  - Resolved 2026-07-26: No. Propagate all failures after invocation begins.
  - Rationale: Issue #1451 prohibits silent CPU/GPU movement, and E7-F1 selects
    from declared capabilities rather than exception-driven probing.
  - Evidence: `docs/Features/Roadmap/data-oriented-gpu.md:1494-1497` and
    `.opencode/plans/sections/features/E7-F1/architecture_design.md:34-37`.

- [x] Where may explicit CPU fallback occur?
  - Resolved 2026-07-26: Before upload/mutation while CPU state is authoritative,
    or after a caller explicitly checkpoints/finalizes and restores CPU state.
  - Rationale: This makes every transfer and synchronization caller-visible.
  - Evidence: `.opencode/plans/sections/epics/E7/implementation_strategy.md:20-31`.

- [x] Should low-level GPU imports emit unconditional experimental warnings?
  - Resolved 2026-07-26: No. Mark stability in module/API documentation and
    release notes so supported `-Werror` workflows remain unchanged.
  - Evidence: repository focused commands routinely use `-Werror`; issue #1451
    requires experimental status but does not require runtime warnings.

- [x] What exact E7-F1 symbol names will E7-F6 extend after T1 implementation lands?
  - Resolved 2026-07-27: Extend `ExecutionRequest`, `ExecutionResult`,
    `ExecutionContext`, and `CapabilityMatrix` in place, and consume
    `CPUExecutionAdapter`; bind any lower-level protocol names to the merged
    E7-F1 implementation rather than creating parallel types.
  - Rationale: These names are fixed by the accepted upstream architecture, and
    E7-F6 owns policy enrichment rather than replacement contracts.
  - Evidence:
    - `.opencode/plans/sections/features/E7-F1/architecture_design.md:11` - the
      accepted design names the request, matrix, context, CPU adapter, and result.
    - `.opencode/plans/sections/features/E7-F1/architecture_design.md:55` - E7-F6
      is explicitly assigned availability and error-policy extension.
  - Resolved by: plan-question-resolver
