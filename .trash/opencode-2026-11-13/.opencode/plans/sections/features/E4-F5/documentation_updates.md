# Documentation Updates

- Issue #1302 updated `condensation_step_gpu()` and scratch-validation
  docstrings with the binary `(n_boxes, n_species)` partitioning contract,
  P1-only sidecar validation, atomic `ValueError` behavior, and the unchanged
  particle-only/no-gas-mutation contract.
- Issue #1303 adds only a short private-helper docstring stating that input is
  already gated.
- Issue #1304 updates implementation docstrings/comments to distinguish raw
  work proposal, P2-finalized applied transfer, gas coupling, finalized total,
  and energy output. They also record aggregate-preflight atomicity and the
  intentional no-rollback boundary for later fresh-proposal failures.
- Issue #1305 adds regression evidence only; it does not change user-facing API
  documentation. The CPU and public-hook conservation gate is now satisfied.
- Issue #1306 completed P5 in
  `docs/Features/Roadmap/data-oriented-gpu.md` and
  `docs/Features/data-containers-and-gpu-foundations.md`. The documents now
  describe authoritative active-device `wp.float64` gas concentration in
  `kg/m^3`, binary per-box/species `wp.int32` partitioning, inventory-limited
  P2-finalized applied transfer, four fixed coupled substeps, and mutable
  caller-owned scratch work/output storage.
- The P5 documents explicitly preserve the CPU↔Warp helper boundary, Warp CPU
  as the installed-Warp evidence path, optional guarded CUDA evidence, and the
  absence of hidden transfers or synchronization.
- Issue #1272's production-hook and same-change conservation-regression gate
  passed in issue #1305; do not expand support claims beyond its test evidence.
- Cross-references distinguish the P1-P4/#1272 direct-kernel and production-hook
  regression evidence from broader E4 production claims, which remain gated by
  E4-F6/E4-F7.
