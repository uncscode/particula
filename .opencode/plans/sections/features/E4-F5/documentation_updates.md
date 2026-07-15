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
- User-facing roadmap text remains deferred to P5; P4 production/integration
  evidence is still required before broader support claims.
- P5 remains responsible for recording scratch ownership, no-hidden-transfer
  rules, Warp CPU support, and optional CUDA behavior in the GPU feature guide.
- Preserve issue #1272 wording: do not claim gas-coupled production support
  until the production hook and same-change conservation regression pass.
- Cross-reference E4-F3/E4-F4 and note that broader production claims remain
  gated by E4-F6/E4-F7.
