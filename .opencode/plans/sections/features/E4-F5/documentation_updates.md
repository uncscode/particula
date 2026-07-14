# Documentation Updates

- Issue #1302 updated `condensation_step_gpu()` and scratch-validation
  docstrings with the binary `(n_boxes, n_species)` partitioning contract,
  P1-only sidecar validation, atomic `ValueError` behavior, and the unchanged
  particle-only/no-gas-mutation contract.
- Do not update user-facing roadmap text yet: inventory limiting, in-place gas
  mutation, finalized coupled transfer, and conservation remain P2--P4 work.
- P5 remains responsible for recording scratch ownership, no-hidden-transfer
  rules, Warp CPU support, and optional CUDA behavior in the GPU feature guide.
- Preserve issue #1272 wording: do not claim gas-coupled production support
  until the production hook and same-change conservation regression pass.
- Cross-reference E4-F3/E4-F4 and note that broader production claims remain
  gated by E4-F6/E4-F7.
