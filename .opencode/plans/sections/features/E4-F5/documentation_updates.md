# Documentation Updates

- Update `docs/Features/Roadmap/data-oriented-gpu.md` with the implemented
  partitioning, inventory, four-substep gas-update, and conservation contract.
- Document that gas concentration is mutated in place and returned transfer is
  finalized applied transfer, including units and per-box/species semantics.
- Record scratch ownership, no-hidden-transfer rules, Warp CPU support, and
  optional CUDA behavior in the relevant GPU feature guide.
- Preserve issue #1272 wording: do not claim gas-coupled production support
  until the production hook and same-change conservation regression pass.
- Cross-reference E4-F3/E4-F4 and note that broader production claims remain
  gated by E4-F6/E4-F7.
