# Open Questions

- [ ] Which exact two-way masks are part of the first executable support matrix?
  - Proposed decision: register every pairwise combination whose two owning
    mechanism tracks have shipped, retain Brownian-plus-charged, and verify each
    row table-wise. Confirm against the implementation issue before P1 closes.
- [ ] Are three-way masks deliberately unsupported in E5-F6 or should the
  capability matrix include all non-empty subsets?
  - Proposed decision: fail closed unless issue #1320 explicitly requires them;
    E5's done signal requires supported two-way and full four-way evidence, not
    an implicit three-way claim. Document the final choice.
- [ ] Should an unused turbulent dissipation/fluid-density argument be rejected
  for a mask without turbulent shear?
  - Resolve in P1 by applying E5-F1/E5-F5's established excess-input policy
    consistently; do not silently diverge only for combined calls.
- [ ] Is `sum(component_majorants)` too conservative for the bounded trial cap
  in realistic four-way fixtures?
  - Measure in P2/P3. Correctness does not depend on tightness; any optimization
    requires a proof and all-pairs regression evidence and may be deferred.
- [ ] How should device code surface a material `total_rate > total_majorant`
  violation without host synchronization?
  - Reuse E5-F1's device guard/diagnostic contract. A tiny roundoff clamp may be
    allowed, but invalid physics must never become acceptance probability > 1.
