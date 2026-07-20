# Success Metrics

- [x] All nine feature tracks E5-F1 through E5-F9 meet their issue #1320 done
  signals and their dependency order is respected.
- [x] Existing Brownian-only calls, caller-owned buffers, and persistent RNG
  repeated-call behavior remain backward compatible.
- [x] Approved charged models meet declared deterministic CPU/Warp pair
  tolerances and remain finite for documented edge cases.
- [x] Charged-only and Brownian-plus-charged aggregate collision rates satisfy
  bounded expectations using one sampling pass.
- [x] SP2016 efficiency-1 sedimentation pair rates match the selected CPU
  reference; equal settling velocities produce zero rate.
- [x] ST1956 turbulent-shear pair rates match the selected CPU reference for
  scalar and per-box validated inputs.
- [x] Supported two-way and four-way combined rates equal the declared sum of
  component CPU rates within stated tolerances.
- [x] Every mechanism conserves species mass; all charge-bearing merges
  conserve charge in separately reported assertions.
- [x] Validation covers multi-box arrays, inactive slots, sparse/degenerate
  populations, fixed buffers, invalid shapes/values, and no mutation or RNG
  advancement on preflight errors.
- [x] Warp CPU tests pass whenever Warp is installed; optional CUDA evidence
  runs when available and skips cleanly otherwise.
- [x] A direct GPU coagulation example runs on Warp CPU with explicit transfers
  and no unsupported high-level integration claims.
- [x] The independent CPU/Warp condensation walkthrough reports physics,
  conservation, and energy criteria separately and assigns every deferred
  capability to a downstream owner.
- [x] Both roadmap files record E5 and E5-F1-F9 IDs, completed scope, artifact
  links, and the next epic's status; all links resolve before E5 closes.
- [x] Test coverage thresholds remain unchanged and changed code maintains at
  least 80% coverage with tests committed alongside implementation.
