# Success Criteria

- [ ] E6-F5 remains the mandatory and only slot-classification/activation
  dependency; E6-F6 does not duplicate or weaken its predicates.
- [ ] Resampling defaults to enabled and representative-volume scaling defaults
  to disabled; controls remain independently selectable.
- [ ] When both are enabled and exhaustion occurs, resampling is planned first;
  scaling runs only if planned resampling cannot satisfy all demand.
- [ ] Capacity-sufficient calls bypass exhaustion mutation regardless of policy
  switches; both switches off with exhausted demand raises before every write.
- [ ] Invalid or unsatisfiable calls preserve particle fields, volume, request,
  diagnostics, work buffers, identities, and any caller RNG state.
- [ ] No represented demand is truncated: with scale `s`, represented demand is
  exactly `s * gas_admitted_demand`; otherwise the entire call fails closed.
- [ ] Per-box post represented number, every species mass, and charge equal
  `s * pre_state + represented_demand` at recorded float64 tolerances. Existing
  and source intensive concentrations are unchanged by representation scaling.
- [x] P2 satisfies deterministic tie-break rules and validates represented
  number, species mass, signed charge, weighted radius cubed, mean-radius,
  surface, and Riemer diversity/mixing bounds before commit.
- [x] P3 direct Warp resampling agrees with the independent deterministic
  remapping/diagnostic oracle on its explicit-release-count boundary; CUDA
  coverage is optional and skips cleanly when unavailable.
- [x] P4 direct CPU and Warp helpers validate every supplied sidecar before
  writes, scale only selected rows' volume, concentration, and provisional
  demand, return the required resolved-scale diagnostic, and preserve protected
  state and caller identities.
- [x] P4 tests establish CPU/Warp independent-oracle parity, atomic invalid and
  later-invalid rejection, diagnostic-only no-selected behavior, valid empty
  boxes, optional CUDA skips, and concrete-only import surfaces.
- [ ] Arrays remain fixed shape and preserve container/array identity, dtype,
  device, density, requests, and untouched boxes/slots.
- [ ] Focused/full tests, Ruff, mypy, and documentation validation pass without
  lowering coverage.

## Metrics

| Metric | Baseline | Target | Source |
|--------|----------|--------|--------|
| Silently dropped represented demand | Undefined | 0 | Policy diagnostics and oracle |
| Number/species-mass/charge conservation failures | No policy | 0 | Multi-box conservation matrix |
| CPU/Warp plan or state mismatches | No shared API | 0 | Parity tests |
| Invalid exhausted calls with observable writes | E6-F5 rejects capacity only | 0 | Snapshot tests |
| Dynamic resizes/hidden transfers | Out of scope | 0 | Identity tests and review |
| Distribution-moment violations | Undefined | 0 under future P2+ resampling bounds | Independent NumPy oracle |
| Changed-code coverage | Repository threshold | At least 80% | pytest-cov |

## Delivered P1 Evidence (#1422)

- [x] The concrete CPU module exposes strict frozen controls, fixed-shape
  sidecars, immutable plans, policy constants, resolver, and float64
  tuple-backed inventory only at `particula.particles.exhaustion`.
- [x] Resolver tests establish validate-all-before-resolve behavior and preserve
  every supplied sidecar on successful, invalid, and fail-closed calls.
- [x] Focused tests establish capacity activation independent of flags and
  resampling-first selection with deferred releases/scaling; later commit,
  resampling, scaling, discovery, GPU, and export criteria remain open.

## Delivered P2 Evidence (#1423)

- [x] `plan_resampling` creates immutable detached CPU plans from cached,
  validated all-box state, and `apply_resampling` validates all plan/state
  inputs before its only mutation boundary.
- [x] Co-located tests demonstrate stable equal-stratum remapping, independent
  conservation-oracle agreement, exact released-slot clearing, and no mutation
  for malformed, stale, overlapping, or later-box-invalid plans.

## Delivered P3 Evidence (#1424)

- [x] `resampling_step_gpu` provides allocation-stable, fixed-capacity direct
  Warp remapping with caller-owned `ResamplingBuffers`, read-only preflight,
  active-device staged planning, diagnostic gating, and one commit.
- [x] Only the direct step is exported; focused tests cover ownership,
  validation, deterministic parity, diagnostic no-commit behavior, and optional
  CUDA skips.

## Delivered P4 Evidence (#1425)

- [x] `apply_representative_volume_scaling` and
  `representative_volume_scaling_step_gpu` implement the bounded direct P4
  transform with all-box preflight and selected-row-only writes.
- [x] The CPU and Warp helpers remain intentionally unexported; focused
  exhaustion and kernel-export tests verify their concrete-only boundaries and
  the CPU/Warp validation, isolation, conservation, and optional-CUDA contract.
