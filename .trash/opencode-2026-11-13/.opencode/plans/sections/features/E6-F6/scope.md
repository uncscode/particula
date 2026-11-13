# Scope

E6-F6 adds CPU and direct Warp exhaustion planning and commit primitives for
fixed-slot particle-resolved boxes, preserving E6-F5 classification and
activation contracts.

## In Scope

### Delivered in P1 (#1422)

- Concrete CPU-only read-only planning in
  `particula/particles/exhaustion.py` with focused tests in
  `particula/particles/tests/exhaustion_test.py`.
- Strict frozen controls, fixed-shape `int32` sidecars, all-box pre-resolution
  validation, immutable tuple-backed plans, and resampling-first deferred
  policy selection.
- Float64 tuple-backed weighted number, per-species mass, charge, and
  volume-normalized inventory accounting.

- A policy/configuration contract with `resampling=True` and
  `representative_volume_scaling=False` defaults.
- Independent controls and explicit resampling-first precedence when both are
  enabled.
### Delivered in P2 (#1423)

- CPU-only immutable plan-then-commit equal-weight resampling in
  `particula/particles/exhaustion.py`, with no resize, append, compaction, or
  activation.
- Cached all-box physical-state/P1-record preflight; deterministic stable-order
  interval-sweep remaps; conservation and named-moment diagnostics; and a
  single all-box-preflighted mutation boundary.
- Co-located coverage in `particula/particles/tests/exhaustion_test.py` for
  conservation, tie ordering, detached plans, fixed-capacity slot clearing, and
  invalid/stale/later-box plan atomicity.

### Delivered in P3 (#1424)

- Allocation-stable direct Warp resampling in
  `particula/gpu/kernels/exhaustion.py` using explicit release counts and
  concrete-only caller-owned `ResamplingBuffers`.
- Read-only preflight, active-device staged bitonic-sort/interval-sweep
  planning, diagnostic status gating, and one all-box commit; failed planning
  skips commit and successful calls preserve fixed capacity and caller
  ownership.
- Export of `resampling_step_gpu` only through `particula.gpu.kernels`, plus
  focused tests in `particula/gpu/kernels/tests/exhaustion_test.py` for parity,
  validation, diagnostics, ownership, and optional CUDA execution.

### Delivered in P4 (#1425)

- Concrete-only CPU `apply_representative_volume_scaling` and direct Warp
  `representative_volume_scaling_step_gpu` in their respective exhaustion
  modules. Neither API is re-exported.
- All-box preflight of particle state and caller-owned length-`B` sidecars,
  including factor/minimum-volume bounds and selected-row scaled-volume
  feasibility. Successful selected rows scale volume, concentration, and
  provisional source demand by the same requested factor; other rows retain
  state and receive `resolved_scale == 1.0`.
- Focused CPU and Warp tests for independent-oracle parity, isolation,
  protected-state preservation, atomic rejection, diagnostic-only no-selected
  calls, valid empty boxes, optional CUDA, and concrete-only export surfaces.

### Delivered in P6 (#1427)

- Test-only conservation validation in
  `particula/particles/tests/exhaustion_test.py` and
  `particula/gpu/kernels/tests/exhaustion_test.py` for existing P2 resampling
  and P4 representative-volume scaling.
- Independent NumPy float64 particle/source ledgers cover multi-box,
  multi-species sparse and full rows, zero-demand and all-inactive no-ops,
  highest valid releases, and over-capacity rejection without caller mutation.
- Deterministic downstream-shaped source fixtures are committed only after the
  tested boundary; they prove per-box `pre + source` and, for P4,
  volume-inclusive `scale * pre + source` accounting without implementing a
  policy or source-construction surrogate.

P6 is a no-public-contract phase: it changes neither production behavior nor
public APIs, exports, or documentation contracts.

### Delivered in P7 (#1428)

- Documentation-only P7 updates for CPU planning and direct Warp primitive
  contracts, including validation commands and explicit deferred boundaries;
  P7 adds no pytest module or runtime behavior.

## Out of Scope

- Dynamic allocation, array resizing/appending, compaction, hidden transfers,
  CPU fallback, or container-schema changes.
- E6-F5 authoritative slot discovery, free-index classification, and
  activation; E6-F6-P5 runtime policy composition remains deferred until that
  boundary exists.
- Nucleation rate physics, CPU source construction and gas-inventory mutation
  (E6-F7), or direct Warp nucleation (E6-F8).
- E6-F9 integrated direct-process sequencing and runtime exhaustion handling.
- High-level GPU `Runnable` APIs, backend selection, scheduling, resident loops,
  graph-capture claims, differentiability, and performance claims (Epic G+).
- Exact CPU/CUDA RNG-stream matching; the bounded design is deterministic and
  does not add policy-owned stochastic state.
- Runtime policy composition or source-packaging integration for the P4 direct
  helpers; E6-F5/E6-F6-P5 remain deferred.
