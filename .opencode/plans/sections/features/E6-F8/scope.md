# Scope

Deliver the direct, particle-resolved Warp implementation of the bounded E6-F7
nucleation contract. The step works entirely on explicit caller-owned device
state, preserves fixed shapes and identities, and uses E6-F5/E6-F6 services for
capacity rather than creating a second slot or exhaustion model.

## In Scope

- **Delivered P1 (#1438):** concrete-only frozen configuration/sidecar records
  and private read-only Warp preflight in
  `particula/gpu/kernels/nucleation.py`, with co-located Warp tests.

- **Delivered P2 (#1439):** concrete-only `_plan_nucleation_demand(...)` in
  `particula/gpu/kernels/nucleation.py`. It computes survival-included rates,
  potential demand, common inventory-limited admission, planned precursor
   removals, and gate diagnostics, then commits only P2-owned sidecars.

- **Delivered P3 (#1440):** concrete-only `_stage_nucleation_slots(...)` in
  `particula/gpu/kernels/nucleation.py`. It privately converts exact,
  representable `accepted_demand * volume` values to full `wp.int32` counts,
  reuses E6-F5 diagnostics, and commits only the five caller-owned P3/E6-F5
  sidecars. Counts beyond free capacity are retained; selected indices contain
  only the deterministic free-slot prefix and `-1` tails.

- Device evaluation of E6-F7 activation `J=A*C` and kinetic `J=K*C^2` models,
  including the same SI conversions, closed validity domains, composition,
  gates, and no-op semantics.
- Per-box/species inventory finalization before mutation; admitted represented
  particle mass must exactly correspond to gas mass removed.
- Direct integration with E6-F5 fixed-shape slot diagnostics; later activation
  and E6-F6 resampling-first, optional-scaling exhaustion planning.
- Caller-owned same-device `wp.float64`/`wp.int32` configuration, request,
  scratch, work, and exact diagnostic sidecars with stable documented shapes.
- A low-level `nucleation_step_gpu(...)`, intended lazy kernel export, complete
  preflight, atomic all-box planning, and no-op/failure immutability.
- Independent float64 CPU parity, Warp CPU conservation evidence, and optional
  CUDA evidence that skips cleanly when unavailable.

## Out of Scope

- **Still deferred after P3:** a direct execution entry point or export, E6-F6
  capacity policy, slot activation, particle or gas mutation, and any fallback
  allocation. P3 remains a private sidecar-only staging seam.

- New nucleation equations, chemistry, extrapolation, or a full Vehkamaki/CNT,
  ion-induced, heterogeneous, or cluster-dynamics implementation.
- Dynamic particle resizing/append, slot compaction, demand truncation, hidden
  CPU/Warp transfer, CPU fallback, or implicit synchronization.
- A high-level GPU `Runnable`, backend selector, scheduler, graph capture,
  autodiff, multi-box transport, CFD coupling, or performance claim.
- Exact cross-backend floating-point or RNG sequence identity beyond the
  recorded deterministic parity and conservation tolerances.
