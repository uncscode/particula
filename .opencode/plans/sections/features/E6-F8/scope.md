# Scope

Deliver the direct, particle-resolved Warp implementation of the bounded E6-F7
nucleation contract. The step works entirely on explicit caller-owned device
state, preserves fixed shapes and identities, and uses E6-F5/E6-F6 services for
capacity rather than creating a second slot or exhaustion model.

## In Scope

- Device evaluation of E6-F7 activation `J=A*C` and kinetic `J=K*C^2` models,
  including the same SI conversions, closed validity domains, composition,
  gates, and no-op semantics.
- Per-box/species inventory finalization before mutation; admitted represented
  particle mass must exactly correspond to gas mass removed.
- Direct integration with E6-F5 fixed-shape slot diagnostics/activation and
  E6-F6 resampling-first, optional-scaling exhaustion planning.
- Caller-owned same-device `wp.float64`/`wp.int32` configuration, request,
  scratch, work, and exact diagnostic sidecars with stable documented shapes.
- A low-level `nucleation_step_gpu(...)`, intended lazy kernel export, complete
  preflight, atomic all-box planning, and no-op/failure immutability.
- Independent float64 CPU parity, Warp CPU conservation evidence, and optional
  CUDA evidence that skips cleanly when unavailable.

## Out of Scope

- New nucleation equations, chemistry, extrapolation, or a full Vehkamaki/CNT,
  ion-induced, heterogeneous, or cluster-dynamics implementation.
- Dynamic particle resizing/append, slot compaction, demand truncation, hidden
  CPU/Warp transfer, CPU fallback, or implicit synchronization.
- A high-level GPU `Runnable`, backend selector, scheduler, graph capture,
  autodiff, multi-box transport, CFD coupling, or performance claim.
- Exact cross-backend floating-point or RNG sequence identity beyond the
  recorded deterministic parity and conservation tolerances.
