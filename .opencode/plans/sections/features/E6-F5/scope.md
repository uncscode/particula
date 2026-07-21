# Scope

Add bounded CPU and direct Warp fixed-slot inspection and activation primitives
for particle-resolved `ParticleData` and `WarpParticleData`, with identical
predicates, deterministic mappings, exact diagnostics, and atomic preflight.

## In Scope

- A canonical active slot: positive finite concentration and positive finite
  total species mass.
- A canonical free inactive slot: zero concentration, every species mass zero,
  and zero charge; mixed or contradictory states are invalid rather than free.
- CPU and Warp read-only discovery of ascending free indices and exact per-box
  active/free counts.
- Fixed-shape activation requests with per-box requested counts; request rank
  `r` maps to the `r`th ascending free slot in that box.
- Exact caller-owned `wp.int32` diagnostics for active, free, and activated
  counts, plus a fixed-shape free-index sidecar using `-1` for unused entries.
- Complete shape, dtype, device, finiteness, state-consistency, request-validity,
  and capacity validation before particle or diagnostic mutation.
- CPU/Warp parity on sparse, multi-box, multi-species, zero-request, and exact-
  capacity cases; Warp CPU required and CUDA optional.

## Out of Scope

- Resampling, merge selection, representative-volume scaling, or policies for
  insufficient capacity; E6-F6 owns those decisions.
- Nucleation equations, gas inventory transfer, or particle-source policy;
  E6-F7 and E6-F8 consume these primitives.
- Dynamic allocation, array resizing, compaction, slot reordering, hidden
  CPU/GPU transfer, host fallback, or synchronization for reporting.
- Changes to container schemas, density, volume, backend selection, a high-level
  GPU runnable, scheduling, graph capture, or performance claims.
