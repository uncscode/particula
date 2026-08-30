# Scope

E8-F3 extends the concrete-only resident resource boundary to prepare one
complete, immutable capture resource set for a fixed session and E8-F2 prepared
timestep. It covers process, communication, diagnostic, control, validation,
and RNG sidecars; exact identity reuse; atomic publication; and overflow-safe
logical byte accounting.

## In Scope

- Define canonical capture-resource roles, shapes, dtypes, capacities, and
  deterministic manifest order in `particula/execution/gpu_resources.py`.
- Fill gaps in registry-owned preallocation for prepared process controls,
  validation/status work, selected-lane work, diagnostics, communication, and
  persistent RNG state required by the fixed prepared sequence.
- Add one setup-only capture-set acquisition operation that validates all
  supplied storage, allocates omitted storage, and publishes only a complete
  nonaliasing set.
- Pin the exact session, resource views, native records, arrays, optional
  communication map, capacities, and prepared-resource signature.
- Return established views and arrays by identity on compatible reacquisition,
  with no allocation, reseeding, transfer, synchronization, or payload read.
- Compute deterministic logical bytes from checked shape products and manifest
  dtype sizes, with per-role, per-family, and total summaries.
- Add co-located unit and integration tests, plus developer-facing contract
  documentation.

## Out of Scope

- Graph capture/replay lifecycle itself (E8-F1) or process enqueue refactoring
  (E8-F2).
- CPU/uncaptured/captured physics parity (E8-F4), scaling benchmarks (E8-F5),
  the broader state/checkpoint/tape memory model (E8-F6), examples (E8-F7), or
  profiling/closeout (E8-F8).
- Runtime resizing, compaction, automatic recapture, fallback, migration,
  allocator-pool tuning, or claims about allocator overhead/reserved bytes.
- New public package exports or changes to direct-kernel ownership contracts.
