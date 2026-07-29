# Success Criteria

- [x] P1 typed declarations accept exactly the supported closed-catalogue nodes
  and dependency pairs and reject duplicates, cycles, unavailable requirements,
  and malformed declarations before any runtime behavior (#1492).
- [x] P1 equivalent declarations normalize to identical node-ID and edge-pair
  ordering, independent of input declaration order; this is not a scheduler or
  topological execution order (#1492).
- [x] P1 remains unexported, pure, and backend-neutral: it does not import
  Warp/GPU modules, access resources, schedule, execute, or mutate state
  (#1492).
- [x] P2 resolves registration-independent lexical topological order and a
  declaration-only selected schedule with explicit/derived closure, reviewed
  direction policy, and no disabled-endpoint edges (#1493).
- [x] P2 remains direct-import-only and prelaunch-only: no package export,
  lifecycle/resource/backend import, launch, transfer, synchronization, or
   mutation occurs during schedule resolution (#1493).
- [x] P4 accepts only exact immutable requests bound to the active resident
  session, pinned registry, resolved graph, and canonical environment/gas update
  node; malformed bindings reject before payload work or writes (#1495).
- [x] P4 validates fixed Warp schemas, resident device, contiguity, protected
  array aliasing, and physical values before ordered in-place copies; rejected
  calls preserve resident primaries and prescribed payloads (#1495).
- [x] P4 preserves container and primary-array identities, leaves protected
  fields untouched, and treats canonical empty-box/zero-species inputs as
  write-free no-ops (#1495).
- [x] P4 remains concrete-only and direct-import-only: it adds no package
  export, scheduler execution, derived refresh, transport, host transfer,
   fallback, or lifecycle behavior (#1495).
- [x] P5 binds by exact identity to an active resident session, pinned registry,
  resolver-produced graph, and resolved schedule; caller-reported ordinary
  nodes update only coordinator-owned freshness markers (#1496).
- [x] P5 writes stale vapor pressure then saturation immediately before only the
  next scheduled condensation or diagnostics callback, preserves primary
  identities, and uses the resident SI saturation formula with `GAS_CONSTANT`
  (#1496).
- [x] P5 writer/callback failures suppress the consumer and leave the cursor
  unchanged; successful vapor refresh remains fresh if a following saturation
  writer fails (#1496).
- [x] P5 remains direct-import-only and concrete-only: it adds no lifecycle,
  resource acquisition, whole-timestep scheduler dispatch, transfer,
   synchronization, fallback, gas restore, or package export (#1496).
- [x] P6 diagnostics accept only the two closed snapshot operations, exact
  resident bindings, nonaliasing caller-owned outputs, and canonical empty
  write-free no-op schemas (#1497).
- [x] P6 accepts only the exact resolver-produced ten-node schedule; it
  preflights whole-loop ownership/requests before one token and dispatches the
  resolved order with condensation/diagnostics consumer windows (#1497).
- [x] P6 dispatches condensation, Brownian coagulation, dilution, wall loss, and
  nucleation through their owning execution boundaries without physics rewrites
  (#1497).
- [x] Within P5's explicit consumer bracket, reported environment/gas changes
  precede needed derived-state writes and condensation/diagnostics observe the
  current resident gas and thermodynamic state (#1496).
- [x] Repeated normal P6 timesteps perform no bulk conversion, host readback,
  checkpoint/finalize call, explicit synchronization, or silent fallback (#1497).
- [x] Resident container, array, sidecar, diagnostic-output, and RNG identities
  and fixed shapes remain stable across successful P6 timesteps (#1497).
- [x] P6 prelaunch failures preserve state; post-writer uncertainty faults the
  session and retains the original exception without rollback claims (#1497).
- [x] Warp CPU complete-loop tests pass with explicit deterministic tolerances,
  conservation/loss checks, and independent-box equivalence; optional CUDA
  skips cleanly (#1497).
- [x] P7 documentation states the concrete-only scope, profile/graph-dependent
  order, freshness and authority limits from issue #1451, and does not absorb
  E7-F7/E7-F8/E7-F9, Epic H, or Epic I work (#1498).

## Metrics

| Metric | Baseline | Target | Source |
|--------|----------|--------|--------|
| Canonical supported processes in one step | Illustrative direct calls only | 5 | Scheduler integration tests |
| Intermediate bulk transfers/syncs per normal GPU timestep | Caller-dependent | 0 | Conversion/sync spies |
| Stale thermodynamic consumer paths | Possible in ad hoc order | 0 | Graph/order tests |
| Registration-order variants yielding identical order | Not defined | 100% | Process-graph parametrization |
| Changed-module coverage | N/A | >=80% | pytest-cov |
| Required routine GPU backend | None integrated | Warp CPU | CI/focused tests |
