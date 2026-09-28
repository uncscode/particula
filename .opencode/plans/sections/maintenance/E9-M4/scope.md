# Scope

## Modules and directories

- `particula/runnable.py`: retain `RunnableABC.__or__`, sequence ordering and
  full-cycle-per-substep execution; migrate native fixtures in
  `particula/tests/runnable_test.py`.
- `particula/dynamics/particle_process.py`: native `rate`/`execute` access for
  `MassCondensation`, `Coagulation`, `WallLoss`, `Dilution` and `Nucleation`;
  remove native-path reliance on atmosphere, facade strategy and cache state.
- `particula/dynamics/dilution.py`: all-gas native preflight, physical rates,
  concentration updates, snapshots and documented error propagation.
- `particula/dynamics/tests/`: dilution strategy/runnable/export tests,
  nucleation runnable and wall-loss runnable behavioral fixtures.
- `particula/dynamics/nucleation/particle_source.py` and adjacent tests are
  retained transaction authority; change only a necessary native orchestration
  interface and approved count/volume normalization, never its scientific law
  or resampling-before-scaling exhaustion policy.
- `particula/execution/adapters/condensation.py` and `coagulation.py`: existing
  CPU carriers/dispatch and corresponding adapter/integration tests under
  `particula/execution/tests/`; retain concrete-only exports.
- `particula/integration_tests/`: native composed-process regression fixtures.
  Narrow developer contract documentation accompanies closeout.
- Required D1–D2 corrections in CPU/GPU dilution, nucleation, exhaustion,
  communication and volume evolution; resident/prepared composition, metadata
  compatibility and versioned checkpoint restore. Preserve lifecycle ownership.

## Interfaces and acceptance boundary

Use the approved M1 units/mapping and M2 accessor spelling rather than inventing
another aggregate interface. Process configuration remains process-owned;
environmental values come from the held environment, not a stale second
authority. M3 strategies perform science. Existing selected CPU adapters keep
their supported profiles and exactly-once delegation, not a broader CPU/GPU
capability matrix. Native workflows must not construct facades internally.

## Out of scope

M1 contract redesign, M2 builders/presets/replacement implementation, M3 physics
rewrites, M5 broad tutorials/notebooks, and M6 final class/export/bridge deletion
are separate. No general multi-box CPU execution, permanent compatibility
facade, new exports, new adapter families, hidden transfer/synchronization,
fallback, automatic GPU rebind, resident scheduler or graph-capture redesign,
precision/array-layout changes beyond approved metadata, performance claims or
Epic I implementation. `execution/process_adapters.py` remains resident GPU
infrastructure; only required D1–D2 semantic integration is included, not a CPU
adapter redesign. CPU↔Warp conversion helpers remain supported.
