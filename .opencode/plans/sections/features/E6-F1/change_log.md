# Change Log

## 2026-07-21 — P2 Complete (issue #1390)

- Added unexported concrete-only `dilute_aerosol()` in
  `particula/dynamics/dilution.py`.
- Implemented strict scalar coefficient/time validation, exact physical
  particle and atmosphere gas-group decay, representation-volume conversion,
  complete preflight, ordered commit, and rollback recovery.
- Added regression coverage in `particula/dynamics/tests/dilution_test.py` for
  normal and no-op behavior, underflow, boundary errors, preflight atomicity,
  and commit-failure recovery.
- Deliberately made no strategy/runnable, package-export, public-documentation,
  or GPU changes.

## 2026-07-21 — P1 Complete (issue #1389)

- Implemented validated, NumPy-broadcasting coefficient and instantaneous-rate
  helpers in `particula/dynamics/dilution.py`.
- Added the exact, concrete-module-only `get_dilution_step()` and preserved the
  existing `particula.dynamics` export surface.
- Added P1 numerical-contract coverage in
  `particula/dynamics/tests/dilution_test.py`, including validation,
  broadcasting, no-op, extreme-decay, non-mutation, and export-boundary cases.
- Deliberately made no changes to containers, strategies/runnables, public user
  documentation, examples, GPU code, or package exports.

## 2026-07-21 — Initial Draft

- Created the first-pass E6-F1 feature plan from issue #1377 track T1.
- Preserved the issue acceptance contract: particle and gas concentrations
  match the CPU reference; zero flow/time are exact no-ops; particle mass,
  charge, density, and volume remain unchanged.
- Recorded no inbound feature dependency, E6-F2 as the direct downstream parity
  consumer, and E6-F9 as the integrated closeout consumer.
- Added five issue-sized phases with co-located tests and a final documentation
  phase.
- Bounded scope to CPU strategy/runnable work and explicitly deferred direct
  GPU, backend selection, scheduling, and unrelated Epic F processes.
- Research challenge: the required `codebase-researcher` delegation was blocked
  by the subagent-depth limit, so the draft used issue #1377, E6 sections,
  roadmap references, and direct repository reads instead.
