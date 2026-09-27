# Scope

## Modules and directories

- `particula/dynamics/condensation/condensation_strategies.py`, existing
  mass-transfer utilities and native strategy builders/factories as needed,
  with adjacent `condensation/tests/` scientific fixtures.
- `particula/dynamics/coagulation/coagulation_strategy/`: ABC/native helpers,
  Brownian, charged, sedimentation, turbulent-shear, turbulent-DNS and combined
  strategies; adjacent tests and required `coagulation/particle_resolved_step/`
  consumers for existing binned/resolved execution.
- `particula/dynamics/wall_loss/wall_loss_strategies.py` and BOTH
  `particula/dynamics/tests/wall_loss_strategies_test.py` and
  `particula/dynamics/wall_loss/tests/wall_loss_strategies_test.py`.
- Narrow developer contract guidance and scientific test traceability.

## Interfaces and APIs

Complete native CPU condensation using `ParticleData`/`GasData` and explicit
process-owned activity, surface and vapor-pressure configuration. Complete
native coagulation and neutral/charged wall-loss rates/steps. Environmental
values retain K/Pa meaning from `EnvironmentData`; M4 owns extracting them in
runnables/adapters. Reuse M1 helpers/mapping, M2 native initialization and
current formulas/distribution modes. Enduring scientific fixtures must not
permanently construct facades through `from_representation` or `from_species`.

## Out of scope

M1 contract redesign, M2 aggregate construction/replacement, M4 runnable
composition/dilution/nucleation orchestration/CPU adapters, M5 broad examples
and notebook migration, M6 facade/bridge/export removal. Preserve approved
temporary compatibility until M6; inventory each seam and its consumer.
No new physics, multi-box CPU execution, precision/storage redesign, new GPU
API, hidden transfer/fallback, automatic resident rebinding, capture redesign,
performance claim or Epic I work. CPU-to-Warp container transfers are retained.
