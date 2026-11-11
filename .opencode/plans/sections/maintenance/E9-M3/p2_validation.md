# P2 implementation evidence (in progress)

P1 source checkpoint: `95dd3d78d`, following CPU admission/rates commit
`673007b9c`. P1 full-package validation passed 7,078 tests, with 19 skips and
92% coverage. This is baseline evidence, not coverage evidence for P2.

## CPU stepping implementation checkpoint

- Isothermal and latent-heat native steps stage changes before publishing to
  caller-owned arrays. Evaporation is bounded before crediting released gas to
  uptake. Gas deltas use actual representable particle mass changes. Read-only
  destinations and protected-field storage overlap reject before mutation.
- Population weighting uses resolved counts/V, PMF density and full-radius-grid
  PDF quadrature. Full-gas mapping preserves unmapped material and gas lanes.
  `rate` documents native per-node contributions in kg/m^3/s; per-particle
  `mass_transfer_rate` remains kg/s. Latent energy uses extensive applied mass
  in J. A disabled gas update retains the external reservoir unchanged.
- Staggered stepping retains two theta passes, seed/generator/shuffle/batch
  controls and the existing frozen original material/radius evaluation. Each
  node uses its own theta duration. Later batches see current private gas and
  remaining particle inventory; complete physical state publishes only after
  both sweeps. PDF weights are computed on the full grid before batching.
  Late model failures preserve caller physical arrays; no RNG rollback is
  promised after stochastic scheduling has begun.
- Temporary untagged/facade algorithms remain separately supported.

New independent fixtures: `native_step_test.py` (58 tests),
`native_staggered_test.py` (102 tests), and `native_step_admission_test.py`
(72 tests). These cover PMF/PDF and resolved V=.25/1/4, unequal weights,
mapped uptake/evaporation, fixed reservoirs, extensive latent heat, actual
inventory limits, current-gas coupling, per-node theta durations, RNG
continuation, repeated steps, no-work cases, storage identities and rejection
snapshots. Equation oracles do not call production physics/weight helpers.

Validation on 2026-11-11 against the pending CPU P2 changes:

- `.venv/bin/pytest particula/dynamics/condensation/ -q --no-cov --tb=short`:
  636 passed (latest confirmation 5.39 s).
- `.venv/bin/pytest particula/dynamics/condensation/tests/staggered_performance_test.py
  -v -m 'slow and performance' --no-cov`: 8 passed, 1 deselected. This preserves
  the existing manual reproduction; it makes no native performance claim.
- `PATH="$PWD/.venv/bin:$PATH" .opencode/tools/run_linters.py`: all Ruff/mypy
  checks passed. Logs are in `.artifacts/e9-m3/p2-cpu/`.
- Focused tests establish assertions, not full-package coverage. Full current
  revision coverage and final completion checks remain outstanding.

## Remaining P2 work

Direct/prepared GPU condensation must consume the explicit species binding,
normalize resolved count density, preserve full-gas thermodynamics and
partitioning, and account for extensive latent energy. Add independent
fixed-four-substep scientific fixtures, distinct particle/gas comparisons,
non-unit-volume conservation, disabled/skipped lanes and prepared execution
checks. Replace the interim tagged-condensation rejection guard with verified
positive admission. P2 is not complete; no M3 completion or M4 handoff is claimed.
