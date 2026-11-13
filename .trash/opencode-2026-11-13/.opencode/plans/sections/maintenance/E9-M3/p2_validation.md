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

Validation on 2026-11-11 against CPU P2 commit `ab5b84798`:

- `.venv/bin/pytest particula/dynamics/condensation/ -q --no-cov --tb=short`:
  636 passed (latest confirmation 5.39 s).
- `.venv/bin/pytest particula/dynamics/condensation/tests/staggered_performance_test.py
  -v -m 'slow and performance' --no-cov`: 8 passed, 1 deselected. This preserves
  the existing manual reproduction; it makes no native performance claim.
- `PATH="$PWD/.venv/bin:$PATH" .opencode/tools/run_linters.py`: all Ruff/mypy
  checks passed. Logs are in `.artifacts/e9-m3/p2-cpu/`.
- Focused tests establish assertions, not full-package coverage. Full current
  revision coverage and final completion checks remain outstanding.

## GPU stepping implementation checkpoint

The pending GPU slice replaces the interim condensation guard with explicit
native admission. Tagged PMF/resolved calls require their exact species
binding; PDF and untagged binding misuse reject. Full particle composition
remains separate from complete gas order in direct and prepared execution.
Both wrappers share the existing raw physics implementation, including its
one-species autodiff branch; the legacy kernel signature is unchanged.

Native inventory reductions use N/V for resolved gas coupling and stored PMF
density at V=1. Finalization records representable particle changes. Extensive
energy uses raw resolved counts (PMF density times unit volume), full-gas latent
parameters and full-gas output width. Transfer scratch remains material ordered;
demand/release/scale buffers follow full gas order. Thermodynamics refreshes all
gas lanes, even unused ones. Material activity references match declared mapped
gas pairs; unmapped material remains part of composition and geometry.

Private prepared calls retain all arrays, maps, volume references and semantic
configuration. Enqueue does not allocate, transfer, read back, synchronize,
validate or resolve changed carrier attributes. Partitioning values remain live
full-gas state. Setup checks physical state and overlapping writers before
mutation; zero-sized arrays have empty memory ranges. The standalone private
inventory test helper stays legacy-only and cannot silently reinterpret tags.
No automatic resident/checkpoint integration or new graph-capture claim follows.

`particula/gpu/kernels/tests/condensation_native_test.py` supplies 230 cases
across Warp CPU and available CUDA: independent four-substep equations,
particle/gas deltas checked separately, species inventory at V=.25/1/4, PMF,
unequal counts, full-gas skipped/disabled lanes, latent energy, single-material
nonidentity mapping, both unequal-width directions, ideal/kappa activity and
static/weighted surface tension, no-work/empty state, malformed input snapshots,
array identity and frozen prepared execution/replay.

Validation on 2026-11-11 (repository environment, normal Warp cache/hardware
access), against CPU commit `ab5b84798` plus this GPU implementation diff:

```bash
.venv/bin/pytest particula/gpu/kernels/tests/condensation_native_test.py \
  particula/gpu/kernels/tests/condensation_test.py \
  particula/gpu/kernels/tests/condensation_graph_capture_test.py \
  particula/gpu/kernels/tests/condensation_autodiff_test.py \
  particula/gpu/kernels/tests/condensation_stiffness_test.py \
  particula/gpu/tests/population_metadata_test.py \
  particula/gpu/tests/population_boundary_test.py -q --no-cov --tb=short
PATH="$PWD/.venv/bin:$PATH" .opencode/tools/run_pytest.py
PATH="$PWD/.venv/bin:$PATH" .opencode/tools/run_linters.py
.venv/bin/mkdocs build --strict
```

- Focused: 601 passed, 4 skipped, 1 xfailed in 2.43 s. All 230 native cases
  passed, including CUDA on NVIDIA GeForce RTX 5060 (Warp 1.11.1). Logs:
  `.artifacts/e9-m3/p2-gpu/focused-tests.log` and `capability-skips.log`.
  Three legacy tests intentionally skip removed host-readback checks; Warp CPU
  graph capture skips because it requires CUDA. The expected CUDA failure is
  public-step capture with validation readbacks, not private native enqueue.
  The literal capability reproduction is
  `.venv/bin/pytest particula/gpu/kernels/tests/condensation_test.py
  particula/gpu/kernels/tests/condensation_graph_capture_test.py -q --no-cov
  --tb=short -rsx` (284 passed, 4 skipped, 1 xfailed).
- Untargeted package: 7,540 passed, 19 skipped in 170.44 s; 92% coverage against
  the unchanged 80% gate. Log: `.artifacts/e9-m3/p2-gpu/package-tests.log`.
- Repository Ruff/mypy and strict documentation build passed. Logs:
  `.artifacts/e9-m3/p2-gpu/linters.log` and `docs.log`.

## Phase handoff

P2 CPU/GPU implementation and focused/full-package validation are complete at
this branch checkpoint. Scientific review and the eventual single-PR final
revision gate remain pending. P3/P4 coagulation, P5 wall loss and P6 integration,
assertion/compatibility reconciliation, examples/infrastructure tests and final
M4 handoff are still required. M3 is not complete.
