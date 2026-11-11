# P5 direct/prepared GPU wall-loss checkpoint

Entry revision: `80dd07553`; this record accompanies the GPU P5 change.
The CPU gate is recorded separately in `p5_cpu_validation.md`. Final M3
integration, final-revision gates and scientific review remain P6 work.

## Contract and implementation

- Admit explicit resolved raw-count metadata in the existing direct/prepared
  boundary. Keep untagged legacy dispatch and reject tagged PMF/PDF before
  payload access. No removal kernel or probability law changes were needed:
  each eligible representative survives with `exp(-k*dt)` and keeps its count.
- Validate current physical/derived state and nonoverlapping contiguous
  native storage before caller-owned writers. Unequal positive weights and
  V=.25/1/4 are supported. Protected sources include density, volume, raw and
  normalized environment arrays, saturation inspection storage and field data.
- Retain the exact neutral/charged spherical/rectangular kernels, signed field
  conventions, image charge and slot clearing. GPU removal clears M/count/q;
  CPU charge-retention behavior is not changed or presented as identical.
- Prepared calls pin references and scalar arguments. Enqueue has no host
  validation, allocation, readback, transfer, synchronization or RNG setup.
  The frozen kernel reset flag still applies on each eligible launch. Supply
  initialized state and `initialize_rng=False` for persistent replay. A true
  flag intentionally resets on each launch; zero-work preserves supplied words.
- Selected-box entry points are resident-only M4 seams and reject tagged
  native data. Add the missing guard to the unprepared selected-box boundary;
  do not infer native resident support from direct/prepared support.

## Independent evidence

`wall_loss_native_test.py` covers both Warp CPU and optional CUDA:

- Species/weighted-charge sinks, survivors and inactive storage at V=.25/1/4;
  unequal fractional/multiple counts; independent extensive and density sums.
- Weighted survival mean `p*sum(w)` and variance `p*(1-p)*sum(w**2)` for
  4,095 representatives, survival .2/.5/.8, four configurations, direct and
  prepared calls. Expected coefficients use independently derived radius and
  harmonic material density supplied to existing CPU coefficient laws.
  This tests native plumbing/statistics, not a rederivation of those laws.
  Six-sigma bounds plus .002*sum(w) retain the existing charged coefficient
  parity allowance; species sinks retain rtol=3e-15, atol=0 with accurate sums.
- Same-backend native/legacy persistent streams and explicit reset replay;
  no exact cross-backend RNG claim. Frozen-reference launch-spy checks are
  structural evidence, with actual prepared science covered separately.
- Current malformed state, overflow, occupied zero mass, invalid dtype/stride,
  aliases, empty/inactive/zero-time calls and direct/explicit environment
  ownership. fp32 direct environment arrays normalize locally; typed
  WarpEnvironmentData remains fp64.

Initial failures were test-fixture errors: large naive reductions lost ulps,
and Warp typed structs reject fp32 member assignment before the API runs.
Use accurate independent sums and a plain carrier for malformed dtype
admission; no scientific or coverage tolerance was relaxed.

## Validation

Repository `.venv`, CPU and CUDA available:

```bash
pytest particula/gpu/kernels/tests/wall_loss_native_test.py \
  particula/gpu/tests/population_boundary_test.py \
  particula/gpu/kernels/tests/wall_loss_test.py \
  particula/gpu/kernels/tests/wall_loss_parity_test.py \
  particula/gpu/dynamics/tests/wall_loss_funcs_test.py \
  particula/gpu/tests/conversion_test.py \
  particula/gpu/tests/kernel_exports_test.py -q --no-cov --tb=short
python .opencode/tools/run_linters.py --no-auto-fix
mkdocs build --strict
python .opencode/tools/run_pytest.py
```

Focused result: 624 passed, 1 skipped in 7.77 s. Repository Ruff/mypy passed;
strict docs built in 1.44 s. Logs: `.artifacts/e9-m3/p5-gpu/`.
Untargeted package validation passed in 148.38 s at 92% coverage against the
unchanged 80% threshold; see `package-tests.log` for the assertion totals.
