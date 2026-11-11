# P4 direct/prepared GPU coagulation checkpoint

Entry: CPU checkpoint `9c76f390a`. This record covers the next GPU commit,
not completion of P4 or M3. The CPU fixed-grid species-conservation correction
in [p4_validation.md](p4_validation.md) remains unfinished.

## Supported native contract

- Native resolved metadata is accepted. PMF/PDF and unknown codes reject before
  payload access. Untagged legacy callers retain separately tested behavior.
- Supported masks remain 1/2/4/8/3/5/6/9/10/12/15: Brownian, charged hard sphere,
  SP2016 sedimentation, ST1956 shear, the existing pairs and four-way sum.
  Deferred triples and GPU DNS remain unsupported.
- Native storage stays fp64 with unchanged layouts. All mechanisms validate
  finite nonnegative masses/counts, finite charge, positive densities, physical
  derived mass/volume and N/V, and equal positive active counts within each box.
  Inactive slots can retain finite material/charge sentinels. Existing charged
  and sedimentation caps remain specific to those mechanisms.
- Primary particle volume is the native count basis. A separate scalar/array
  volume must agree exactly. Invalid state, unsupported weights, incompatible
  volume and overlapping writer/protected storage reject before caller writes
  or RNG initialization. Read-only preflight can allocate flags and synchronize;
  post-launch rollback is not promised.
- One shared selector retains existing pair laws, majorants, bounded scheduling,
  candidate/acceptance draws, active-pair removal, representable-merge admission
  and output compaction. The original kernel signature delegates with legacy
  scheduling. A private native wrapper includes current common count for every
  supported mask: `trials = K_majorant * pair_count * count * dt / V`.
  It does not allocate an effective V/count array or freeze host-derived counts.
- The unchanged apply kernel retains recipient counts, sums original species
  masses/charge, and clears donors. Equal active counts establish weighted
  inventory conservation; unequal weights are rejected.
- The frozen prepared record pins native scheduling and device references.
  Enqueue reads current device counts without allocation, readback,
  synchronization, reseeding or metadata resolution. Callers retain compatible
  state and keep resources alive. Changed bindings/semantics require explicit
  preparation. Resident/checkpoint integration remains M4-owned.

## Owning scientific and boundary tests

`particula/gpu/kernels/tests/coagulation_native_test.py` contains 328 passing
cases on installed Warp CPU and available CUDA (no native skips):

- All eleven mechanism rows, direct/prepared, three replays, multiple materials,
  volumes .25/1/4 and common counts .25/1/4. Independent count-weighted species
  and charge inventories, exact identities, donor/recipient accounting and
  inactive sentinels. Empty/one-active and zero-time cases preserve RNG.
- Independent one-pair fixtures use explicit NumPy reference equations for
  enabled mechanisms. A known kernel fixes the duration, then Bernoulli
  expectations use `K * count * dt / V`, not device scratch or observed outputs.
  Nine volume/count combinations each have 1,024 box streams; probabilities
  range from .003125 to .8. Six standard errors plus one sample bound each
  observed mean; species inventories are checked separately.
- The same matrix prepares with unit counts and changes counts on the pinned
  arrays before enqueue, proving that prepared scheduling reads current counts.
  This is not a CPU/GPU per-seed trajectory equality claim.
- Malformed schema rejects before launch. Physical mutation, occupied zero
  material, wrong volume, writer aliases across primary/output/RNG/environment
  storage, and unsupported mechanisms reject with unchanged state snapshots.
- Explicit environment/direct-array inputs, matching scalar/array volume,
  raw-count round trip and unchanged mechanism-specific active bounds.
- Prepared reference pinning after carrier rebinding, with allocation,
  readback, synchronization, metadata resolution and RNG initialization patched
  to fail during enqueue. Compilation is warmed outside the assertion.

Existing coagulation runtime, independent-law and stochastic suites retain
their legacy assertions. The structural singleton-shear majorant diagnostic
now inspects the shared implementation; its O(A) and alternate pair-scan checks
remain. `population_boundary_test.py` replaces interim resolved rejection with
PMF/PDF-before-payload rejection. New numerical tests establish acceptance.

## Validation evidence

2026-11-11 against `9c76f390a` plus this GPU diff:

- Old/new coagulation and population boundaries: 1,276 passed in 9.81 s before
  adding the final 12 environment/active-limit cases.
  `.artifacts/e9-m3/p4-gpu/coagulation-tests.log`.
- Final native module: 328 passed in 14.52 s, Warp CPU and CUDA, no skips.
  `.artifacts/e9-m3/p4-gpu/native-tests.log`.
- Untargeted `.opencode/tools/run_pytest.py`: 8,370 passed, 19 skipped in
  175.07 s; 92% coverage against the unchanged 80% threshold.
  `.artifacts/e9-m3/p4-gpu/package-tests.log`.
- `PATH="$PWD/.venv/bin:$PATH" .opencode/tools/run_linters.py`: passed Ruff and
  mypy. `.artifacts/e9-m3/p4-gpu/linters.log`.
- `.venv/bin/mkdocs build --strict`: passed in 1.53 s.
  `.artifacts/e9-m3/p4-gpu/docs.log`.

The initial shared-function extraction failed Warp CPU compilation because
`wp.tid()` requires launch scope. Both wrappers now pass the box index to the
shared implementation. The full runtime matrices above passed after correction;
no physics assertion was removed to bypass the compiler error.

No M3 completion, M4 acceptance, performance, general graph capture, new
precision/layout, unsupported mechanism or unequal-weight algorithm is claimed.
