# Testing Strategy

Every implementation phase includes co-located fast tests in the same change.
The configured minimum 80% coverage is retained or improved; no threshold or
scientific tolerance may be weakened. Test modules use the `*_test.py` suffix.

## Per-Phase Approach

- **P1 transport primitives:** In `particula/gpu/properties/tests/` and
  `particula/gpu/dynamics/tests/`, compare each reused or new device function to
  an independent CPU implementation over representative radius, density,
  temperature, and pressure values. **Shipped evidence:**
  `particula/gpu/properties/tests/particle_properties_test.py` covers the
  consolidated property helpers, slip's exact zero/invalid sentinels,
  branch-adjacent Debye quadrature/asymptote behavior against an independent
  host oracle, and `x_coth_x` series/direct threshold and invalid-domain
  behavior. Migration consumers retain their behavioral coverage, and an
  import-surface regression confirms property exports and the absence of moved
  names from `particula.gpu.dynamics`. Warp tests use guarded device execution;
  CUDA remains optional.
- **P2 coefficient functions (shipped, #1402):**
  `particula/gpu/dynamics/tests/wall_loss_funcs_test.py` guarded-imports Warp,
  requires Warp CPU when installed, and marks optional CUDA runs for clean
  availability-based skips. It compares scalar diffusion/gravity regimes and
  vector state/geometry lanes against the CPU system-state oracles, checks
  finite fp64 outputs, and separately smoke-launches both helpers. Rectangular
  parity uses `rtol=1e-10, atol=1e-20`; spherical parity records
  `rtol=1.002e-3` due to the measured CPU Debye endpoint quadrature difference.
- **P3 contract/preflight (shipped, #1403):**
  `particula/gpu/kernels/tests/wall_loss_test.py` is Warp-guarded and covers
  lazy step import, concrete-only frozen configuration, spherical/rectangular
  exclusivity, scalar/per-box/hybrid/explicit-environment forms, particle
  schemas/domains, time, and RNG metadata. Valid calls return the identical
  particles object without writes; rejected calls snapshot and preserve particle
  fields and supplied sidecars. Coefficient helpers are monkeypatched to prove
  that neither valid nor rejected P3 calls invoke them.
- **P4 removal:** In the same kernel test module, cover all-inactive, sparse
  gaps, one/multi-box, one/multi-species, exact `time_step == 0` no-op,
  controlled all-survive/all-remove cases, and complete clearing of mass,
  concentration, and charge. Assert density, volume, survivor values, shapes,
  devices, dtypes, and object/array identities are unchanged.
- **P5 persistent RNG:** Reproduce the coagulation lifecycle matrix: omitted
  convenience state; caller-owned initialize-once; repeated calls without
  reseeding; explicit reset; independent per-box advancement; supplied-buffer
  identity; and invalid-call non-initialization/non-advancement. A fixed seed
  must reproduce Warp runs after explicit reset, not CPU NumPy bitstreams.
- **P6 parity (shipped, #1406):**
  `particula/gpu/kernels/tests/wall_loss_parity_test.py` uses an independent
  NumPy system-state oracle and a non-mutating Warp diagnostic to compare
  complete-slot spherical/rectangular eligibility and coefficient results.
  Eligible finite rates use spherical `rtol=1.002e-3, atol=1e-20` and
  rectangular `rtol=1e-10, atol=1e-20`. The matrix includes one-/multi-box,
  per-box environment, nanometer/micrometer, sparse, and unusable-slot cases.
  Fresh-seed and persistent-sidecar evidence each uses exactly 100 seeds and
  fixed 3-sigma binomial bounds; zero-time and all-inactive paths are exact
  no-ops. Import smoke coverage proves the lazy step remains public while the
  configuration remains concrete-module-only. Warp CPU is the baseline; CUDA
  rows are additive and absent when unavailable.
- **P7 documentation:** Validate links, import snippets, SI units,
  support/deferred tables, and focused commands.

## Regression and Focused Commands

- Keep CPU tests in `particula/dynamics/wall_loss/**/` and
  `particula/dynamics/**/wall_loss*_test.py` green; E6-F3 must not alter CPU
  strategy semantics.
- Keep direct coagulation persistent-RNG and fixed-slot tests green because they
  define shared invariants.
- Focused commands include:
   `pytest particula/gpu/dynamics/tests/wall_loss_funcs_test.py -q -Werror`,
   `pytest particula/gpu/kernels/tests/wall_loss_test.py -q -Werror`, and
   `pytest particula/gpu/kernels/tests/wall_loss_parity_test.py -q -Werror`.
- Deterministic coefficient acceptance uses documented float64 tolerances.
   Stochastic acceptance uses expected distributions, never exact CPU/GPU draw
   order. Zero-time and inactive-slot no-ops remain exact.

## Shipped P4-P5 Evidence (#1404, #1405)

`particula/gpu/kernels/tests/wall_loss_test.py` now exercises P4 in the focused
Warp-guarded suite. It covers both geometry paths, deterministic private-mask
application, exact zero-time no-op behavior, sparse fixed slots across
one-/multi-box and one-/multi-species layouts, complete removal clearing,
controlled survivor/removal paths, seeded same-device results, and aggregate
interior-probability stochastic behavior. Snapshots assert preserved identities,
shapes, dtypes, devices, density, volume, survivors, inactive gaps, and supplied
`rng_states`; invalid pre-launch calls remain atomic.

P5 adds same-device lifecycle evidence in the same module: omitted private
state repeatability; supplied-sidecar initialize-once/reuse; explicit reset;
independent per-box progression; eligible-only consumption; all-ineligible
no-draw; and zero-time/rejection preservation. The opt-in
`@pytest.mark.benchmark` smoke test records the sequential per-box lifecycle
path without a throughput threshold. These tests intentionally do not assert
CPU/Warp or CPU/CUDA stochastic-stream identity.
