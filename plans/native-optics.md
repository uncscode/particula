# Native CPU/GPU optics buildout

Accepted 2026-11-15. Branch `codex/mie-optics`, isolated worktree based on
upstream main. Push progress commits to origin. Create one upstream PR only
after every completion gate below passes; no merge or release is authorized.

## Required outcome

Implement the complete numerical stack under `particula/optics`: homogeneous
sphere Mie coefficients and efficiencies; dimensional cross sections; angular
scattering; sphere Mueller matrices and Stokes transformations; population
optics; optical depth, transmission and mass-specific opacity. CPU and real
CUDA float64 qualification extends through size parameter 10,000. This is the
foundation for a later custom process that changes aerosol/environment state;
feedback, transport, inverse retrievals, core-shell/nonspherical models and
material mixing laws are outside this PR.

## Contracts

- Public CPU functions use SI radius and vacuum wavelength, angles in radians,
  absolute particle index n+ik (k >= 0), and positive real medium index (1 by
  default). Normalize exactly once: relative index m/n_medium and size
  parameter 2*pi*r*n_medium/lambda_vacuum.
- Broadcast-compatible numerical inputs, explicit singleton axes for outer
  products; append angle and then 4x4 matrix axes. Named results, reusable
  coefficients and valid term counts. Lazy explicit GPU imports.
- Outputs: Qext/Qsca/Qabs/g/Qback/Qpr, corresponding cross sections [m2],
  S1/S2, polarized/unpolarized differential cross sections [m2/sr], full sphere
  Mueller matrices, phase matrices normalized to unit integral of element 11.
- Stokes: Q=|Eparallel|2-|Eperpendicular|2, U=2 Re(Eparallel Eperpendicular*),
  V=-2 Im(Eparallel Eperpendicular*). Document scattering frames, rotation
  direction and matrix multiplication order.
- Population weights: discrete number densities; represented count/volume;
  radius-PDF quadrature. Sum cross sections/Mueller elements, not amplitudes.
  Bulk g uses scattering weights, SSA uses summed scattering/extinction.
  Numerical GPU quadrature does not imply native GPU PDF simulation support.
- Geometry is explicit: layer thickness/path length, not inferred from volume.
  Return layer/total optical depth and direct-beam exp(-tau). Mass-specific
  opacity requires supplied mass density.
- Zero radius, inactive and index-matched cases produce zero scattering.
  Undefined normalized results use zero with validity indicators. Reject
  invalid input and x > 10000. Bound convergence; never silently truncate or
  conceal substantial negative absorption. CPU raises, asynchronous GPU
  reports explicit status requiring observation before consuming failed data.

## Numerical and GPU implementation

Implement independent published Mie equations using bounded continued fraction,
downward logarithmic derivatives, stable Riccati-Bessel recurrence and an
explicit small-sphere treatment. Reuse coefficients across angular calculations.
Handle forward/backward endpoints exactly and integrate over solid angle.

Warp uses float64 real/imaginary components, case/angle tiling, explicit
workspace allocation and caller-owned device input/output buffers. Repeated
enqueue does not allocate, download, synchronize or fall back to CPU. Offer
bulk-only reductions without retaining all per-particle angular outputs.

## Reference evidence

Generate deterministic JSON in an external disposable environment against
PyMieScatt 1e79410c21fc9ef36088dd06fee83a31aa42b0dc. Commit data, provenance,
source hashes, versions and regeneration recipe. No PyMieScatt dependency in
runtime, development, testing or extras; tests read fixtures offline. Preserve
raw reference values separately from conversions; document known cross-section,
medium and approximate-angular discrepancies, never encode them as correct
physics. Supplement with independent analytic and published numerical oracles.

Cover small through x=10000, absorbing/nonabsorbing/index-matched/index-below-one
cases, endpoints and fine forward angles, coefficients/amplitudes/Mueller
entries, discrete/resolved/PDF reductions and single/multiple layers.

## Serial implementation and acceptance

- [ ] Reference corpus, provenance, conventions and independent expectations.
- [ ] CPU coefficients, efficiencies and cross sections.
- [ ] CPU angular scattering, Mueller matrices and Stokes helpers.
- [ ] Warp equivalents and CPU/Warp-CPU/real-CUDA parity.
- [ ] Population optics, path/column helpers on both backends.
- [ ] Executed spectral, polarization, cloud-depth and CUDA examples with
      saved compact numerical data; later-feedback usage sketch only.
- [ ] Google-style SI/shape/domain docstrings, primary citations, guide/API
      navigation and numerical/performance validation record.
- [ ] Rayleigh limits, lossless energy balance, extinction=scattering+absorption,
      bounded g; angular integrals recover Csca/g and normalized phase.
- [ ] Polarization sign/rotation round trips, valid Stokes states, medium/unit
      invariance, population equivalence, optical-depth additivity, near-zero
      and invalid inputs, chunk-size independence and workspace reuse.
- [ ] Qualified reference Q target rtol=1e-6/atol=1e-12; CPU/CUDA target
      rtol=1e-8/atol=1e-12. Document scale-aware angular errors near zeros and
      independent absolute checks of weak absorption. Investigate failures,
      never automatically weaken tolerances.
- [ ] Real CUDA small/medium/large workloads, compilation/transfer/execution
      timing separated; no unmeasured speedup claims.
- [ ] Relevant numerical suites, full required checks, Ruff/mypy, examples,
      documentation/site build, clean wheel installation and offline fixture
      tests with no external Mie solver installed.
- [ ] Review final diff and evidence, push final branch, then create upstream
      PR and follow its checks/review to completion. Do not merge.

## Progress

- Worktree created from upstream main a73fa1974; unrelated development preserved.
- Earlier NVIDIA-SMI observation could not access the driver. Real CUDA remains
  a completion requirement; investigate actual device access during validation.
- CPU coefficient/efficiency foundation implemented with bounded continued
  fraction and independent small-argument Bessel series. Focused suite: 99
  passed, including all qualified reference cases through x=10000, independent
  Bessel checks of nine documented upstream discrepancies, Rayleigh limits,
  zero cases, medium normalization, broadcasting and invalid-input rejection.
  Cross-section, angular, GPU and remaining full-stack work still pending.
- Added CPU dimensional cross sections, reusable-coefficient angular amplitudes,
  sphere Mueller/phase matrices, explicit Stokes rotations and matrix action.
  Independent Jones-field and solid-angle integral checks accompany raw angular
  references. Full high-x angular/CUDA qualification remains pending.
- Added CPU population-weight helpers, bulk scalar/Mueller reductions,
  column optical depth/transmission and mass-specific opacity. Four additional
  raw PyMieScatt population cases cover discrete and continuous distributions.
- Device access verified outside the restricted execution environment:
  NVIDIA GeForce RTX 5060, 8151 MiB, driver 615.78.08. Actual Warp CUDA
  numerical/performance validation remains pending kernel implementation.
- Direct Warp float64 complex arithmetic, bounded continued-fraction Mie
  coefficients and efficiencies implemented. Real RTX 5060 plus Warp CPU:
  170 complete optics tests passed, including every saved sphere input through
  x=10000, coefficient/efficiency parity at the planned tolerances, repeated
  buffer reuse, invalid inputs and insufficient-capacity status propagation.
  Command: XDG_CACHE_HOME=/tmp/optics-cache .venv/bin/pytest
  particula/optics/tests --no-cov -q -Werror (shared existing environment).
  Public workspace validation, angular/polarization/reduction kernels, device
  tiling and full performance qualification are still pending.
- Added direct Warp angular amplitudes, scaled full Mueller matrices, basis
  rotations and Stokes application. Device coefficients are reused across
  angles, with separate per-angle failure propagation. Full current optics
  suite: 174 passed on Warp CPU and native RTX 5060. The angular pipeline
  covers 27 cases through x=10000 and exact/fine forward/backward angles;
  per-case peak-normalized CPU/CUDA angular error is bounded by 1e-8.
  Mypy and Ruff pass. GPU population/column kernels, workspace API, tiling,
  examples and independent large-x angular qualification remain pending.
- Added direct Warp number-density/PDF weights, bulk scalar and signed Mueller
  reductions, layer/column optical depth, direct transmission and mass opacity.
  Current full optics suite: 182 passed on Warp CPU and RTX 5060, with three
  population measures, unequal volumes, all bulk outputs, failure propagation
  and empty columns. Ruff/mypy pass. A local type annotation initially failed
  Warp compilation; removed it and reran the complete suite successfully.
  Explicit workspace/launch admission and tiled execution remain pending.
- Added public explicit `particula.optics.gpu.prepare_mie` and immutable
  MieWorkspace handles with retained input/output/scratch arrays and compiled
  launch commands. Case and angle tiles reuse bounded coefficient scratch;
  enqueue executes commands without host transfers, and check is the explicit
  observation boundary. Metadata/stream/capacity admission precedes preparation.
  Full optics suite: 185 passed with CPU/CUDA tile invariance, pointer reuse,
  changing input values and invalid-result observation. Ruff/mypy pass.
  Bulk-only streaming execution and convenience admission for reduction kernels
  remain pending, as do examples, documentation and full qualification.
- Added public `prepare_population_mie` streaming bulk workspaces. Particle
  chunks share coefficient/angular scratch and accumulate population optical
  coefficients, scattering-weighted g and differential Mueller matrices.
  No full-population particle optical arrays are retained. Preparation and
  observation retain the explicit device/stream contract. Tail chunks, repeated
  reuse, zero weights, changing inputs and per-group failure isolation are
  tested against CPU particle reductions. A Warp literal precision discrepancy
  was fixed using a typed double constant without weakening tolerances.
  Complete current optics suite: 190 passed on Warp CPU and native RTX 5060;
  Ruff and mypy pass. Examples, documentation, further API/failure audit,
  independent large-x angular checks and performance qualification remain.
- Added independent large-sphere angular qualification at x=100,1000,3000,
  10000 for three complex indices, endpoints and fine forward angles on NumPy,
  Warp CPU and native CUDA. AMOS Bessel ratios/Jacobi polynomials provide the
  oracle; one highly absorbing case needs a saved externally generated
  2000/2200-digit upward-recurrence reference because scaled AMOS underflows.
  No mpmath dependency was added. Generator/provenance are recorded alongside
  the reference documentation. Composite angular quadrature at two resolutions
  recovers Qsca/g through x=10000 with rtol=1e-8. A single high-order quadrature
  rule was insufficient; the acceptance tolerances were preserved.
  Full current suite: 230 passed on CPU and RTX 5060; Ruff/format/mypy pass.
  The public API/failure audit, examples, documentation, performance and broad
  repository/packaging qualification remain before the PR.
- Added named CPU perpendicular/parallel/unpolarized differential cross
  sections in m²/sr, with batch/angle contracts and explicit overflow handling.
  Added a direct Warp per-particle cross-section kernel retaining ext/sca/abs/
  back/pr in m², including upstream and invalid-value status propagation.
  This preserves individual absorption for later feedback without mutating
  particle state. Absolute units, medium scaling, buffer reuse and failures
  are covered. Full optics suite: 240 passed on CPU and native RTX 5060;
  Ruff/format/mypy pass. Remaining API/failure review, examples, public guides,
  performance qualification and broad repository/packaging checks remain.
- Added the public optics guide and a registered CPU example covering spectra,
  polarized angular scattering and cloud depth/transmission. Full refresh
  completed with three visually inspected figures, printed diagnostics and
  compact saved plot data. Physical inputs are explicitly illustrative and
  later thermal/evaporation feedback remains a separate state-changing model.
  Source documentation structure passes; the site builds successfully with
  API generation and its normal link verification. The additional rendered
  structure checker reports six retired-route collisions requiring follow-up.
  Broad changed-example validation (the registry selects all examples) and
  documentation tests are running; do not treat those as passed yet.
