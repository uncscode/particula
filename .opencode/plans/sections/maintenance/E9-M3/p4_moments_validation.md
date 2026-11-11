# P4 CPU density-moment correction

Entry revision: `daf119778` plus the previously uncommitted `_native_step.py`
species-moment implementation. This follow-up remains part of the single M3
PR. It closes the PMF/PDF composition defect diagnosed in
[p4_validation.md](p4_validation.md), subject to the recorded checks below and
final scientific review. M3 completion and M4 admission are not claimed.

## Implemented behavior

- Preserve the existing explicit number-density update and its kernel laws,
  interpolation, integration and PMF/PDF conversion. No global rescaling,
  clipping, new collision law or new representation is introduced.
- Transport each species-mass density and signed-charge density through the
  same gain quadrature, carrying the sum of both donors' per-particle quantities.
  Loss removes each departing particle's quantity. Recover per-particle means
  from the staged updated number density rather than assigning destination
  bins' old composition to gained particles.
- Stage every species, charge and count update before publication. Reject
  nonfinite/negative material inventory, nonzero inventory at zero number
  density, arithmetic overflow, invalid loss output, protected aliases and
  unwritable destinations. Failure during a later species leaves all caller
  arrays unchanged. Zero-inventory empty nodes retain their material carrier;
  newly occupied PDF nodes receive both donors' material and signed charge.
- Preserve container and array identity, material density, volume and the
  explicit PDF radius grid. Zero time, population or kernel is byte-preserving.
  Resolved and untagged legacy algorithms remain unchanged by this correction.

## Independent scientific evidence

`native_coagulation_moments_test.py` contains 49 cases:

- An analytic constant-kernel/constant-radius-PDF collision integrand with
  size-dependent species composition and sign-changing charge. SciPy `quad`
  integrates analytic donor quantities independently of the production spline
  and gain helpers. The comparison recovers gain from the actual finite-step
  inventory change, testing PMF and PDF separately at six destination sizes.
  Its `rtol=8e-5` bounds the retained finite trapezoid/spline error; it is not a
  global conservation claim for that finite-boundary fixture.
- Independent before/after species and charge inventories for lognormal
  populations with varying composition, both equal and different material
  densities, and 128/256/512 radius nodes. Defects are normalized by timestep
  times gross absolute loss, not the original inventory. Both species and
  signed charge converge: successive errors fall below factors .55 for PMF
  and .3 for PDF, and the final normalized errors are below .007 and 3e-5,
  respectively. This establishes the retained grid-dependent accuracy rather
  than claiming exact conservation or a new order.
- Every one of the ten CPU mechanism rows retains the original number update
  and caller-owned array identities for both representations.
- Exact absent-species/zero-charge and no-op cases; an empty PDF destination
  independently receives twice the uniform donors' mass and charge; read-only
  and alias rejection; late negative, overflowing and orphaned material
  proposals preserve every caller array.

The older `test_density_step_equals_existing_rate_times_duration` now checks
the unchanged number law, identities and protected density/volume. Its former
mass/charge-unchanged assertions are superseded by the independent moment
tests, not discarded without a successor. Existing uniform-composition
number-quadrature refinement remains a separate retained regression.

## Validation

- New moment module: 49 passed in 1.73 s.
- Whole coagulation directory before the final empty-destination test:
  775 passed in 3.36 s; the additional test passed in the focused run above.
- Repository lint/type runner: passed; final added-test Ruff check and format
  checks passed. Focused production mypy also passed.
- Strict documentation build: passed in 1.37 s.
- Untargeted package runner: 8,418 passed, 19 skipped in 151.83 s, 92%
  coverage with unchanged 80% threshold. Log at
  `.artifacts/e9-m3/p4-moments/package-tests.log`.

The full runner began before the final empty-destination test was added. Its
result describes the production correction and the first 48 new tests;
the 49th has separate focused evidence above. P6 must run all final-revision
gates after wall-loss and integration work, retaining the normal threshold.
