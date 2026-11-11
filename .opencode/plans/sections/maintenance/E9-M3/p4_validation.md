# P4 coagulation stepping — in progress

Entry: P3 committed as `b41c908d1`. CPU work below is a validated
implementation checkpoint, not P4 completion. Direct/prepared GPU admission
and normalization remain pending. M3 is not complete and M4 is not opened.

## Native CPU changes

- Tagged `ParticleData` dispatches to a private staged step. Legacy untagged
  containers and facade execution retain their existing paths.
- Admission revalidates physical state, representation, one-box support,
  mutable mechanism controls, finite environment/time values, writable
  destinations and protected-field aliasing. Resolved bin controls and RNG
  ownership are validated before draws, including zero-work calls.
- PMF/PDF retain the existing gain/loss laws and explicit Euler update in
  density storage at V=1. Nonfinite or negative proposals reject without
  clipping or caller writes. This does not establish a conservative species
  transport algorithm; the scientific decision below is still required.
- Resolved steps accept equal positive active weights, including fractional
  weights, and reject unequal weights before RNG use. The existing selector
  receives `time_step * common_weight` and the original physical volume, so
  its scheduling factor is `K * weight * dt / V`. Inactive positive-mass
  slots are excluded from selection, while their original state is retained.
- Temporary kernel representatives use count-weighted material means and
  summed counts at the original volume. Zero material lanes are allowed.
  Existing median-charge kernel approximation is retained; authoritative
  charge bookkeeping uses original donor and recipient charges. Explicit
  edges may not silently discard active particles above their upper edge.
- The existing selector and its direct charged callbacks remain in use.
  A single-point interpolation value is defined only at its tabulated radius;
  off-grid values remain zero. This avoids SciPy's NaN for a 1x1 linear grid.
  Identical material vectors retain their exact value when binned, avoiding
  a rounded mean that moves the representative one ulp away from its slots.
- Selector self pairs are no-ops, including final roots produced by chain
  resolution. Other output must have in-bounds integer indices, unique donors
  and resolved donor/recipient sets. Repeated recipients use accumulation.
  Mass, counts and charge are staged and validated before in-place publication.
  No RNG rollback is promised after selection has consumed draws.

`native_coagulation_step_test.py` currently has 248 passing tests: all ten CPU mechanism
rows, five direct charged models, resolved volumes .25/1/4 and common weights
.25/1/4, independent species/charge inventories, inactive slots, empty work,
mutable input/control rejection, protected aliases, zero-material binning,
same-backend normalized-hazard replay, actual selector execution and uniform-
composition quadrature refinement. This is not cross-backend seeded parity.

## Scientific decision pending: fixed-grid species conservation

The existing PMF/PDF number-only update cannot conserve arbitrary
size-dependent species composition. The masses/composition assigned to each
radius node stay fixed while gain is added at the destination radius; the
destination composition generally differs from the combined donor composition.
This is distinct from radius-grid integration error.

An independent inventory audit used two equal-density materials (1000 kg/m3),
constant kernel 1e-15 m3/s, a lognormal number distribution with total 1e10 m-3,
geometric mean radius 1e-7 m and log standard deviation .35, on radii
1e-9 through 1e-6 m. Per-particle total mass is `4*pi*rho*r**3/3`.
The first mass fraction is either constant .4 or
`clip(.5 + .3*log(r/1e-7), .05, .95)`; the second is its complement.

For each species the table reports signed integrated mass-rate defect divided
by its gross loss mass rate. These ratios do not become small merely by
choosing a small timestep. PDF uses independent trapezoid integration of
mass times the rate on the explicit radius grid. PMF uses bin sums and the
existing forward-radius-width conversion of that PDF.

| Representation | Nodes | Constant fraction (both species) | Varying fraction, species 1 | Varying fraction, species 2 |
|---|---:|---:|---:|---:|
| PDF | 128 | -0.000477454 | +0.0773959 | -0.122399 |
| PDF | 256 | -0.000116687 | +0.0777885 | -0.122088 |
| PDF | 512 | -0.0000289361 | +0.0778835 | -0.122011 |
| PDF | 1024 | -0.00000721221 | +0.0779070 | -0.121992 |
| PMF | 128 | -0.0269343 | +0.0488777 | -0.145628 |
| PMF | 256 | -0.0134779 | +0.0633862 | -0.133819 |
| PMF | 512 | -0.00674232 | +0.0706471 | -0.127906 |
| PMF | 1024 | -0.00337204 | +0.0742800 | -0.124947 |

The uniform-composition test checks first-order PMF and second-order PDF
refinement against gross transfer. It expressly does not claim arbitrary
species conservation. Total-mass error also converges for the varying-fraction
case, while individual species errors persist.

On 2026-11-11 the maintainer was asked to choose between retaining the method
with an explicit scientific limitation and expanding M3 to conservative species
transport. No answer is recorded yet. Neither narrowing support to uniform
composition nor silently replacing the solver is authorized by this evidence.
The existing objective's per-species conservation requirement remains unmet
for this row until the decision and its implementation are complete.

## GPU audit for the next implementation unit

The GPU selector currently multiplies its schedule by common concentration
only for sedimentation-containing masks. Native resolved admission must require
equal positive active counts for every supported mask and apply the common
count factor to every native schedule. Preserve the separately tested legacy
path and the supported masks 1/2/4/8/3/5/6/9/10/12/15. Prepared enqueue must
retain frozen device references and avoid new validation/readback/allocation.
Native primary arrays, volume authority, output/RNG aliasing and failures must
be validated before any caller writer. Interim tagged-state rejection tests
must be replaced by numerical support tests, not simply removed.

The existing `_validate_sedimentation_particle_physics_kernel` already scans
mass/count/density/charge, checks equal positive counts and active material
geometry, and accepts an explicit active-count bound. Reuse its verified
invariants without imposing the sedimentation cap on other mechanisms that
currently have different bounds. Native volume/count-density validation and
writer/protected-field overlap checks still need coverage. A separate native
selector wrapper around shared kernel implementation can carry the scheduling
tag without changing the existing low-level kernel signature or untagged
legacy behavior. Do not implement common-weight scheduling by a potentially
underflowing `V/weight` effective-volume allocation, or freeze mutable counts
as host-derived scalar state in a prepared call.

## Validation

- Focused new CPU module before the last monodisperse expansion: 237 passed
  in 0.59 s; final whole-directory run below includes the additional 11 cases.
- Focused Ruff and mypy (19 source files): passed.
- Whole coagulation directory after the monodisperse fix: 727 passed in 1.68 s.
- Repository lint/type runner: passed.
- First untargeted run found a test-basename collision with condensation.
  Renamed the new module to `native_coagulation_step_test.py`; its rerun passed.
  That run preceded the monodisperse rounding fix.
- Final untargeted `.opencode/tools/run_pytest.py` against `b41c908d1` plus
  this CPU P4 diff: 8,041 passed, 19 skipped in 147.90 s; 92% coverage with
  unchanged 80% threshold. Log:
  `.artifacts/e9-m3/p4/package-tests-final.log`. This includes installed-Warp
  CPU and available CUDA package regressions, not native GPU P4 evidence.
- Strict documentation build: passed in 1.43 s.

This checkpoint does not satisfy P4/P6 or final-revision validation.
