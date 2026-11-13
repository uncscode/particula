# P3 coagulation rates (in progress)

Entry checkpoint: CPU/GPU condensation implementation through `086fa97a4`.
P2 untargeted package validation passed 7,540 tests with 19 skips and 92%
coverage. This does not establish P3 coverage or M3 completion.

## Executable support target

- Native kernel laws: Brownian; charged hard-sphere, Dyachkov 2007, Gatti 2008,
  Gopalakrishnan 2012 and Chahl 2019; SP2016 sedimentation; ST1956 shear;
  AO2008 turbulent DNS; existing compatible sums. Kernel units are m^3/s and
  depend on per-particle material geometry/mass/charge, not population counts.
- Representation metadata must match the process on every native entry.
  CPU requires one box. Resolved volume is positive and counts normalize as
  N/V; PMF/PDF remain density at V=1. PDF rate quadrature uses radius_grid,
  while kernel geometry remains mass/density derived.
- Rate APIs retain existing discrete/continuous-PDF support. Resolved kernel
  evaluation does not invent resolved gain/loss rates; P4 owns selection and
  merges. Gain interpolation retains its existing cubic algorithm and needs
  at least four increasing positive nodes for nonzero populations; zero-work
  rates and empty PMF populations are handled separately.
- Zero-mass, zero-count slots retain zero interaction rows/columns. Occupied
  zero-mass particles reject as lacking physical coagulation geometry. A
  private material-only kernel view is not a new population or authoritative
  copy. No facade is constructed; legacy untagged callers remain separate.
- Preserve existing arithmetic mean of mass-weighted per-particle effective density for sedimentation
  and DNS and DNS's existing ambient-pressure convention. Do not silently
  replace physical algorithms while migrating native state.

## Implementation and focused evidence

Native admission and full-layout kernel evaluation now wrap the six concrete
strategy families. Validation checks representation, physical derived values,
scalar environment inputs and current mechanism controls. It rejects cyclic or
incompatible combined configurations and invalid charged-model replacement.
Private material views evaluate the existing laws only on positive-mass lanes;
zero-mass, zero-count slots receive zero matrix rows/columns. This keeps original
slot indices without changing algorithms or constructing facades.

Rate methods validate supplied pair matrices and use the explicit PDF radius
coordinate for quadrature. PMF retains its existing physical-radius conversion
to PDF for gain interpolation. Shared density helpers distinguish resolved N/V
from stored PMF/PDF density. Rate docs now state number-density/PDF-density
units, and friction-factor docs state kg/s.

`native_rates_test.py` contains 253 tests. Brownian, sedimentation, shear and
combined simple-law tests use independent equations. All five charged models
and DNS compare independently derived per-particle inputs with existing
low-level laws, alongside their retained lower-level scientific regressions;
this is not an independent rederivation of those complex laws. Constant-kernel,
constant-PDF fixtures independently establish gain/loss quadrature and units.
The matrix includes PMF/PDF and resolved V=.25/1/4, unequal weights, zero-mass
slots, empty layouts, malformed/mutated state, configuration and pair kernels,
resolved-rate rejection, and the existing cubic interpolation support boundary.

2026-11-11 focused checks against entry `086fa97a4` plus the pending P3 diff:

- `.venv/bin/pytest particula/dynamics/coagulation/ -q --no-cov --tb=short`:
  479 passed in 1.00 s. `.artifacts/e9-m3/p3/coagulation-tests.log`.
- `PATH="$PWD/.venv/bin:$PATH" .opencode/tools/run_linters.py`: repository
  Ruff/mypy passed. `.artifacts/e9-m3/p3/linters.log`.
- `PATH="$PWD/.venv/bin:$PATH" .opencode/tools/run_pytest.py`: untargeted
  package gate passed 7,793 tests with 19 skips in 143.49 s; 92% coverage against
  the unchanged 80% threshold. `.artifacts/e9-m3/p3/package-tests.log`.
- `.venv/bin/mkdocs build --strict`: passed in 1.46 s.
  `.artifacts/e9-m3/p3/docs.log`.
- The final decorator annotation preserves the original method signature for
  static type checking; focused checks also pass after that typing-only change.

P3 implementation and validation checkpoint are complete on the branch.
Single-PR review, final-revision validation and M3 completion remain pending.

## P4 audit items

The current resolved selector does not receive stored particle multiplicities.
Its merge retains recipient concentration, adds donor per-particle material
mass/charge and clears donor concentration. Equal positive active weights can
conserve weighted mass/charge; unequal active weights cannot. P4 must validate
that support before RNG use and include common weight in collision frequency,
while keeping the physical volume and caller's raw state explicit.

Current temporary kernel binning uses arithmetic material means, summed raw
concentration and median charge, but discards a bin if *any* material lane is
zero. P4 must retain valid zero-material compositions, distinguish temporary
kernel state from authoritative population state, account for active slots and
volume, and test supported charge semantics. P4 also owns staged native
PMF/PDF updates, finite timesteps, mutation boundaries and discretization-
appropriate species inventory evidence. No stepping completion is claimed here.
