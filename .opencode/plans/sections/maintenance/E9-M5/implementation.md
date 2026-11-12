# M5 single-PR implementation record

## Accepted delivery — 2026-11-11

The maintainer confirmed M4 completion and requested implementation of M5,
followed by one PR. Base: `3393e47fa` (merged upstream PR #1633).
Branch: `codex/e9-m5-native-examples-docs`; target: `uncscode/particula:main`;
issue: #1609. This supersedes per-phase issue generation and separate PRs.

## Serial checkpoints

1. Scope and exhaustive inventory (in progress).
2. P1 foundational tutorials with runtime fixtures and executed pairs.
3. P2 condensation/coagulation and repository helper example.
4. P3 wall loss, dilution and nucleation.
5. P4 composed simulations, customization, Activity and Equilibria.
6. P5 direct-Warp and resident examples.
7. P6 migration, replacement guidance and public quick starts.
8. P7 contributor guidance and final consumer audit.
9. Final validation and M6 handoff.

Each example-changing checkpoint includes its tests and executed notebook
outputs. Preserve existing paths and scientific content. No production physics,
new API, facade workaround, legacy deletion, release or performance claim.
Native capability gaps block the owning migration rather than justify reducing
its scientific workload. M6 acceptance and PR merge remain separate decisions.

## Inventory and evidence

`example_inventory.json` enumerates all 58 Python sources and 46 paired
notebooks at the base. Pending rows are not migration or execution evidence.
Expand the ledger with published snippets, links, quantities, tolerances,
commands and final-revision results during the owning checkpoint. Audit the
whole tree again at closeout; absence of legacy spellings is insufficient.

Use the repository `.venv`, existing notebook sync/execution tools, and current
`examples_tests/` runtime suites. Old plan references to removed documentation
tests do not authorize recreating prose-only assertions. All published workloads
must execute; timeout skips remain incomplete evidence. Required final gates:
untargeted repository coverage, examples and infrastructure suites, both explicit
CPU wall-loss suites, GPU/resident boundary checks, lint/type checks, payload
consistency and strict MkDocs. Preserve thresholds and scientific tolerances.

## P1 partial validation — 2026-11-11

- Aerosol, gas species and gas/environment tutorials migrated; each standalone
  source and synchronized notebook executed successfully. Three numerical
  fixtures pass in `examples_tests/native_foundations_example_test.py`.
- Canonical data-container example now stores explicit resolved counts at
  physical volume, with the guide forwarding to it. Both entrypoints execute;
  `WARP_CACHE_PATH=/tmp/particula-m5-warp .venv/bin/pytest
  examples_tests/data_containers_example_test.py -q --no-cov`: 14 passed.
  Enabled transfer failures propagate instead of claiming CPU-only success.
- Notebook tools require `PYTHONPATH=/home/kyle/Code/Agent` for the existing
  ADW utility dependency. Execution uses repository `.venv` on PATH and local
  Jupyter kernel access. Initial restricted-kernel failures were rerun with
  access; no failed attempt is counted as a successful run.
- Gas-species coefficients and their convention are retained as illustrative
  inputs, not requalified scientific data. The altitude tutorial evaluates
  only its published water diagnostic over the cold range, retaining all gas
  lanes without extrapolating unqualified organic coefficients.

P1 remains incomplete. Particle tutorials, remaining functional foundations,
landing-page reconciliation, plot inspection and strict rendering are pending.
These working-tree results are provisional until the P1 commit and closeout.

### Foundation checkpoint validation

All ten P1 notebooks passed source-first sync followed by full execution;
all ten retain output cells. Three unchanged Python sources were restored after
bidirectional sync exposed stale notebook text; execution was then repeated.
Do not sync output-bearing notebooks back through stale sources after execution.
The six native-foundation assertions and fourteen transfer assertions pass
(20 total). Strict MkDocs passed. Plot montages were inspected; radius-PDF
coordinate conversion now includes the explicit 2e6 Jacobian and an independent
population/material-integral regression. Final P1 ledger and snippet audit
remain pending; this checkpoint does not complete M5.

### P1 execution and inventory audit

All twelve P1 Python entrypoints executed directly with repository `.venv`,
`MPLBACKEND=Agg`, `MPLCONFIGDIR=/tmp/particula-m5-mpl`, and
`WARP_CACHE_PATH=/tmp/particula-m5-warp`; every exit code was zero. All ten
paired notebooks also executed and retain outputs. Functional activity,
surface and vapor-pressure examples retain their independent strategies;
no executable facade imports or conversion bridges remain in P1 sources.
Topic-index paths remain stable; discovery-based MkDocs navigation includes
these pages. The inventory assigns later published snippet migration to P6/P7.
P1 implementation gate is passed at the foundation checkpoint plus this
record. Final M5-wide validation and acceptance remain pending.

### P2 condensation checkpoint

All five condensation source/notebook pairs now use native data and explicit
process strategies/species maps. Full original workloads execute successfully;
saved outputs are retained. Closed resolved cases assert species inventory
conservation at rtol=1e-12 and atol=1e-30. The fixed-gas mass-bin experiment
remains an open reservoir, explicitly distinguished from conservation tests.
Native latent energy (J) is divided by represented volume for J/m³ reporting.
Eleven focused tests pass. Plot inspection supports finite trajectories and
roundoff-level conservation, not a universal stability ranking. P2 coagulation
and helper starter migration remain pending; this is not the P2 completion gate.

### P2 coagulation and helper checkpoint

All eleven coagulation pairs execute at their original published workload sizes,
including four smaller examples under Functional and both charged-kernel routes.
All 16 plots inspected. Resolved runs conserve independently weighted inventory
at rtol=1e-12, atol=1e-30; native binned diagnostics match independent sums or
radius quadrature. The old preset builder regenerated default bimodal inputs,
overwriting the declared concentration array. Native initialization now honors
the declared modes/numbers; the correction is explicit in each affected example.
For the two-hour 80 nm pattern cases, material mass drifts are -0.00225778 (PMF)
and -0.0000324505 (PDF); no conservation tolerance was relaxed or exact
conservation claimed. The 50,000-particle time comparison retains 12 calls of
600 s with 50 substeps. Charged kernels have 1,281 exact-zero entries from the
broad input range; final arrays are finite, nonnegative and symmetric. Plots mask
zeros and retain/document intermediate underflow warnings.

The helper starter and doctor now require the native checkout, with no facade
fallback. Native default condensation and coagulation pass their existing 1e-6
mass tolerance (errors 7.9e-16 total and 2.5e-9 particle, respectively). Historical
failed high-loading evidence is retained. Installation instructions distinguish
the native checkout from old released packages. OpenCode payload regenerated.

P2 gate: 64 focused native/example scientific tests and eight plugin distribution
tests passed; generated payload check and diff whitespace check passed.

### P3 implementation and validation

CPU dilution now uses unified gas with three ordered species and native resolved
counts [4,8] in 2 m³, retaining reported densities [2,4]. Identity and protected
mass/charge/volume assertions supplement exact exponential decay. The supported
CPU nucleation script was already native after M4 and remains unchanged.

Wall-loss strategy tutorials retain the original 250-node grid and full 6000 s
(spherical) / 1800 s (rectangular) horizons. Native Euler rejects negative updates,
so internal substeps use ceil(max(k)*output_dt/0.9), respectively 2 and 9.
Independent repeated-Euler expressions check final densities at rtol=1e-12,
atol=1e-30; fixed masses remain identical. Rate labels are m^-3 s^-1, not 1/s.
The combined tutorial uses 150 nodes over 10 nm–10 µm: the original 1 nm–100 µm
coagulation tails produced negative interpolated gains after depletion. The
native admission failure was not bypassed. The formerly skipped condensation
snippet now uses an explicit low-volatility organic fixed reservoir (1e-8 kg/m³,
1e-5 Pa, molar mass 0.12), with full ordered gas configuration. This is an
illustrative input correction, not preservation of the old placeholder physics.
The independent chamber functional example keeps all 1000 times/250 nodes and
asserts finite nonnegative distributions and monotonic total-number loss.

Custom nucleation retains the saturation-dependent law, initial 100-particle
400 nm-radius sample, 0.1 cm³ volume, five stored times and two process substeps.
It removes facade monkeypatches, uses 100,000 fixed slots with unit weights,
and debits gas by newly represented mass divided by volume. Capacity rejects
before gas debit. The original debit omitted the volume factor. Full runs
activate 2,281 then 10,481 nuclei for the seeded initial sample; stochastic
coagulation is assessed through inventory, not exact particle identities.
Independent extensive inventory includes the external vapor source at every
stored time (rtol=1e-12, atol=1e-30). Plot bounds now include the initial 400 nm
mode and 2 nm-radius nuclei; prior 100 nm upper bounds hid the initial population.
A focused fixture checks two nuclei at non-unit volume and capacity failure.

P3 gate: 507 focused tests passed with warnings as errors, including both CPU
wall-loss strategy suites. All six notebook pairs executed without errors; 12
plots inspected. Dilution and supported CPU nucleation standalone runs passed.


### P4 migration and scientific audit (execution in progress)

The six simulation sources now use native ordered gas/environment/particle
containers and explicit condensation species maps. Preserved workloads include
10,000 biomass particles with four 3,600 s/10,000-substep scenarios; the organic
20 s/100,000-substep pulse followed by 60 coagulation steps; 1,000 cough droplets;
5,000 soot particles, 24 species and 100 condensation substeps per update; and
both original cloud-cycle workloads. None is replaced by its reduced fixture.

Corrections found while auditing published quantities:
- Organic process molar masses now use kg/mol rather than accidentally passing
  g/mol. The later 600 s stage runs coagulation alone, as in the source; it
  reports binned inventory drift without claiming exact conservation.
- Biomass DNS uses its DNS kernel alone, not a separate added Brownian kernel.
  Sedimentation is a collision mechanism, not a wall-removal process. Its
  initial particles already contain water. Radius/duration labels are corrected.
- Cough and soot retain their update counts, with time axes reporting actual
  0..19.9 s and 0..19.96 s rather than the original inclusive-linspace labels.
  Soot retains the additive cooling formula: its starting temperature is 1600 K,
  not the incorrectly described 1200 K. The existing 1500 K table endpoint
  clamp and illustrative forcing are now explicit; no new combustion fit.
- Cloud histories retain prescribed humidity resets (external water sources)
  and condensation followed by wall loss. Multi-cycle dilution was disabled;
  the text now says so. Its separate dilution exercise decays represented
  counts, not per-particle masses. Dry-material retention and removed original
  particle identities replace misleading wet-size-bin subtraction. Peak water
  fractions now select every humid-to-dry boundary, not an incorrect stride.
- Adding particles reserves 120,000 slots, starts 100,000 unit-count particles,
  and activates two 10,000-particle external sources at 1 cm³. It checks injected
  mass, held arrays, gas preservation and exhaustion rejection without resizing.
- Activity/Equilibria remain independent functional/strategy examples. Printed
  kg/m³-to-microgram conversions were mislabeled nanograms and are corrected.
  Equilibrium phase species entries include associated water; organic inventory
  checks subtract it. Nonideal activity is not capped at one in the plot.

`native_simulation_examples_test.py` and
`native_activity_equilibria_examples_test.py` currently pass 17 focused checks:
all six setup families, volume-aware per-species inventories, held storage,
source/capacity behavior, cooled soot process order, all three short cloud
compositions and forcing, independent dry-loss/phase-peak diagnostics, analytical
ideal/kappa relations and bounded equilibrium outputs. Full standalone biomass,
organic, cough, both clouds, injection and all seven activity/equilibrium scripts
have passed (cloud multi has later reporting edits awaiting full notebook).
The seven functional notebooks have executed; eleven functional plot outputs
were visually inspected, including the refreshed activity axis. All seven
functional pairs match their sources and retain successful execution counts.
Full final-revision simulation notebook execution and plot inspection remain
pending. P4 is not complete at this checkpoint.


P4 follow-up: committed the functional activity/equilibrium audit as `f41e7a066`.
Cloud/cough full notebooks passed and 14 plots were inspected. The retained
cloud dry-phase trajectories show step-scale oscillations; both tutorials now
explicitly distinguish material conservation from timestep/trajectory accuracy.
The multi-cycle activation plot uses a 2 µm threshold, correcting its old 1 µm
label; the single-cycle dry humidity is 85%, correcting its old 65% description.
Formatting-only notebook updates retain outputs only after per-cell AST equality
checks. Any calculation or plot-label change receives fresh execution. Biomass
and soot full final publication runs remain active. Seeded initial sampling is
not claimed to seed the separate stochastic coagulation generators.


The organic notebook's complete retained coagulation stage reports per-species
relative inventory defects from -0.5866 to +0.3493. This is the size-dependent
composition limitation explicitly accepted in M3 (`p4_preserved_method.md`:
"Preserve method; document limitation"), not a new native capability gap or a
relaxed conservation tolerance. Its leading warning now labels this stage a
limitation demonstration, directs composition-sensitive work to the supported
resolved example, and distinguishes the conserved condensation pulse. No claim
of quantitative binned species forecasting remains. The original workload is
retained. The governing fixed-composition regression suite is included below
with P4 validation; its existing method and bounds remain unchanged.


### P4 completed publication gate

All fourteen P4 source/notebook pairs have completed full execution and retain
outputs; final source/cell identity and nonempty-cell execution counts were
verified. All forty plot outputs were inspected. The final biomass notebook
completed in 642 s and the full soot notebook in 1,456 s; no timeout or reduced
workload substitutes for those runs. Final organic plots include quantity units
and the source's first section carries the accepted PMF composition limitation.
Formatting/prose-only updates were checked for per-cell AST identity before
retaining outputs; changed plot labels received fresh full execution.

Validation: 17 example regressions plus 26 unchanged preserved-coagulation
method tests passed (43 total). Scoped Ruff checks/format and `git diff --check`
passed. Strict MkDocs passed; existing informational link messages remain for
the P6/P7 navigation audit. The fourteen standalone entrypoints also executed
with the full migrated workloads. Chemical-property assumptions and extrapolated
high-temperature tables remain illustrative, not independently requalified.
P4 is complete; P5–P7 and final repository-wide gates remain pending. No PR,
push, merge, release, M6 deletion or performance qualification is claimed here.


### P5 native boundary migration started (2026-11-11)

Direct nucleation now tags particle-resolved counts and explicitly prepares a
NucleationSpeciesMap binding for the exact transferred carriers and full gas
order. Its independent inventory sums particle mass times counts plus gas mass
density times represented volume. The regression executes both one- and two-m³
cases, checks activated slots and binding ownership, and observes explicit
transfer/preparation/dispatch/synchronization/restore order.

The resident lifecycle example now carries explicit resolved metadata for one
particle in 1e-6 m³. Restart inspection checks metadata, raw count, mass and
volume preservation. The printed checkpoint compatibility version is corrected
to schema-v5; this zero-duration example still performs no physical process.

Validation on the current working changes with Warp 1.11.1 and
WARP_CACHE_PATH=/tmp/particula-m5-warp:
- `.venv/bin/pytest examples_tests/gpu_direct_nucleation_example_test.py -q
  --no-cov`: 5 passed, including the standalone documented command.
- `.venv/bin/pytest examples_tests/gpu_resident_session_example_test.py -q
  --no-cov`: 11 passed.
- `.venv/bin/python docs/Examples/gpu_resident_session.py`: exit 0, explicit
  Warp CPU lifecycle completed. Host reports no CUDA-capable device.
- Scoped Ruff check and format check for the four changed Python files passed.

P5 remains in progress: the other six direct/resident sources, owning tests and
complete phase gate are pending. No final full-suite evidence or PR yet.


### P5 direct coagulation, condensation and five-process sequence

The direct coagulation fixture now tags resolved counts, retains the six
unit-count active slots and deliberately inactive material, and verifies
extensive mass conservation. Those inactive slots are explicitly not advertised
as canonical free slots for activation. Final restoration uses sync=False
after the existing explicit synchronization.

The condensation quick start preserves its original number density (1 m^-3
per slot) by storing 1e-6 counts at 1e-6 m³. An explicit concrete species binding
is prepared after transfer and reused across both calls. The energy regression
now checks count-weighted extensive J with atol=1e-30; gas coupling retains its
original density-space tolerance after dividing particle count-weighted change
by volume. Disabled paths retain lazy kernel imports.

The complete sequence retains all five original calls and time steps. Native
coagulation rejected the former unequal 2/5 active counts, so the illustrative
fixture now uses equal 2/2 counts, documented at construction. This changes the
fixture inventory intentionally rather than bypassing native admission; it is
not a calibrated population or a promise of legacy numerical parity. Both
condensation and nucleation mappings are prepared explicitly for the exact
transferred carriers. Test-only observations check species inventory at each
stage, exponential dilution, bounded wall removal, and zero-time nucleation
identity. The published example retains one final synchronization/restore with
no intermediate inspection.

Validation: five migrated P5 suites (direct nucleation, resident lifecycle,
direct coagulation, direct condensation quick start, complete process sequence)
run together via `.venv/bin/pytest <five owning example files> -q --no-cov`:
61 passed in 11.92s. The three newly changed standalone sources each exited 0
under `.venv/bin/python`, Warp 1.11.1 CPU, using
WARP_CACHE_PATH=/tmp/particula-m5-warp. Scoped Ruff and format checks passed.
Host CUDA discovery reports no CUDA-capable device; no CUDA evidence claimed.
The parity walkthrough and resident multi-timestep/capture examples remain
pending, as do the complete P5 gate, P6/P7 and final repository-wide gates.


### P5 completed native example gate (2026-11-11)

All eight P5 standalone sources now carry explicit native population semantics.
The remaining parity walkthrough retains its independently ordered four NumPy
substeps and density-space mass-transfer physics. Builders store resolved counts
as density times volume; limiting and gas updates divide counts by volume.
Oracle and device energy are extensive count-weighted joules. The added volume
scaling case verifies unchanged mass/gas trajectories and proportional energy;
existing physics and conservation tolerances remain unchanged. Its standalone
command reports physics, conservation and energy passed.

The resident multi-step fixture preserves total box number densities 1.5/3/4.5
m^-3 while equalizing within-box counts to satisfy native coagulation. This
intentionally changes the illustrative per-slot weighting; it does not promise
legacy trajectory parity. Wall loss now explicitly uses resolved stochastic
survival matching the tag, replacing the former untagged density-decay config.
All twelve nodes and source/restart dispatch counts are retained. Source setup
and explicit restart each prepare new exact-carrier species bindings, with
tests asserting ownership and independent identities. Native total-mass
diagnostics carry an explicit species map. Initial and restart inventories use
particle mass times counts plus gas density times volume. Prescribed gas and
wall loss make this an open system: zero-ledger residuals are independently
checked as net inventory change, not falsely asserted to be zero.

The capture example shares that native request and fixes its extensive baseline.
Its standalone unavailable-CUDA path exited 0 without fixture/upload/capture or
CPU fallback; native capture remains unexecuted and performance unmeasured.
The subprocess stderr check now narrowly accepts both known Warp no-driver and
no-device diagnostics, retaining rejection of unrelated stderr and -Werror.

Final P5 focused command (Warp 1.11.1 CPU):

```bash
WARP_CACHE_PATH=/tmp/particula-m5-warp .venv/bin/pytest \
  examples_tests/gpu_direct_kernels_example_test.py \
  examples_tests/gpu_coagulation_direct_example_test.py \
  examples_tests/gpu_complete_process_sequence_example_test.py \
  examples_tests/gpu_condensation_parity_walkthrough_test.py \
  examples_tests/gpu_direct_nucleation_example_test.py \
  examples_tests/gpu_resident_session_example_test.py \
  examples_tests/gpu_resident_multi_timestep_docs_test.py \
  examples_tests/gpu_resident_graph_capture_docs_test.py \
  examples_tests/data_containers_example_test.py \
  particula/gpu/tests/conversion_test.py \
  particula/gpu/tests/kernel_exports_test.py -q --no-cov
```

Outcome: **265 passed, 2 skipped in 29.46s**. Skips are optional native CUDA
parity/capture. An initial invocation used the pre-move data-container test path
and collected no tests; the command above uses its actual examples_tests owner.
All eight standalone sources were executed at their migrated revision (capture
reported unavailable as above). Scoped Ruff checks/format and git diff --check
passed. `.venv/bin/mkdocs build --strict` exited 0, built in 1.58s; informational
link messages remain for P6/P7 reconciliation (`/tmp/m5-p5-mkdocs.log`).
P5 is complete. P6, P7, full final-revision gates and the PR remain outstanding.


### P6 native access and migration guide checkpoint (2026-11-11)

Added docs/Examples/native_state_access.py with single-box resolved counts at
0.25 m³, two material lanes and complete inert/organic/water gas provenance.
It demonstrates held raw-array identity, detached density properties, individual
container replacement, coordinated gas/environment width expansion, explicit
new process configuration and independent mapped condensation conservation.
Eight behavioral regressions check numerical units, copy isolation, derived
property detachment, unchanged old/candidate data on individual/coordinated
rejection, old process rejection after layout change, direct mutation validation,
and real material uptake without inert-gas changes. Three more tests execute
the exact Python snippets in construction, containers and dynamics Markdown.

Rewrote all five particle-data-migration pages around the native target. Removed
executable facade wrapping/bridge guidance, kept historical names in a clearly
nonexecutable terminology table, corrected raw count versus density mappings,
and replaced stale claims that tagged GPU uploads are rejected. Current native
transfer/species-binding/schema-v5 restart boundaries and open-system resident
diagnostics are stated explicitly. Construction retains its useful initializer,
unit and sampling details. This does not remove legacy production APIs or claim
a published release.

Validation: `.venv/bin/python docs/Examples/native_state_access.py` exited 0;
`.venv/bin/pytest examples_tests/native_state_access_example_test.py -q --no-cov`
reported 11 passed in 1.53s. Scoped Ruff and format checks passed. Initial strict
rendering caught two nonexistent example-index links; replaced with existing
process guides. `.venv/bin/mkdocs build --strict` then exited 0, built in 1.58s
(`/tmp/m5-p6-mkdocs.log`). Existing informational link audit remains for P6/P7.

P6 remains in progress: broader foundations/process/API snippets, public
quick-start integration fixture, navigation and inventory reconciliation are
still pending. P7 and final full-repository gates remain outstanding.

### P6 public quick starts and foundations checkpoint (2026-11-11)

README now publishes a self-contained native resolved condensation/coagulation
pipeline with explicit counts, 1e-6 m³ volume, process-owned physics and ordered
species mapping. The documentation home quick start retains its original
250-node lognormal PMF, 100 nm mode, GSD 1.2, 1e8 m⁻³ population, density,
vapor concentration/pressure and single 10-second step. Its exaggerated
100 kg/m³ vapor concentration is explicitly illustrative, not atmospheric
calibration or accuracy evidence. Both pages identify the selected E9 checkout
rather than claiming older wheels contain these APIs. README simulation links
now expose the recorded organic-composition and cloud oscillation limitations.
The integration test executes each exact published snippet and independently
checks native interpretation, number where applicable, uptake and total mass.

The foundations guide now states native discrete/resolved Warp metadata
preservation, PDF rejection, actual aggregate access/replacement semantics,
explicit species binding and fresh restart binding ownership. It replaces
stale M1/M2 handoff language and unweighted resolved latent-energy notation.
Its concrete construction snippet declares resolved counts; a regression
executes that exact snippet and checks number density and extensive particle
and gas inventories at non-unit volume. Broader process snippets and the rest
of the foundations page still require the P6 audit; this checkpoint does not
claim that audit complete.

Validation: native access and quick-start suites report **14 passed in 0.06s**;
scoped Ruff check/format and git diff --check pass. Strict MkDocs builds after
quick-start edits and foundations edits both exit 0 (logs
`/tmp/m5-p6-quickstarts-mkdocs.log` and
`/tmp/m5-p6-foundations-mkdocs.log`). P6 remains In Progress, P7 and final gates
remain outstanding, and no PR has been opened.

### P6 wall-loss and coagulation guide checkpoint (2026-11-11)

Migrated both process guides' executable CPU snippets to explicit native
ParticleData and Aerosol construction. The previously undefined pipeline
fixtures now use a documented truncated 50–500 nm, 64-bin PMF, 100 nm mode,
GSD 1.5 and 1e8 m⁻³. The original elapsed pipeline durations are retained,
with 0.25-second substeps. This newly specified fixture is illustrative,
not a convergence or calibrated-atmosphere claim. Initial attempts to use
default broad bins or a narrower GSD failed native positivity checks in tiny
tails; no production clipping or timestep-policy change was introduced.
The separate quick starts retain their original default two-mode PMF; wall
loss's original 10 seconds now uses 40 equal steps to satisfy native positivity.

Added full species/physics configuration for the condensation compositions,
separate resolved-count samples instead of accidentally feeding PMF state to
resolved coagulation, matching distribution kinds in the combined factory,
and required DNS factory unit arguments. Corrected prose that promised
clipping or still described resident integration as future M4 work.
All 11 wall-loss and 21 coagulation Python fences execute in their documented
shared context, including optional GPU declarations. Six focused regressions
verify exact snippets, native pipeline outcomes, finite-step analytical wall
loss decay and number loss with unchanged PMF bin masses. Existing PMF method
limitations remain explicit; no exact coagulation inventory claim was added.

Validation: `pytest examples_tests/native_process_guides_test.py -q --no-cov`
reports **6 passed in 3.87s**; scoped Ruff check/format and git diff --check pass.
Strict MkDocs exits 0 (`/tmp/m5-p6-process-guides-mkdocs.log`). Inventory now
records completed quick-start and migration-page snippet work, these two guides,
and the still-partial foundations-page audit. P6 remains in progress; remaining
process/API/theory/navigation audits, P7 and final gates are not complete.

### P6 condensation guide checkpoint (2026-11-11)

Migrated all 19 Python fences in the condensation guide to aligned native
state, explicit mapping and process-owned activity/surface/vapor-pressure
configuration. Each recipe declares fresh state with its own molar-mass layout;
two-species selective examples no longer accidentally reuse one-species gas.
Resolved batching now actually constructs resolved counts at 1e-6 m³. The
single-material discrete quick start retains the original default two-mode,
250-bin initializer and 10-second step. Previously undefined recipe fixtures
use the explicitly documented finite PMF or two-slot resolved illustration.
Prescribed 1e-3 Pa vapor pressure is an illustrative choice, not calibrated
species physics. Composed elapsed durations are retained with 0.25-second
substeps. Added missing required builder/factory units and removed unsupported
isothermal-factory update_gases input while retaining its True default.
Builders still demonstrate explicit set_update_gases, including False.

Corrected native latent bookkeeping to extensive J and removed the false claim
that a numerical radius clamp establishes continuum validity. Ten closed-recipe
regressions independently sum raw count/bin-weighted particle mass plus gas
concentration times volume, including selective species and latent energy.
All Python fences execute, including the optional direct-GPU import. The combined
process-guide suite reports **18 passed in 5.59s**; scoped Ruff check/format,
git diff --check and strict MkDocs pass. Render log:
`/tmp/m5-p6-condensation-guide-mkdocs.log`. Remaining selected-backend prose in
this guide is explicitly still pending reconciliation; snippets passing is not
proof of every adjacent claim.

Classified 44 docs/.assets/temp snippet rows as generated exclusions after
verifying every path is untracked and ignored; .gitignore:14 owns the ignore and
mkdocs.yml excludes .assets/ from publication. No generated or supported files
were deleted. P6 remains In Progress, with remaining process/API/theory and
navigation audits outstanding; P7, final gates and PR creation remain pending.

### P6 dilution, source and direct-call guide checkpoint (2026-11-11)

Made dilution's public snippet self-contained with resolved counts at 2 m³ and
unified partitioning/inert gas. Corrected raw population units, held array
ownership and native CPU write restoration against dynamics/dilution.py.
Made the nucleation guide executable with three fixed slots at 0.25 m³,
precursor/inert gas, explicit ordered source mapping and concrete commit
controls. Corrected legacy-Aerosol and equal-species-width statements, CPU
count-weight packing, representative-volume scaling and GPU source binding/
independent-width guidance using the existing native source contract. All
original source coefficient, formation size, elapsed duration and substeps
are retained; inert gas is explicitly unconsumed.

The native scientific process page now constructs/configures its direct-call
example through the existing native access fixture. Replaced its stale blanket
M4 pending statement with current bounded integration pointers. During the
cross-guide check, current wall_loss/_native.py confirmed removed occupied
slots clear charge as well as mass/count. Corrected the wall-loss guide's
remaining M3 charge-retention claim and protected-field description; untouched
inactive storage is still not repaired. No production behavior changed.

Three additional exact-snippet regressions check dilution's exp(-1) counts,
number density and all gas lanes; precursor extensive mass conservation with
inert gas unchanged; and direct composition's inert gas and bounded sink
inventory. Current process-guide suite: **21 passed in 6.14s**. Scoped Ruff
check/format and git diff --check pass. Strict MkDocs exits 0, built in 1.44s
(`/tmp/m5-p6-native-source-guides-mkdocs.log`); informational navigation messages
remain and are not claimed resolved. P6, P7 and final gates remain incomplete;
no PR has been opened.

### P6 design-pattern theory checkpoint (2026-11-11)

Replaced executable facade construction in the object-oriented patterns guide
with native aggregate/process composition and ParticleDataBuilder. Preserved
its original 100/200/300 nm radii, 2.5 g/cm³ density, 1e2/1e3/1e4 cm⁻³ bin
concentrations, neutral charge and surface-strategy construction. The native
example computes per-particle masses explicitly and demonstrates stored versus
derived fields. Corrected the explanation that builders inject methods or
particle-owned physics. Fake factory/decorator examples are now real API calls;
conceptual FancyBuilder/mixin sketches are explicitly text, not executable
Python. No supported scientific example was removed.

The design overview now supplies previously undefined functional-kernel inputs
and names ParticleDataBuilder. Its functional/strategy snippets execute, with
symmetry and cubic-radius scaling checked independently. The Activity index
snippet is unchanged and checked against independently calculated ideal mole
fractions; it needs no native aggregate migration.

Validation: new exact-snippet theory suite reports **3 passed in 1.45s**;
scoped Ruff check/format, git diff --check and strict MkDocs pass (log
`/tmp/m5-p6-theory-guides-mkdocs.log`). Initial diff checking caught changed
Markdown lines with trailing spaces; those changed lines were corrected.
Inventory records these three page dispositions. The activity feature guide
still contains incomplete recipes, and remaining theory/reference boundaries,
API targets, navigation, P7 and full final gates remain outstanding. P6 remains
In Progress and no PR has been opened.
