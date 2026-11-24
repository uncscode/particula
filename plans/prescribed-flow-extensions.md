# Turbulent particles, changing winds, dust emission and independent CFD replay

Follow-on to the [CFD coupling roadmap](cfd-coupling.md). At
`426e269f98`, Particula can move gas and binned aerosol through fixed grids,
track resolved grains with prescribed settling, and replay its own accepted
forcing on CPU and CUDA. The [parcel guide](../docs/guide/processes/parcels.md)
and [dune comparison](../studies/prescribed-flow/grid_comparison.py) are the
starting points. All four milestones now have runnable studies and controls.
The sections below retain their design and acceptance criteria.

## Implemented experiments

| Study | Evidence at the supplied settings |
| --- | --- |
| [Turbulent puff](../docs/studies/examples/gas-transport/turbulent_puff.py) | At 4 s, the analytical variance is 2.00 m². The finest grid gives 2.50 m² and the 4,096-particle estimate gives 2.00 m². The absorbing-bed study reports timestep bias separately. |
| [Turning wind](../docs/studies/examples/gas-transport/turning_sounding.py) | The integrated parcel and the fixed-grid dust centroid both finish 20 m north of their release. CPU and CUDA checks recover both donor transfers across an unresolved wind reversal. |
| [Two-gust dust release](../docs/studies/examples/particle-processes/gust_dust_emission.py) | The first gust exhausts the 50 mg fine inventory. Each gust lifts about 155 mg of the coarse class. The example closes bed, airborne, deposited and exported mass at each observation. |
| [OpenFOAM cavity](../docs/studies/validation/prescribed-flow/cavity_replay.py) | The maximum normalized field error is 3.91% and the centroid error is 0.351 mm over 40 native intervals. A [pinned case and export recipe](../studies/prescribed-flow/openfoam-cavity/README.md) reproduces the independent data and refinement bounds. |

The [transport theory](../docs/theory/dynamics/prescribed-transport.md),
[parcel forcing equations](../docs/theory/thermodynamics/parcel-dynamics.md)
and [finite-bed emission model](../docs/theory/dynamics/surface-emission.md)
describe the assumptions behind those numbers. They qualify algorithms with
synthetic controls and a solver comparison. Atmospheric wind, turbulence and
dust-flux calibration require independent measurements.

## Where the changes belong

Keep flow, mixing, inventory sources and replay independently selectable. A
single changing wind should be able to drive both a parcel and a stationary
grid without duplicating the aerosol physics.

| Capability | Implementation boundary | Theory and runnable example destinations |
| --- | --- | --- |
| Turbulent particle motion | Add a dispersion strategy to `GridRun` and its Warp particle mover. Read the diffusivity and face mixing already recorded in each grid interval. | `docs/theory/dynamics/prescribed-transport.md` and `docs/studies/examples/gas-transport/turbulent_puff.py` |
| Changing soundings | Add a sequence provider and builder to `FlowFactory`. Use the same profiles in the dune provider and retain accepted times in `GridReplay`. | Extend `docs/theory/thermodynamics/parcel-dynamics.md` and add `docs/studies/examples/gas-transport/turning_sounding.py` |
| Finite dust supply | Add a surface-emission strategy to the grid runner. Keep per-patch source state with the aerosol checkpoint. | `docs/theory/dynamics/surface-emission.md` and `docs/studies/examples/particle-processes/gust_dust_emission.py` |
| External CFD replay | Translate solver-native cell and face identities into the existing grid interval contract in a concrete importer. Keep solver-specific parsing outside the transport kernels. | Extend the prescribed-transport theory page and add `docs/studies/validation/prescribed-flow/cavity_replay.py` |

## 1. Give resolved particles and the grid the same turbulent spread

Implement an opt-in particle dispersion strategy in the existing grid runner.
Keep deterministic advection and settling as the default. Begin with a flat
grid and diagonal, constant eddy diffusivity, reading the same m²/s values as
the field transport. Give each particle a displacement with mean air motion
plus terminal slip, and independent turbulent increments with variance
$2K_i\Delta t$ per active axis. In a 2D slice, suppress stochastic
displacement across its omitted dimension. Grid and particle runs of the
same release are alternative representations of one inventory.

Key random draws by seed, stable particle ID, accepted interval, substep and
axis. Persist the algorithm version, counters and schedule needed for exact
same-device continuation. Preserve each grain's count, charge and material
through periodic wrapping, reflection, deposition and escape. For absorbing
boundaries, include first-passage handling or demonstrate timestep convergence
against a first-passage reference. A diffusing particle can hit the bed even
when both sampled endpoints are above it.

**Theory and example.** Add a transport theory page under `docs/theory/dynamics/`.
Derive the Gaussian puff moments, the distinction between eddy and molecular
diffusion, and the artificial variance of the existing upwind scheme.
Publish *How wide is the same puff on a grid and as individual particles?*
Plot mass-weighted centroid and variance against time for several grid sizes
and particle counts, and show the deposited fraction in a separate absorbing-bed
control. For particles starting at height $h$ above a flat absorbing bed
with zero vertical drift and constant $K_z>0$, the reference fraction
deposited by time $t$ is $\operatorname{erfc}(h/\sqrt{4K_z t})$. Place the
upper boundary far enough away to make its effect negligible over the test.
Include the zero-diffusivity run.

**Acceptance.** Check inventory by fate, analytical mean and variance over
independent seeds, and exact same-device restart. Set sampling intervals from
the ensemble size before evaluating results. Compare CPU and CUDA distributions
and deterministic budgets. Extend to a dune only after a well-mixed test on
unequal physical cell volumes passes. Specify the drift and metric terms for
the chosen variable-diffusivity coordinates and check grid/time refinement
before enabling that combination. One candidate for the dune is to draw
particle transitions from the same symmetric face mixing volumes as the
grid. With physical volumes $V_i$ and substep transition probabilities
$P_{ij}$, test $V_iP_{ij}=V_jP_{ji}$ across each shared face. This directly
tests whether a uniform concentration stays uniform in cells of unequal size. [Visser's random-walk analysis](https://orbit.dtu.dk/en/publications/using-random-walk-models-to-simulate-the-vertical-distribution-of/)

## 2. Drive a parcel and grid with a sequence of soundings

Add a time-dependent sounding builder and factory entry alongside the current
single-profile provider. Take ordered timestamped profiles in the existing
SI format with a common vertical datum, time origin and provenance. At the
requested height, interpolate each bracketing profile vertically, then
interpolate east/north wind, temperature, pressure and dry-air vapor mixing
ratio in time. Use the shared height range and time range. Reject out-of-range
queries. Allow the dune provider to use the same changing upstream atmosphere.

Advance exactly to profile knots. Split transport when a face reverses flow
so opposite donor transfers survive recording and replay. For linear winds on
fixed geometry, integrate signed face volume on each constant-sign interval.
Sample moving parcels along their path. Save forcing at the actual accepted
times and extend the archive schema if subinterval fields need new metadata.
Continue reading existing steady grid archives and parcel CSV/JSON pairs.

**Theory and example.** Extend [parcel dynamics](../docs/theory/thermodynamics/parcel-dynamics.md)
with height/time interpolation, external humidity ownership and the difference
between a moving observation and a fixed-cell observation. Publish *Where does
a dust puff go when the wind turns?* Use two or three supplied synthetic
profiles with wind rotation and a humidity change. Plot the prescribed wind
vectors, the parcel path against the grid centroid, and wet radius along the
path. Include the input table and instructions for substituting measured
soundings with units and height datum.

**Acceptance.** Recover every input knot and the existing steady case. For a
spatially uniform wind linear in time, compare displacement to its exact time
integral. Force a sign reversal between requested output times and check each
donor inventory. Verify humidity forcing and replay at common times, including
restart on either side of a profile change, on CPU and native CUDA.

## 3. Emit dust continuously from finite, size-resolved bed patches

Add a surface source strategy that computes vertical emission by patch and
size class at each accepted interval. Use the existing friction-velocity
diagnostic $u_*=\sqrt{C_D}\,\sqrt{u^2+v^2}$, recording the drag coefficient
and sampling height above the local bed. Start with an illustrative excess-shear law
$F_i=E_i\max[(u_*/u_{*t,i})^2-1,0]$, with $E_i$ in kg m⁻² s⁻¹, a positive
threshold in m/s and physical erodible patch area in m². Supply the
coefficients per size class. Define the flux per unit sloping bed area and
include the physical crosswind thickness for a 2D slice. Derive binned
counts and resolved statistical weights from the same single-particle dry
mass. Document the dry, loose-soil assumptions for
any optional size-dependent threshold. Quantitative calibration of vertical
dust emission requires flux observations. [Shao and Lu on threshold friction velocity](https://doi.org/10.1029/2000JD900304)

Integrate across gust knots and threshold crossings, cap each class by its
available bed mass, then debit and inject the same quantity once. Use one
destination per emitted class in a run. Binned injection adds both represented
number and material mass. Resolved injection creates weighted records with
stable IDs and declared ejection height and terminal slip. Check available
record capacity before debiting the bed. Keep redeposited material in a
separate reservoir for this milestone. Save patch inventories, source phase
and particle ID allocation in the checkpoint.

**Theory and example.** Add a surface-emission theory page covering friction
velocity, mass flux versus area-integrated release, grain diameter, emitted
aerosol size and finite-bed depletion. Publish *What does the second gust
lift from a partly depleted dune?* Use the changing sounding from milestone 2.
Plot shear and class thresholds, emitted size distribution for each gust, and
bed, airborne, deposited and exported kg by class.

**Acceptance.** Check zero emission below threshold, an exact constant-shear
release, and a linearly increasing $u_*$ that crosses a class threshold
between output times. Compare emitted kg with the analytic integral from
the crossing to the interval end. Check depletion within an interval and
a two-gust case with one exhausted
class. Check per-class number and species mass, including rejected capacity
requests and restart during emission. Compare equivalent binned and resolved
source totals and CPU/CUDA ledgers.

## 4. Replay a passive tracer from an independent CFD solver

Start fixture acquisition alongside milestone 1. Use a pinned OpenFOAM
incompressible lid-driven cavity on a uniform Cartesian mesh one cell thick.
Run the host velocity solution and evolve a nonuniform passive scalar using
the host transport equation. Choose a prescribed constant diffusivity,
impermeable walls and zero normal scalar flux. This exercises rotating flow
and face orientation within the current grid geometry. [OpenFOAM scalar transport](https://doc.openfoam.com/2312/tools/processing/solvers/rtm/basic/scalarTransportFoam/)

Export native cell IDs, centers and physical volumes, face owner/neighbour
and patch orientation, accepted times, signed volumetric face fluxes and
independent scalar snapshots including the initial condition. Record the
solver revision, dictionaries, scalar units, reference density where needed,
advection scheme, timestep, redistribution terms and data redistribution
rights. Use declared constant thermodynamic inputs for this isothermal case.
Keep kinematic pressure distinct from the absolute pressure used by aerosol
physics. Retain a small redistributable fixture with checksums and an export
recipe. The published replay should run offline without installing OpenFOAM.

Map the host rows and faces to Particula once. Reorder by stable identity,
convert rates to m³ over actual intervals and check volume continuity and
opposite internal signs. Reject incompatible geometry, ambiguous units or
incomplete transfer history. If a candidate export only contains cell-center
velocity, obtain the solver's face output before claiming conservative replay.

**Theory and example.** Extend the transport theory page with the
finite-volume balance, face signs, dry-air versus volume normalization, and
transport ownership. Publish *Can Particula reproduce the cavity's tracer
circulation?* Show host and replay concentrations on the same color scale,
a residual panel, and error and inventory at common times. Add local aerosol
uptake as a second experiment after passive transport passes.

**Acceptance.** Compare total scalar inventory, centroid and volume-weighted
field error at common physical times. On the same host mesh, halve the
host timestep to isolate temporal error. In a separate run, double the
resolution along each active axis and volume-average the fine scalar onto
the replay cells. Record the host face-flux timestamp and interval
quadrature convention. Freeze the comparison tolerances using the
refinement differences and the two transport schemes before scoring the
reference case. Exercise
permuted cell IDs, reversed face orientation, missing intervals, and restart.
Compare CPU and native CUDA against the same frozen external data. Publish
the source export and residuals as verification against an independent solver.

## Delivery and publication

1. Deliver milestones 1 and 2 independently against the committed baseline.
   Combine them for the dune checks. Build milestone 3 on the changing-wind
   provider. Milestone 4 can proceed as soon as a usable host fixture exists.
   Carry the existing factory names, deterministic defaults and forcing units
   through each change.
2. Ship each capability with its theory, worked example and tests in the same
   change. Put the three transport examples under the Studies examples tree,
   grouped with gas transport or particle processes as appropriate. Put the
   external replay under Studies validation with its runnable companion.
   Link each from the parcel/flow guide and its governing theory page.
3. Register each percent-format Python source and study in
   `website/examples.json` and `website/topics.json`; refresh the generated
   study listings. Save input provenance, run settings, scheme, precision,
   device, numerical results and figure data. Record seeds and ensemble size
   for stochastic runs. Each example should explain what the reader can change
   and which plotted diagnostic would reveal its effect.
4. Run the affected CPU/CUDA suites and each full example, then check study
   and topic registration, docs structure, saved-output rendering and the
   website build. Inspect the rendered equations, units, figures and links.
   Keep algorithm verification, comparisons against another solver, and
   agreement with field measurements identifiable in every report.
