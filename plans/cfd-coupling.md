# Particula parcels, parcel clusters and CFD coupling

The prescribed-flow milestone now includes a closed warm-cloud ascent, offline replay, conservative parcel clusters, and size-dependent particle crossings. A 100-second Particula-generated trajectory supplies the controlled forcing. A pinned OpenFOAM cavity export now provides an independent passive-tracer comparison. Live solver coupling remains a later step. [Replay study](../docs/studies/examples/thermodynamics/parcel-experiments/parcel_replay.py) · [Cluster study](../docs/studies/examples/thermodynamics/parcel-experiments/parcel_cluster.py)

## First implementation PR — one differentiable moist parcel on CUDA

Give a researcher one runnable ascent from subsaturated air through cloud base. Prescribe a constant updraft and evolve pressure, temperature, physical volume, vapor and droplet water together. At a fixed observation time the researcher can ask how much more liquid water the parcel would contain for a slightly faster updraft. The same forward calculation supplies the trajectory and its derivative.

### Physical state and one accepted step

Start with one closed, non-entraining, warm parcel, a fixed dry-air mass `m_d` in kg, water as the only condensable, and one soluble dry aerosol mode discretized into fixed particle classes. Initialize height (m), pressure (Pa), temperature (K), vapor mixing ratio (kg water per kg dry air), dry size, hygroscopicity and represented number by class. Equilibrate the initial wet radii at the specified subsaturated humidity and include their liquid water in the initial inventory. Hold that initial state fixed when varying the updraft.

1. Advance height using the prescribed velocity (m/s) and pressure using hydrostatic balance. Document the air-density and heat-capacity approximations used for the whole benchmark. In the dilute dry limit the temperature follows the dry adiabat.
2. Evaluate signed condensational growth for each class using current vapor, temperature, pressure and wet radius, with a documented κ-Köhler equilibrium and heat- and diffusion-limited water growth law. Sum the per-particle changes weighted by represented number. Apply that single mass exchange to the vapor and liquid budgets with opposite signs. Use the same exchange for latent heating on condensation and cooling on evaporation, alongside expansion cooling in the first law. Integrate these coupled tendencies on the same time grid. [Reference parcel equations](https://pyrcel.readthedocs.io/en/stable/user_guide/sci_descr/)
3. Compute the physical parcel volume in m³ from the gas equation of state, `V = (m_d R_d + m_v R_v) T / p`, neglecting condensed-phase volume and dry aerosol loading. Here `m_v` is vapor mass and the two gas constants are specific to dry air and vapor. Diagnose vapor concentration as `m_v / V`, particle number density as represented number divided by `V`, and supersaturation as `e / e_sat(T) - 1`, with `e = m_v R_v T / V`. Expansion alone dilutes concentrations and preserves extensive inventories. Keep physical volume in the parcel state and respect the native discrete population's 1 m³ normalization when exposing concentrations. [Native particle measures](../docs/guide/data-model/index.md)

### Implementation boundaries

- Put the state contract, input validation and CPU reference step with the parcel dynamics. Introduce one concrete runner that takes initial state, updraft, timestep, step count and device. Keep the interface small until a second parcel use case exercises it.
- Put the Warp thermodynamic update, class growth, water reduction and observation in the GPU kernels. Keep all evolving fields, including the differentiable updraft input and scalar observation, on the selected device for the full trajectory. Allocate distinct step outputs for the reverse pass. Read back histories and gradients at the observation boundary. Start with a fixed integration schedule and demonstrate convergence for the cloud-base case.
- Reuse the smooth out-of-place Warp operations and reverse-pass pattern where their equations match. The existing inverse condensation path differentiates a terminal radius with a shared reservoir and prescribed growth coefficients. The parcel needs derivatives through pressure, temperature, volume, saturation and the latent-heat feedback at every step. The existing raw-rate and terminal-radius tests provide smaller regression cases. [Inverse condensation](../particula/execution/inverse_condensation.py) · [Raw-rate tests](../particula/gpu/kernels/tests/condensation_autodiff_test.py) · [Terminal-radius tests](../particula/execution/tests/inverse_condensation_test.py)

### Acceptance checks for the implementation PR

| Check | Acceptance evidence |
| --- | --- |
| Dry ascent and dilution | With phase exchange disabled, recover the dry potential-temperature invariant in the vapor-free limit. In a separate passive-vapor case, verify constant vapor mass and particle counts as volume increases, including the predicted concentration ratio. |
| Stationary and moist controls | An equilibrated zero-updraft parcel stays stationary. In a closed moist ascent check water mass, dry aerosol mass and particle count throughout, and independently evaluate the first-law residual from pressure work, temperature and liquid-water changes. Exercise evaporation as well as condensation and reject invalid inventories. |
| Cloud-base benchmark | Match initial wet spectrum, bin weights, constants and activation criterion to an independent parcel calculation such as pyrcel. Compare supersaturation versus height, peak supersaturation, wet radii and number-weighted activated fraction. Refine the timestep and resolve the supersaturation peak. Establish tolerances from reference accuracy and refinement before judging the production case. |
| CPU and CUDA | Compare histories for height, pressure, temperature, volume, vapor, liquid water, wet radii and supersaturation against the CPU reference at common times. Record the actual NVIDIA device, precision, timestep and maximum discrepancies with budget errors reported separately. |
| End-to-end derivative on CUDA | At a fixed elapsed time in the moist ascent, differentiate liquid-water mixing ratio with respect to constant updraft speed through multiple coupled steps using Warp Tape. Check a finite, nonzero gradient against centered perturbations of the entire forward run over a range of perturbation sizes. Use an interior case that stays on the same numerical branches, report the agreement and check timestep refinement. Peak supersaturation and thresholded activated fraction are forward diagnostics. |
| Researcher example | Ship a percent-format Python example with the initial condition, device, solver settings and saved results. Show cooling and expansion, the supersaturation peak and wet growth versus height, then report the signed updraft sensitivity with its units and finite-difference check. |

The [worked example](../docs/studies/examples/thermodynamics/moist_parcel.py) executes a 100 s ascent from 90 kPa and 288.15 K. The CUDA runner resolves the supersaturation peak and its reverse pass differentiates terminal liquid water with respect to updraft. The independent SciPy Radau calculation integrates pressure, temperature, vapor and squared wet radius. Both implementations assume continuum droplet growth and the same thermodynamic coefficients. Agreement measures numerical error under these assumptions. [Implementation](../particula/dynamics/parcel.py) · [Physical and gradient regression tests](../particula/dynamics/tests/parcel_test.py)

The complete temperature, pressure, volume and water-vapor records from the independent Radau calculation are available in the [forcing table](../docs/data/parcel-replay/ascent.csv) and [initial state and provenance](../docs/data/parcel-replay/ascent.json).

## Gas ownership and updates at each step

Choose an owner for each named gas species when setting up an experiment. Water vapor and an organic vapor can have different owners in the same Particula box. Each exchange declares which physical processes have already contributed to the supplied state or change.

| Gas owner | At an accepted step |
| --- | --- |
| Parcel or CFD solver | Read the native gas value at the agreed point in the step and convert its units and air-mass basis to Particula concentration. For live two-way coupling, return the signed gas mass change from the aerosol processes delegated to Particula. The owner commits it once. With a read-only trace, retain the exchange as a diagnostic and account for any gas restored by the next prescribed sample. |
| Particula | Initialize the named gas once and carry its inventory forward. Apply the parcel or solver's reported volume change and any declared species source, sink or mixing increment over the interval. Then apply Particula chemistry and aerosol exchange. With several moving cells, transport requires either the external solver to carry an auxiliary tracer or enough information from the solver to supply conservative species transfers. |

In the mixed-species experiment, let the parcel supply water vapor and let Particula track an organic vapor absent from the parcel model. In a closed parcel with organic gas mass `M` and new volume `V_new`, expansion alone gives `C_new = M / V_new`. With an external mass increment `ΔM_ext`, the pre-chemistry concentration is `(M + ΔM_ext) / V_new`. This uses the actual parcel volume in m³. Entrainment also needs incoming composition or a supplied species mass increment. If a source is supplied as a rate, integrate it over its declared interval and basis. Apply a supplied step total once. Changes in temperature and pressure can update thermodynamic diagnostics, while any external gain or loss of species mass needs a physical source.

Map updates by species name into Particula's ordered gas fields so updating water preserves the evolving organic vapor. Record the ownership, units, update type (value, increment or rate), and interval in the forcing history. Save the last applied interval with the state on restart. Validate the mixed case by doubling the parcel volume with no sources, then enabling a known organic addition and condensational sink. Check organic gas plus particulate mass, the water feedback budget, and replay across a restart.

## Replay the same aerosol experiment

Save the parcel's imposed history at the actual forcing times, then feed it to the same aerosol calculation with the same initial state. Compare the wet-size history, supersaturation and water budget at matching times. This first replay checks timing, units and field ownership. It also gives researchers a way to vary aerosol composition or process choices under the same external conditions.

Then read one trajectory exported from an existing CFD simulation. Start with an ordinary table containing trajectory ID, increasing time in seconds, temperature in K, pressure in Pa and a defined dry-air density or equivalent volume history. Include position or altitude when available. For humidity, record its variable, units and air-mass basis, and whether it is initial-only, externally imposed or a diagnostic. Prescribing vapor at every record makes it an external reservoir and its exchanges enter the budget. Record the source run, coordinates, sampling cadence and interpolation. Replay native timestamps first, define how intermediate forcing is sampled, and stop at the last record. A fixed-grid-point history drives a stationary box. A Lagrangian history follows moving air. Gridded snapshots require a path and a sampling method before they can supply a parcel history.

The worked replay integrates the same κ-Köhler water-growth law under prescribed thermodynamics. It compares six combinations of forcing cadence and growth timestep with the closed ascent. A passive organic tracer has a first-order uptake coefficient in s⁻¹. Its condensed mass stays separate from the dry solute and water activity in this experiment. The water ledger reports signed condensation and the external mass needed to enforce the trace. The organic ledger includes interval sources and endpoint pulses.

## Parcel clusters

`ParcelCluster` groups regular or custom arrangements of parcels in one Particula experiment. Each parcel keeps a stable ID, a center position in metres, a physical volume in m³, its own forcing history and one row of the batched aerosol state. Run local aerosol processes over the batch and retain results by parcel ID. Researchers can track parcels together with exchange disabled or enable specific connections for gas, particle populations or both.

| Arrangement | Example dimensions (x, y, z) | Example research use |
| --- | --- | --- |
| Vertical stack | (1, 1, 4) | Follow settling through layers with different humidity. |
| Horizontal line | (6, 1, 1) | Follow aerosol aging across a prescribed plume transect. |
| Plane | (4, 3, 1) or (4, 1, 3) | Compare a horizontal slice with a vertical curtain. |
| Cube | (2, 2, 2) | Track a particle crossing faces along multiple axes. |
| Custom arrangement | Named parcels at supplied coordinates | Replay an irregular group of trajectories with selected links and unequal volumes. |

A regular layout constructor takes cell counts and physical cell widths along x, y and z. Singleton axes still have physical thickness. It generates parcel centers, volumes and shared faces. For custom layouts, the researcher supplies the parcels and connections. A connection can carry a prescribed exchange rate between two identified parcels. For geometric particle crossings, supply cell bounds and oriented shared faces or a provider that locates particles. Positions alone are sufficient to plot tracks. They do not determine which parcels exchange material.

Each internal transfer debits one parcel and credits the other by the same inventory. For prescribed volumetric flow, record direction, units and the donor concentration used to obtain a species mass transfer. A boundary face can retain particles, pair periodically with another face, or record material leaving the cluster according to the experiment. Gas and particle boundary rules may differ. In moving layouts, replay positions by ID and use motion relative to the shared face for crossings. When trajectories separate and cease to share a face, continued exchange needs a declared connection history or updated geometry.

**First cluster check.** A single parcel reproduces the existing parcel run. With exchange disabled, a stack reproduces its parcels run independently under their respective forcing. Then move a known gas mass across a single connection between unequal-volume parcels and check both concentration changes and cluster-wide mass. Use the two-parcel stack for the differential settling experiment below. Verify a cube has eight cells and twelve internal shared faces, and that permuting storage rows in a custom layout preserves results by parcel ID. Save per-parcel aerosol histories, per-connection transfers and cluster totals so a researcher can trace any gain or loss.

## Particle ownership and differential motion

Choose who advances position for each particle population independently of gas ownership. For a particle-resolved experiment, default to Particula owning the particle records and trajectories. The flow provider supplies air velocity, its sampling times, and the geometry needed to locate particles and handle boundary crossings. A solver with its own particle tracker can instead return positions or crossing events keyed by particle ID, while Particula advances composition and size. Exactly one component commits each displacement and each deposition event. For a binned population, the provider can transport number and species mass as conserved fields.

In the simplest resolved case, Particula samples air velocity at the particle position and adds its terminal settling velocity relative to the air. Use an explicit coordinate frame. A vertically moving parcel mesh already follows the parcel updraft, so only velocity relative to the moving mesh determines its face crossings. With finite particle inertia, the experiment would also need particle velocity and an appropriate response model. Start with terminal settling and reuse Particula's existing settling calculations within their validity range. [Settling velocity](../particula/particles/properties/settling_velocity.py)

A resolved particle crossing a face retains its identity, represented count, species masses, charge and associated per-particle state. The source and destination can use different local slots. Reserve destination capacity before retiring the source, record open-boundary exits or deposition separately, and preserve motion state on restart. Particula's existing resident communication kernels handle prescribed transfers between boxes and offer relevant capacity checks. A trajectory-driven mover also needs positions, cell lookup and boundary crossing logic. [Existing GPU communication](../particula/gpu/kernels/communication.py)

**First resolved-particle cluster experiment.** Use two fixed vertical cells in a parcel stack on a closed or explicitly budgeted column and prescribe uniform upward air velocity. Start small and large particles together just above their shared face. Choose an updraft between their terminal settling speeds so the small particle rises and the large one crosses into the lower cell. With constant properties and no aerosol reactions, compare positions and crossing time to the constant-velocity solution. Check each ID arrives exactly once and that represented number, species masses and charge sum to their initial values including any recorded exits. Run the same test on CPU and CUDA. A single parcel trace suffices for the air history inside that parcel. Following a particle after it leaves requires a neighboring parcel or a spatially sampled flow field.

## Intermediate experiment with prescribed spatial flow

A CPU provider in `particula.coupling.prescribed` samples a steady sounding and
moves named parcel centers over a Gaussian dune. The air accelerates through
the reduced cross-section and follows the bed slope. Each center supplies the
existing parcel replay with temperature, pressure, vapor mixing ratio and
physical volume for a fixed mass of dry air. Geometric streamlines conserve
volume flux under an incompressible approximation. Replay diagnoses the
thermodynamic volume separately. The moving centers represent independent
air parcels. Their positions alone define neither shared faces nor exchange.

The [dune experiment](../studies/prescribed-flow/dune.py) loads a
[synthetic sounding](../studies/prescribed-flow/synthetic_sounding.csv).
From the worktree root, run `PYTHONPATH=. python studies/prescribed-flow/dune.py`
to generate `.artifacts/prescribed-flow/dune.png` and the bed mass budget.
An observed radiosonde can supply the same named columns after quality
control and conversion to the common height datum, SI wind components and
water mixing ratio per kg dry air. Profile coverage must include the bed
datum and lid. Wet aerosol initialization requires subsaturation.

The illustrative bed patch releases a specified mass pulse above a friction
velocity threshold. A grain and fine mineral dust start at a prescribed
ejection height with size-dependent terminal slip, and first contact with
the bed ends each track. The bed, airborne and redeposited masses close the
source budget. Threshold, drag coefficient, launch height and available mass
are experimental inputs. Calibration of sand flux, ballistic saltation
and lee-side separation remain open. The extension below adds turbulent
particle motion with a prescribed diffusivity over the mapped dune.

Flat-terrain controls in `particula.coupling.dispersion` describe a
continuous source with Pasquill A–F open-country widths and an instantaneous
puff with constant directional diffusivities. The puff can also use a uniform ellipsoid with the same free-air variances.
The ground-image solutions
predict cross-section mass flux and puff mass, respectively. Stability class
sets continuous-plume spread. The dune takes its wind and thermodynamics
from the sounding. [Briggs widths, Table A-1](https://www.pnnl.gov/main/publications/external/technical_reports/pnnl-20990.pdf)
· [NOAA Gaussian dispersion overview](https://www.arl.noaa.gov/hysplit/)

### Strategies, stationary grids and recorded forcing

The [parcel and grid guide](../docs/guide/processes/parcels.md#use-one-flow-for-parcels-and-stationary-grids)
uses the same flow factory for uniform wind, soundings, dunes and replay.
Concrete builders validate the provider inputs. A spatial flow can move a
parcel or provide air and shared-face transfers to a stationary x–z section,
x–y layer or x–y–z volume. Trajectory records retain one parcel identity.
Independent gridded replay uses accepted intervals and physical volumes with
stable C-order cell IDs. A host can return a shorter interval or supply
already transported inventory and its boundary ledger.

Run `PYTHONPATH=. python studies/prescribed-flow/grid_comparison.py` from the
checkout for the CPU comparison. Add `--device cuda:0 --output .artifacts/prescribed-flow/grid-cuda`
for native CUDA. The [comparison script](../studies/prescribed-flow/grid_comparison.py)
writes a parcel CSV/JSON pair, section and volume forcing archives, a figure
and a numerical summary. It replays each archive and checks the final field
and resolved-particle inventories. The initial binned dust cloud and the two
bed pulses have independent budgets.

The analytical controls cover periodic displacement and numerical variance,
directional diffusion, A–F transverse widths in flat steady flow, uniform
concentration over the mapped dune, extrusion of both 2D orientations,
restart around pickup and CPU/CUDA agreement. The finite-volume method is
first-order upwind with explicit diffusion. Particle motion freezes local
velocity over a bounded substep and traverses each crossed face. The
terrain surface is piecewise linear between supplied bed vertices.
Grid refinement and independent wind and sediment observations address
different sources of uncertainty. The independent cavity replay extends these controls to a frozen OpenFOAM
flow. Host-driven EAMxx and multi-rank transport remain future milestones.

The [prescribed-flow extensions](prescribed-flow-extensions.md) add a seeded
turbulent particle mover, time-varying sounding profiles and finite dust sources.
A [lid-driven cavity replay](../docs/studies/validation/prescribed-flow/cavity_replay.py)
imports native OpenFOAM face fluxes and compares against the host scalar.
The pinned [case and export recipe](../studies/prescribed-flow/openfoam-cavity/README.md)
reproduce the reference, half-step and finer-mesh comparisons.

## Later live coupling interface

For live coupling, each owned CFD physics cell maps to one logical Particula box. The solver supplies flow and transports the inventories delegated to it. Particula can move resolved particles through that flow. The parcel and replay studies establish the fields and units that the eventual adapter must carry.

At setup, validate the box IDs in the initial aerosol against the provider's owned cells, including a deliberate mapping or broadcast when initialized from a single uniform box. The researcher can first select thermodynamic forcing only, then enable conservative transport for named inventories when the provider supports it. Forcing-only histories may refer to fixed cells or moving parcels, according to how they were sampled.

## Existing capabilities

**Particula.** A prepared simulation already batches boxes on a device, retains scratch, and supports explicit process registration. Its public forcing updates accept host values and synchronize. Its prepared timestep and layout are fixed. A CFD adapter needs an accepted-step interface for resident fields and a way to import transported particle inventories before the next physics call. Reuse the existing preparation and numerical kernels. The process registry remains useful for adding local physics. The resident inter-box communication machinery models a prescribed exchange and should stay out of host-transported tracer steps. Baseline reviewed at `51e25ee110`. [Prepared execution](../docs/guide/gpu/prepared.md) · [Forcing implementation](../particula/execution/prepared_updates.py) · [Resident communication](../particula/execution/resident_communication.py)

**E3SM.** Use EAMxx, the C++/Kokkos atmosphere model, and start with DP-SCREAMv1. Its small periodic domain preserves the global model's PG2 physics–dynamics coupling and tracer transport. EAMxx has an atmosphere-process factory, field declarations, and optional support for calling a Python module from a process. At revision `20150836d1`, the cloud-fraction process has a device backend and its Python example constructs a strided CuPy view over EAMxx-owned CUDA memory. That provides a concrete prototype route. Warp compatibility, stream ordering and throughput in a coupled job remain open. [DP-SCREAMv1 paper](https://doi.org/10.1029/2025MS005127) · [Atmosphere processes](https://docs.e3sm.org/E3SM/EAMxx/developer/processes/) · [Python build support](https://docs.e3sm.org/E3SM/EAMxx/user/eamxx2py/) · [Device example](https://github.com/E3SM-Project/E3SM/blob/20150836d1af1cbdafc76f69f3722258d7527eab/components/eamxx/src/physics/cld_fraction/cld_fraction_cupy.py)

## Proposed interface and ownership

After the parcel and trace studies, Particula would maintain optional live adapters in its own Python package. Each adapter translates the shared aerosol inventory contract into the conventions of one solver. The proposed module boundaries are:

| Particula module | What it owns |
| --- | --- |
| `particula.coupling` | The coupled experiment, box identities, definitions and units of the inventories to move, exchange times, and aerosol budget diagnostics. |
| `particula.coupling.eamxx` | Maps EAMxx physics columns and levels, tracer names, air-mass conventions and strided device fields to the same Particula quantities. Serves both doubly periodic and global configurations. |

The base Particula import and installation work without a CFD solver installed. Only a selected adapter needs its solver available. Each solver owns its flow equations, fluxes, boundary conditions and parallel transport. Particula owns the choice of which aerosol quantities travel together, their conversion back to native particle state, and the diagnostics used to interpret the experiment.

**Scheduling.** With a callable provider, Particula requests the next coupling time and the provider reports the time reached. In a live E3SM case, CIME launches EAMxx and the atmosphere driver invokes a thin compiled atmosphere process at its scheduled physics boundary. This entry point loads Particula’s EAMxx adapter, which constructs the same Particula experiment once per rank and calls its local exchange operation for the offered interval. [EAMxx atmosphere-process lifecycle](https://docs.e3sm.org/E3SM/EAMxx/developer/processes/)

| Boundary | Responsibility |
| --- | --- |
| Bind once per rank and device | Particula requests the fields and transport capabilities required by the experiment. The provider describes owned cells, stable IDs, units, air-mass basis, species order, dtype, shape, strides, device and lifetime. Agree on geometry, buffer ownership and the supported coupling interval before allocating persistent state. |
| Transport to the agreed time | For transport-enabled runs, each transport owner advances its assigned fields or particles. Particula reads the provider's current thermodynamics and transported inventories at the agreed time. Start with fixed intervals and transport followed by aerosol physics. A short advance must be reported. Particula must cover exactly the accepted duration or stop at the last consistent boundary. |
| Advance aerosol physics and return changes | Particula advances the configured processes across the mapped boxes and refreshes its aerosol diagnostics. The adapter routes each increment to the declared field owner for a single commit. Read-only trace fields record the corresponding diagnostic exchange. Convert increments to rates only for an interface that expects rates. Thermodynamic feedback requires an explicit energy and water ownership agreement. |
| Coordinate CUDA work | Borrow a view while the provider guarantees its validity. Use one stream or producer/consumer events in both directions. Use persistent device packing for incompatible strides, units or types. Test logical levels, padding, halos and subfields. Report invalid state before committing fields. Retry only from a matching flow and aerosol state. |
| Restart at a completed exchange | Match the Particula checkpoint and the solver restart to the same time and splitting phase. Particula retains aerosol history, species schema and RNG. The solver retains its flow and transported fields. Record which copy of every exchanged quantity is authoritative on restart. Rebind by logical cell ID when decomposition changes. |

EAMxx maps an **owned physics column and physical midlevel** to a row. A batch contains many rows. Bind EAMxx through a thin compiled `AtmosphereProcess` that declares fields and invokes the optional Python adapter once per rank or tile. Probe the existing device-Python path before considering a native bridge. Even a borrowed pointer requires the actual Kokkos strides and a valid stream handoff. [EAMxx field wrapping](https://github.com/E3SM-Project/E3SM/blob/20150836d1af1cbdafc76f69f3722258d7527eab/components/eamxx/src/share/field/field_pyutils.hpp) · [Warp interoperability](https://nvidia.github.io/warp/latest/user_guide/interoperability.html)

## What moves between cells

For the binned tracer path, start with a dry, single-material, single particle class. Have the flow solver carry three nonnegative inventories for each cell, namely vapor mass, particle number and particulate species mass. Here the class is a population with a local mean mass. Mixing populations this way loses within-class size variation. Quantify that closure against a finer-bin reference before using it for size-sensitive physics.

For host air mass density `rho_ref` in kg/m³, let `q_g` and `q_p` be gas and particulate kg per kg of reference air, and `eta` be particles per kg of the same air. Native inputs are gas concentration `rho_ref * q_g`, number density `N = rho_ref * eta`, and mass per particle `m = rho_ref * q_p / N` for occupied cells. Define an empty class as zero number and zero particulate inventory. Reject positive mass with zero number. When the host already carries conserved density in kg/m³, use it directly. Specify dry versus moist air for every ratio and use the matching reference density and layer air mass.

Particula's discrete population uses a 1 m³ normalization. Keep actual CFD volume as adapter geometry and integrate concentrations with physical cell volume or equivalently ratios with the corresponding cell air mass. For a resolved population the native box volume and represented counts have different semantics. Transport of fixed slot numbers across faces would mix unrelated particles. Use the particle migration and capacity rules above for trajectory-driven resolved transport. [Native measures and units](../docs/guide/data-model/index.md)

For the first two-way transport test, use isothermal uptake of a dilute, non-water test vapor. Debit vapor and credit particle mass in the same cell and retain number. Check this transfer in mass units before and after host conversion. For water, identify which host scheme owns activation, condensation, condensate and latent heat, including P3 or MAM where enabled. Introduce that feedback as a separate experiment with total water and energy budgets. A forcing-only run with aerosol left in stationary boxes is valid for testing a local process or exchange timing. Label it explicitly, since it cannot represent transport of that aerosol.

## Research and delivery sequence

| Step | Experiment and decision |
| --- | --- |
| 1. Parcel rise | Implemented for one closed, soluble warm-cloud mode. The worked example and tests above exercise budgets, activation, CPU/CUDA agreement and updraft sensitivity. |
| 2. Offline replay | Implemented for prescribed water and a named, internally carried organic tracer. The 100-second ascent, cadence sweep, reservoir correction, restart around an injection and humidity sensitivity are in the [replay study](../docs/studies/examples/thermodynamics/parcel-experiments/parcel_replay.py). The prescribed-flow extension below adds an independent fixed-cell CFD export. |
| 2b. Parcel cluster | Implemented for regular and custom layouts, specified contacts and unequal volumes. [Cluster controls](../particula/dynamics/tests/parcel_cluster_test.py) cover independent histories, dry material, number, water, organic mass, row permutation, overdraw and smooth CUDA sensitivities. |
| 2c. Differential particle motion | Implemented for creeping-flow terminal settling, constant velocity per cell and axis-aligned bounds. Moving-frame, capacity, corner, periodic, open, deposition and restart cases have [regression controls](../particula/dynamics/tests/parcel_motion_test.py). The [two-cell experiment](../docs/studies/examples/thermodynamics/parcel-experiments/parcel_cluster.py) compares positions and crossing time with the constant-velocity solution. |
| 2d. Prescribed spatial flow | CPU prototype with flat plume and puff controls, sounding-driven dune trajectories, parcel replay and budgeted pickup/deposition. Compare against measured wind and sediment flux before treating the dune as a predictive case. |
| 2e. Flow strategies and fixed grids | CPU and CUDA conservative grid transport, particle crossings, two mixing strategies and recorded replay. The comparison uses one prescribed dune atmosphere across moving parcels, a fixed section and a volume. |
| 2f. Prescribed-flow extensions | Turbulent particles, changing soundings and finite dust release use the shared 2D/3D grid. The independent passive-tracer cavity compares both devices with frozen OpenFOAM data. [Milestones, theory and examples](prescribed-flow-extensions.md) |
| 3. Define the live interface | Use the parcel and replay results to specify forcing and diagnostics. Add the fields that conservative transport requires, including cell IDs, volumes, tracer inventories and accepted exchange times. Check unequal volumes, empty cells, padded fields and CPU/CUDA parity before connecting a solver. [Condensation validation](../docs/guide/gpu/direct/condensation.md) |
| 4. DP-SCREAMv1 on NVIDIA | Reproduce an unmodified small BOMEX case using the published script and save a control. Build a minimal EAMxx atmosphere process that calls Particula’s EAMxx adapter, with Python enabled and a Particula-compatible interpreter. Pass one strided device input to Warp and write a distinct diagnostic output, verified by the host. Then register the test tracers in host transport and restart, run passive and active experiments, and compare 1-rank and multi-rank budgets. Confirm the case's forcing, nudging and boundary sources for each budget. [BOMEX script](https://github.com/E3SM-Project/scmlib/blob/master/DPxx_SCREAM_SCRIPTS/run_dpxx_scream_BOMEX.csh) · [DP run guide](https://github.com/E3SM-Project/scmlib/wiki/Running-the-DP-SCREAM-Configuration) |
| 5. Global EAMxx | Reuse Particula’s EAMxx adapter, the compiled atmosphere process and the tracer names in a short, affordable global atmosphere-only case. Test transport across cubed-sphere face and MPI boundaries, restart, and MPI repartition. Compare mass-weighted global inventories and vertical profiles. Measure incremental bytes per cell, transport cost per tracer, adapter time and total step time against a host-only control. For B bins, S particulate species and G gases, a number-and-species-mass representation requires B × (1 + S) + G transported fields. Use measured costs to choose bin/species counts and resolution before a global storm-resolving run. |

For each numerical comparison, set tolerances from precision, independent reference error and timestep refinement before evaluating the coupled result. Budget forcing and boundaries explicitly. Record the code revisions, device, configuration and observed transfers. The OpenFOAM replay supplies the independently documented single-domain forcing export. Live two-way transport begins with a passive tracer pulse crossing a periodic boundary in the host solver.
