# React gas and particle materials in a resolved box

The [glyoxal chamber example](../../studies/literature/evaporation-and-uptake/glyoxal_chamber.py)
uses reversible condensation and particle chemistry to compare organic loss
and imidazole formation during a measured chamber dilution.

Use a version 2 mechanism to transform material inside resolved particles or
to consume a gas through irreversible reactive uptake. Every particle in a
box draws from its own box's shared gas. Homogeneous gas reactions, bulk
material reactions and uptake channels can run in the same coupled solve.
The [worked uptake and aging example](../../studies/examples/custom-processes/resolved_reactive_uptake.py)
starts from a complete aerosol and compares its history with a radius model.

## Describe the reactions and bind existing lanes

This mechanism deposits gas X as fresh material with a net collision
probability of 0.04. Fresh material becomes aged material at 0.35 s⁻¹.
The three reservoirs have the same molar mass. The synthetic element X
makes the elemental budget explicit.

```yaml
version: 2
basis: mol/m^3-phase
species:
  gas:X: {molar_mass_kg_per_mol: 0.04, elements: {X: 1}}
  particle:fresh: {molar_mass_kg_per_mol: 0.04, elements: {X: 1}}
  particle:aged: {molar_mass_kg_per_mol: 0.04, elements: {X: 1}}
reactions:
  - id: stick
    reactants: {gas:X: 1}
    products: {particle:fresh: 1}
    rate: {type: reactive_uptake, gamma: 0.04}
  - id: age
    reactants: {particle:fresh: 1}
    products: {particle:aged: 1}
    rate: {type: constant, k: 0.35, units: 1/s}
```

Save the mechanism as `aging.yaml`. Supply the actual material columns when
binding. Here column 0 holds an inert core. Gas names match exactly by
default. An optional `gas_overrides` mapping connects a mechanism gas ID to
a different simulation name.

```python
from particula.dynamics.chemistry import ReactionMechanism
from particula.dynamics.chemistry.process import ChemistrySolver
from particula.dynamics.chemistry.resolved import ResolvedChemistry

mechanism = ReactionMechanism.load("aging.yaml")
bound = mechanism.bind_resolved(aerosol, {"particle:fresh": 1, "particle:aged": 2})
chemistry = ResolvedChemistry(bound, ChemistrySolver(rtol=1e-8, atol=1e-14))
chemistry.execute(aerosol, time_step=0.5)
```

All particle species require explicit molar masses (kg/mol). The binding
checks mass balance and checks elemental balance where all participating
compositions are supplied. Gas order, molar masses, material mappings and
densities belong to the binding identity. Rebind and prepare again when
reordering or relabeling material columns. Material storage is positional.
Relabeling equal-density columns is observable only through the mapping
supplied by the caller. New product columns must already exist, and may
start at zero mass. Resolved particle counts, charge and inactive slots
retain their values.

## Concentrations and uptake

For a bulk reaction, the instantaneous material volume is the sum of
$m_j/\rho_j$ over **all** constituents of each particle. At each trial stage
the concentration of species $j$ is $m_j/(M_j V_p)$ in mol/m³ of material.
The product of the molar rate density and $V_p$ is the reaction extent in
mol/s per physical particle. Use the same constant or Arrhenius fields and
per-order rate units as in the [gas mechanism guide](gas-chemistry.md).
Homogeneous gas reactions use mol/m³ of air.

For reactive uptake the molar extent per particle is
$\gamma\,\bar c\,4\pi r^2 c_g/4$, with
$\bar c=\sqrt{8RT/(\pi M_g)}$. The spherical radius follows the current
material volume. Each channel consumes one incoming gas molecule and adds
its declared material products. The sum of channel probabilities for a gas
may be at most one. Each box debits gas by the sum of the same extents
multiplied by represented counts and divided by box volume.
[Gas-kinetic uptake definition](https://iupac-aeris.ipsl.fr/guides/NewHetIntroOct2009.pdf)

Prescribe temperature and a net probability appropriate to the application.
This law assumes well-mixed gas, negligible gas diffusion resistance and
irreversible accommodation and reaction. Explicit adsorption sites, thermal
feedback, charge-changing reactions and particle retirement require additional
models. Condensation can run in the same sequence. Map the condensable vapor
and the reactive gas to their intended physical pathways. In the qualification
case, condensation grows the particle with one vapor before a different gas
reacts on the increased surface area.

## Run, record and resume

RK23 advances a common trial state for gas and every active particle in
each box. Gas absolute tolerance is mol/m³ of air. For a particle it is
mol/m³ of that particle's **initial volume in the current process interval**.
That fixed numerical scale preserves meaning near a zero product inventory.
Physical volume and reaction rates are recomputed at every stage.

Use `CPUSimulation` for NumPy recording and checkpointing. Register the
resolved adapter to prepare CUDA. A prepared owner retains rates, trial
states and the shared inventory on its device.

```python
from particula import ProcessRegistry
from particula.dynamics.chemistry.resolved_adapter import ResolvedChemistryAdapter

registry = ProcessRegistry()
registry.register(ResolvedChemistry, ResolvedChemistryAdapter())
with chemistry.prepare(aerosol, device="cuda:0", time_step=0.5,
                       process_registry=registry) as sim:
    sim.capture()
    sim.step()
    saved = sim.checkpoint()
with chemistry.resume(saved, process_registry=registry) as restored:
    restored.step()
    result = restored.snapshot()
```

CUDA capture is optional. Each replay can take a different number of adaptive
trials inside the fixed launches. The checkpoint includes per-box accepted
and rejected trial counts under `work_counts`. The default 256 MiB scratch
limit covers six trial vectors, two slot arrays and rates per box. A budget
error during preparation reports the required bytes. A failed interval
signals device code 110, and an invalid initial geometry or rate signals
111. Concentrations and masses publish together only after all boxes finish.
[Qualification, costs and limitations](../../studies/validation/condensation/resolved-chemistry.md)

## Differentiate a candidate mechanism through a seeded chamber

For a fixed network, `SeededChamber` combines constant first- or second-order
gas and particle reactions, reversible gas exchange, deterministic wall loss,
and a finite-width size response. Use this concrete inverse model to ask how a
change in precursor dose, a reaction rate, or product volatility affects a
measured channel at a particular time. Supply a native mechanism with declared
molar masses, particle densities in kg/m³, seed **volume** fractions, and the
particle species that contribute to suspended organic mass. Each
`VaporExchange` pairs the same compound in gas and particle phases, with
diffusivity in m²/s and mean free path in m. Exchange assumes ideal mole
fraction activity, a flat interface, and unit mass accommodation.

For a configured `chamber`, the following call uses observations at the
instrument sample times in seconds. In this example the mechanism has two
gases, two first-order reactions, and one exchanging compound. Replace the
control values with those for the experiment being fitted.

```python
from particula.dynamics.chemistry.observations import ObservationSeries
from particula.execution.inverse_chemistry import simulate_chamber

series = ObservationSeries.create(sample_times, measured_values, covariance)
fit = simulate_chamber(
    chamber, sample_times,
    dose=[1e-7, 1e-9],       # mol/m³ of air, sorted gas-species order
    rates=[0.5, 0.2],        # native reaction order; both s⁻¹ here
    saturation=[2e-8],       # mol/m³, in VaporExchange order
    seed=[1e10, 1e-7, 1.4, 0.04],
    max_step=0.05, device="cuda:0", observations=series, gradient=True,
)
dose_gradient = fit.gradients["dose"]
```

The seed controls are initial total number concentration (m⁻³), median
diameter (m), geometric standard deviation, and deposition rate at 100 nm
(s⁻¹). Equal-number cohorts sample the initial lognormal size distribution.
The synthetic deposition law scales inversely with particle diameter. Deposition
removes cohort weight and records lost material in the wall inventory.

Rows of `fit.predictions` follow observation times. Columns give each selected
vapor in kg/m³ in `measured_gases` order, suspended organic mass in kg/m³,
then number per bin in m⁻³ including underflow and overflow. A Gaussian
response in natural-log diameter spreads each cohort across bin edges specified
in metres. Supply measurement covariance in these channel units, either shared
across times or separately per time. Use NaN for an unavailable channel and
omit an exactly dependent total. At each time the observed covariance block
must be positive definite. The likelihood treats different times as independent.

`fit.loss` is the Gaussian negative log likelihood. The four entries in
`fit.gradients` give its derivatives by dose, rate, saturation concentration,
and seed control in their input order. To fit vapor alone or a subset of size
channels, mask the other columns. Choose `device="cpu"` for Warp CPU.
The fixed-step midpoint solve lands on each sample time. Refine `max_step`
when a stage leaves the physical domain and check both predictions and
derivatives under refinement. The [coupled qualification and reproduction
commands](../../studies/validation/chemistry/simba-discovery/index.md)
describe the independent CPU reference, CUDA and Warp CPU checks, and
the synthetic validity range.
