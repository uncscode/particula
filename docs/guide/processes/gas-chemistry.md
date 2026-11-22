# Define a gas reaction mechanism

Use a native mechanism when a gas can react even though it does not condense.
Each reaction specifies the molecules consumed and produced and the kinetic
law that advances it. A binding connects these names to the gas columns in
your aerosol. The same mechanism and binding drive NumPy and resident CUDA.
The [chamber pulse example](../../studies/examples/custom-processes/native_gas_chemistry.py)
contains a complete two-box simulation.

## Write and bind a mechanism

Save this as `combine.yaml`. It describes a closed bimolecular reaction.
The example values are synthetic.

```yaml
version: 1
basis: mol/m^3-air
species:
  gas:A: {}
  gas:B: {}
  gas:C: {}
reactions:
  - id: combine
    reactants: {gas:A: 1, gas:B: 1}
    products: {gas:C: 1}
    rate: {type: constant, k: 2.0, units: m^3/mol/s}
```

Given an existing `aerosol` with gas names `precursor`, `B` and `C`, bind
the precursor alias. Gas molar masses come from the simulation in kg/mol.
For this reaction C must have the combined molar mass of A and B.

```python
from particula.dynamics.chemistry import ReactionMechanism
from particula.dynamics.chemistry.process import ChemistrySolver, GasChemistry

mechanism = ReactionMechanism.load("combine.yaml")
bound = mechanism.bind(aerosol.gas, {"gas:A": "precursor"})
solver = ChemistrySolver(method="rk23", rtol=1e-7, atol=1e-14)
chemistry = GasChemistry(bound, solver)
chemistry.execute(aerosol, time_step=1.0)
```

`ReactionMechanism.from_dict(mapping)` accepts the same records in Python.
`to_dict()` returns an independent normalized mapping. Use
`save("combine.json")` or `save("combine.yaml")` to share a mechanism.
The suffix selects the format. The reader rejects duplicate keys, unknown
versions, unrecognized reaction fields and unsupported rate units.

A gas ID matches the exact name after `gas:` unless the study supplies an
override. Each ID binds to one distinct existing column. Rebind after
reordering columns or changing a bound molar mass, and prepare a new GPU
owner. Concentrations can change during an ordinary run. Give each species
an optional `elements` mapping such as `{"H": 2, "O": 1}` for elemental
balance checks on reactions with complete compositions. Closed reactions
always undergo molar mass balance validation.

## Import an NCAR mechanism

Convert a single gas phase in NCAR Mechanism Configuration 1.3.0 to a
native file. The [gas example](../../studies/examples/custom-processes/native_gas_chemistry.py)
download bundle includes both the A + B → C input and its converted JSON.
Run from the extracted bundle or repository root to reproduce the conversion.

```bash
python -m particula.dynamics.chemistry.ncar docs/data/gas-chemistry/combine-ncar.yaml combine-native.json
```

Load the output with `ReactionMechanism.load` and bind it as above.
When the source supplies molecular weights for all species with a net gain
or loss, the converter checks reaction mass balance before writing. Binding
checks the supplied weights in kg/mol against the simulation gas columns. This converter supports irreversible Arrhenius reactions
with one or two distinct unit-coefficient reactants, positive integer product
coefficients and ordinary evolving species. It accepts either `C` in kelvin or
`Ea` in joules per molecule. Pressure-dependent rates, fixed reservoirs,
other reaction types and unrecognized fields produce an error before export.
[NCAR rate definition](https://github.com/NCAR/MechanismConfiguration/blob/2069f9c97f08399b34ca3dcb104d3a5bf596ce40/docs/source/v1/reactions/arrhenius.rst)

## Rates, units and temperature

For reaction $r$ the molar extent rate per volume of air is
$q_r=k_r\prod_j c_j^{o_{rj}}$, where $c_j$ is mol/m³ and $o_{rj}$ is
the kinetic order. The species tendency sums products minus reactants times
$q_r$ over the mechanism. The process converts the native gas concentrations
in kg/m³ to molar units before integration and converts back on publication.

Stoichiometric coefficients are positive integers. The optional reaction
mapping `orders` overrides individual default reactant orders with
nonnegative integers. An additional rate dependency can have a positive
order and zero net stoichiometry. The rate units follow the sum of the orders.

| Total order | Coefficient units in the native file |
| --- | --- |
| 0 | `mol/m^3/s` |
| 1 | `1/s` |
| 2 | `m^3/mol/s` |
| 3 | `m^6/mol^2/s` |

For temperature dependence, replace the constant rate with an Arrhenius
record. For example, a second-order reaction may use
`{"type":"arrhenius","A":2,"b":1,"Ea_J_per_mol":1000,"T_ref_K":298,"units":"m^3/mol/s"}`.
The rate coefficient is $A(T/T_{ref})^b\exp[-E_a/(RT)]$. Each box supplies
its own positive temperature in K, held fixed during its chemistry interval.

## Integration, failure and restart

Use `rk23` for nonstiff cases. The `ros3` option uses the analytical
mass-action Jacobian and a pivoted dense solve for stiff networks. Both
support scalar or per-species absolute tolerances in mol/m³. A tolerance
tuple follows the sorted mechanism IDs in `mechanism.species`. The initial,
minimum and maximum steps are in seconds. `max_trials` bounds the sum of
accepted and rejected trials per box per interval. Negative trial inventories
trigger step reduction. Choose the interval and tolerances using independent
trajectory and budget checks for the mechanism. [Numerical integration](numerical-integration.md)

For CUDA, register the adapter explicitly. This example starts from whatever
state is in `aerosol` at preparation.

```python
from particula import ProcessRegistry
from particula.dynamics.chemistry.adapter import GasChemistryAdapter

registry = ProcessRegistry()
registry.register(GasChemistry, GasChemistryAdapter())
with chemistry.prepare(
    aerosol, device="cuda:0", time_step=1.0, process_registry=registry
) as simulation:
    simulation.step()
    saved = simulation.checkpoint()

with chemistry.resume(saved, process_registry=registry) as simulation:
    simulation.step()
    result = simulation.snapshot()
```

Preparation uploads the compiled rates, orders, stoichiometry and indices.
Per-box trial states stay on device. With $n$ participating species, ROS3
uses two dense $n\times n$ matrices per box. The adapter checks its
`workspace_bytes` limit before allocation (default 256 MiB for scratch).
Checkpointed work counts are accepted trials, rejected trials, factorizations
and triangular-system solves, including residual corrections. The device
latches failure 100 for an unsuccessful interval and 101 for a nonfinite
rate coefficient. A synchronization, observation or checkpoint reports a
latched failure. A separate publication kernel commits gas only after all
boxes complete the chemistry operation.

Each process interval starts a fresh adaptive controller. NumPy continuation
uses `CPUSimulation.checkpoint()` and `CPUSimulation.resume()`. Prepared
CUDA continuation requires the same device, compatible mechanism, binding,
solver, and adapter. Both profiles resume at completed outer steps. The
[qualification record](../../studies/validation/condensation/gas-chemistry.md)
covers irreversible gas reactions at prescribed temperature. For bulk
particle reactions and shared gas uptake, see the [resolved chemistry
guide](resolved-chemistry.md).

## Search for a hidden intermediate

Does conversion from A to P require a delay while an intermediate accumulates?
A SiMBA-style search proposes reaction networks, and their predicted
concentration histories can be compared with observations. Give each species a
fixed column and supply the elementary steps allowed by the physical hypothesis.
Here A, I, and P are hypothetical isomers with identical molar masses. The
three rows below specify direct conversion and the two steps of a delayed route.
[SiMBA method](https://doi.org/10.1039/D5SC01473E)

```python
from particula.dynamics.chemistry.discovery import compile_network, enumerate_networks

catalog = {
    name: {"molar_mass_kg_per_mol": 0.1, "elements": {"X": 1}}
    for name in ("gas:A", "gas:I", "gas:P")
}
search = enumerate_networks(
    [(-1, 0, 1), (-1, 1, 0), (0, -1, 1)],
    initial=(True, False, False),
    products=(2,),
    net=(-1, 0, 1),
    max_steps=2,
    max_trials=10,
    max_candidates=10,
)
mechanisms = [
    compile_network(network, catalog, rates=[0.5] * len(network))
    for network in search.networks
]
```

The search returns two mechanisms. Each supplied rate here is an initial
first-order coefficient in s⁻¹. Bimolecular coefficients use m³ mol⁻¹ s⁻¹.
The catalog's insertion order defines the columns when compiling. For a known
overall reaction, `net` requires the selected rows to sum to its stoichiometry
with each step counted once. Omit it to explore competing branches. For a larger
bank, `balanced_steps(phases, composition)` generates single-phase steps with
one or two molecules on each side and a conserved element or pool budget.
Use `interchangeable` only for hidden species with equivalent physical
properties. Record `search.examined` and `search.limit` alongside fitted
results. A trial or candidate limit means eligible alternatives may remain.

For a fixed candidate, wrap each run's objective in an `ExperimentLoss` and
call `fit_experiments` from `particula.dynamics.chemistry.inference`. Reuse
a parameter index across runs to share a rate, and assign a different index
to each run-specific dose. Supply finite physical bounds and feasible starting
vectors. The fitter minimizes summed loss with bounded multistart L-BFGS-B
and accepts an analytical or AD gradient, or three-point finite differences.
Equal lower and upper bounds fix a coefficient for a profile or a zero-rate
control. Compare candidates with the same observations, error model, and refit
budget. Inspect each start's convergence and retain observational ties.
If an experiment measures only total conserved mass, every rate in both
example networks gives the same prediction. Time-resolved P adds information
about the lag. [Worked hidden-intermediate comparison](../../studies/examples/custom-processes/discover_hidden_intermediate.py)

For a coupled chamber, pass gas–particle column pairs to `exchange_pairs`.
A particle product can then age after its vapor precursor condenses, even
when its initial inventory is zero. Set `max_intermediates` to bound how
many noninitial, nonproduct columns a candidate can use.

To rank a search result, use `fit_networks(search, prepare, criterion=...,
tolerance=...)` from the inference module. The supplied `prepare(network)`
returns a `CandidateFit` with training experiments, bounds, starts and any
held-out experiments. With `criterion="held_out"`, choose the tolerance in
held-out loss units. With `criterion="aic"`, supply comparable negative log
likelihoods and choose the tolerance in AIC units. Each result carries its
score, gap, fitted parameters and competitive flag. Inspect all competitive
networks and the search's stopping reason before assigning an intermediate.
