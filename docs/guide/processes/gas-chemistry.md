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

## Select chemistry from an atmospheric model

Use the reaction catalog when you need a published equation and rate law in a
Particula box. For example, a study of ozone loss by NO can use one WACCM
reaction from a whole-atmosphere mechanism. Fetch the catalog from the matching
Particula GitHub Release, retrieve a named source recipe, or import a local
model case. Particula keeps the original bytes and the searchable
records in a local SQLite catalog. The Python API and CLI use the same catalog.

### Choose the source files

| Import key | Source files and scientific choices |
| --- | --- |
| `waccm` | Use the `chem_mech.in` from your case or the pinned TSMLT1 mechanism. Some rates refer to CAM routines, and photolysis requires a light field. [TSMLT1 mechanism](https://github.com/ESCOMP/CAM/blob/405a3f21446439d8f952660f0300332d9b860279/src/chemistry/pp_waccm_tsmlt_mam4/chem_mech.in) |
| `e3sm` | Choose a tagged EAM chemistry option. The v3.0.0 chemUCI–Linoz v3–MAM5–VBS file describes both molecular channels and yields into volatility classes. Keep the prescribed oxidants and aerosol treatment associated with that option. [E3SM mechanism](https://github.com/E3SM-Project/E3SM/blob/v3.0.0/components/eam/src/chemistry/pp_chemuci_linozv3_mam5_vbs/chem_mech.in) |
| `mcm` | In York's exporter, select the emitted precursors, inorganic chemistry and generic rates needed for your study. Download the KPP equations and retain the corresponding rate definitions and species table. Reactions that reference RO₂ use a sum of concentrations from the selected mechanism. [York exporter](https://mcm.york.ac.uk/MCM/export) · [Versioned archive](https://www.mcm.york.ac.uk/MCM/about/archive) |
| `wrf-chem` | Select the KPP equation file for the specific gas chemistry and aerosol configuration, such as MOZART with four-bin MOSAIC in v4.6.1. The mechanism includes comments about species omitted from the active equations and calls to WRF-supplied rate functions. [WRF-Chem equations](https://github.com/wrf-model/WRF/blob/v4.6.1/chem/KPP/mechanisms/mozart_mosaic_4bin/mozart_mosaic_4bin.eqn) |
| `cmaq` | Use the named mechanism's definition file. For example, CRACMM2 uses lumped organics, diagnostic species and rates that refer to other expressions. Preserve the source species names when deciding which channels can close a gas budget. [CMAQ 5.5 CRACMM2](https://github.com/USEPA/CMAQ/blob/5.5/CCTM/src/MECHS/cracmm2/mech_cracmm2.def) |
| `camx` | Match the versioned chemistry-parameter file to the gas mechanism selected in the model. The recipe pairs the EPA rate deck with the CAMx 7.20 CMC gain/loss and flux tables from a pinned source mirror. Local imports accept the matching pair or a numbered `mech.*` equation listing via `--companion`. Particula recognizes the ozone titration entry in the exact EPA CB6r5 parameter file below. [CMC source mirror](https://github.com/wcd940816/camx7.2modified/tree/dc4351b4dd5499cc53fc230317875ab831772473/src.v7.20/CMC) · [EPA CB6r5 parameters for CAMx 7.2](https://epa-2022-modeling-platform.s3.amazonaws.com/Model_jobs_v2/2022he_CAMx_base_12US2/CAMx7.2.chemparam.CB6r5_CF2E_NH3Rscale0) · [CAMx guide, §3.1](https://camx.com/Files/CAMxUsersGuide_v7.20.pdf) |
| `pmcamx` | Select the source archive for the biomass-burning aerosol treatment used in your case. Import `CAMx4.chemparam.7.CB5` for gas rates or `SOAP/soachem.f` as the primary file to inspect precursor oxidation, low/high-NOₓ yields and volatility-bin aging. Recipes `pmcamx-soap` and `pmcamx-soap-alt` select the base and alternative VBS schemes. The SOAP records require the host’s prescribed oxidants, yield interpolation and partitioning assumptions to calculate organic aerosol formation. [Archived PMCAMx-SR sources](https://doi.org/10.5281/zenodo.4071362) · [Comparison of biomass-burning schemes](https://doi.org/10.5194/gmd-14-2041-2021) |

For a local CAMx or PMCAMx gas deck, inspect `metadata.source_rate_units`
in the reaction record. The pinned decks use molecular concentrations and
seconds. Local decks can declare `molecular units` or `ppm units` as a
distinct clause in the Description header. For a bimolecular reaction these
mean cm³ molecule⁻¹ s⁻¹ and ppm⁻¹ min⁻¹, respectively. The catalog preserves
declared ppm rates for inspection. Native export requires an implemented
conversion. If the evidence is missing or conflicting, the concentration and
time units are null in the record and the study lock.
[CAMx 7.20 guide, §3.1](https://www.camx.com/Files/CAMxUsersGuide_v7.20.pdf)

### Export a selected gas reaction

Run `particula chem fetch` to download the prebuilt catalog for the installed
Particula version. Python users can call `ChemistryCatalog().fetch()`.
The download includes E3SM, CMAQ and WRF-Chem. Particula verifies the ZIP’s
SHA-256 checksum and merges its snapshots with your local imports. Search,
inspection and export then work offline. Run `particula chem fetch --json`
to see the included snapshot IDs, input digests and source citations.
For a package release that predates the catalog asset, fetch a named recipe.

To add WACCM from the pinned upstream files, fetch its recipe by name. The other recipe names are
`e3sm`, `mcm` (α-pinene), `mcm-full`, `wrf-chem`, `cmaq`, `camx`, `pmcamx`,
`pmcamx-soap` and `pmcamx-soap-alt`. You can set
`PARTICULA_CHEM_CATALOG` to a specific SQLite path. By default the file lives
in the user's cache directory under `particula/chemistry.sqlite3`.

```bash
particula chem fetch waccm
particula chem search --source waccm --species NO --role reactant --json
```

For a local case, use the import command below.

```bash
particula chem import waccm chem_mech.in --release MY_TAG --mechanism MY_CASE
```

Record the origin of local inputs with the
study. A snapshot ID pins the host, release, mechanism, optional aerosol
scheme, input digests and importer version. Use `--snapshot ID` to search just
that import. Pass a snapshot ID to export to lock its complete inventory.

Choose the result for `NO_O3`, the reaction NO + O₃ → NO₂ + O₂. Copy its
exact ID from the search results into `REACTION_ID` below. Inspect the
record before exporting, particularly when another source has a different
Arrhenius coefficient or assumes that a product belongs to a background
reservoir.

```bash
REACTION_ID='replace-with-the-returned-ID'
particula chem show "$REACTION_ID" --json
particula chem export "$REACTION_ID" --target lock --output ozone-loss.lock.json
particula chem export --lock ozone-loss.lock.json --target native --output ozone-loss.json
```

In Python, the WACCM source label can be appended directly to the ID returned
by the recipe. This selects the same reaction even when other snapshots have
been imported.

```python
from particula.dynamics.chemistry.catalog import ChemistryCatalog

catalog = ChemistryCatalog()
snapshot = catalog.fetch("waccm")
catalog.export([f"{snapshot}/NO_O3"], "ozone-loss.json")
```

Keep both files with the study. The lock identifies the source selection,
and the native JSON loads offline in a compatible Particula installation.
For a kinetic check, give an aerosol box gas columns named `NO`, `O3`,
`NO2` and `O2`. Start NO and O₃ at 1 µmol/m³ each by setting each reactant's
concentration in kg/m³ to `1e-6 * molar_mass_kg_per_mol`. Set the product
concentrations to zero and the box temperature to 298 K. With the classes
imported in the binding example above, advance the gas chemistry alone.

```python
mechanism = ReactionMechanism.load("ozone-loss.json")
solver = ChemistrySolver(method="rk23", rtol=1e-8, atol=1e-16)
GasChemistry(mechanism.bind(aerosol.gas), solver).execute(aerosol, time_step=60.0)
```

After 60 s, each reactant should have 0.586070 µmol/m³ remaining.
The analytical bimolecular solution is $c(t)=c_0/(1+k c_0 t)$ with $k$ in
m³ mol⁻¹ s⁻¹. [Original WACCM rate](https://github.com/ESCOMP/CAM/blob/405a3f21446439d8f952660f0300332d9b860279/src/chemistry/pp_waccm_tsmlt_mam4/chem_mech.in#L785)

Inspect dependencies before expanding a single reaction into a pathway.
A photolysis rate needs a channel-specific frequency in s⁻¹. A volatility
class needs a partitioning model to predict particle mass. A hidden oxygen
reservoir can supply atoms that a closed native mechanism must account for.
For the reviewed NO + O₃ channel, `--close-oxygen` at lock creation or direct native export explicitly
includes O₂ in the closed gas inventory when the host treats it implicitly
or as a prescribed species. Provide its gas column when binding the result.
Retain such source records for comparison and check their export diagnostics
before selecting chemistry for a run. To trace a supported gas mechanism
into the implementation with CodeGraph, explore `ReactionMechanism.bind`
and `GasChemistry.execute`.

From a source checkout, run `python -m scripts.assess_chemical_catalog` to
check all seven source families at 250, 298 and 320 K and compare the two
PMCAMx-SR biomass-burning aging coefficients.

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
