# Native scientific process calls

E9-M3 connects the native `Aerosol` aggregate to explicit CPU scientific
strategies. The aggregate owns `ParticleData`, full-order `GasData`, and
`EnvironmentData` references; strategies own physical configuration and RNG.
The integration regression is
`particula/integration_tests/native_scientific_processes_test.py`. It builds
through the M2 public initializers/builders and forbids facade construction.

## Representation and inventory

| Representation | Population storage | Species inventory in kg | Volume |
|---|---|---|---|
| `discrete` | Bin number density, m^-3 | Sum of per-particle mass times bin density, times V | Exactly 1 m^3 |
| `continuous_pdf` | Radius PDF dn/dr, m^-4 | Radius trapezoidal integral of mass times PDF, times V | Exactly 1 m^3 |
| `particle_resolved` | Raw representative counts | Sum of per-particle mass times count | Positive physical V |

Per-particle species mass is kg. Resolved number density is count/V. Gas
concentration is kg/m^3 in complete gas order, including inert lanes. CPU
strategies currently require one box. Container support for more boxes does
not imply multi-box strategy execution.

## Explicit calls and configuration

Pass `aerosol.particles`, `aerosol.gas`, and scalar values from
`aerosol.environment` to the configured strategy:

```python
temperature = float(aerosol.environment.temperature[0])
pressure = float(aerosol.environment.pressure[0])

condensation.step(
    aerosol.particles, aerosol.gas, temperature, pressure, time_step
)
coagulation.step(aerosol.particles, temperature, pressure, time_step)
wall_loss.step(aerosol.particles, temperature, pressure, time_step)
```

These names denote configured process objects, not aggregate methods or a
new orchestration API. Each call preserves its supplied container and primary
array identities. Preflight validates current mutable state before publication;
CPU stochastic draws are not rolled back after consumption. Three successive
calls are three separate transactions, not a whole-sequence rollback guarantee.

Condensation requires an explicit immutable process-owned map, for example
`CondensationSpeciesMap(("inert", "B", "A"), ((2, 0), (1, 1)))`. The map binds
gas A to particle lane 0 and gas B to lane 1. Transport, vapor-pressure and
latent-heat parameters follow **full gas order**; activity/surface composition
follows full particle order. Skip indices refer to gas lanes. Partitioning
remains dynamic; unmapped, disabled and skipped lanes do not transfer.
See [condensation contracts](condensation_strategy_system.md).

Native condensation supports PMF/PDF and arbitrary nonnegative resolved
weights. Its finite-stage limiter accounts for actual representable mass
changes, evaporation, available gas and PDF quadrature. Staggered modes retain
their existing sequential current-gas coupling. Disabling gas updates models
a fixed reservoir and does not establish closed-system conservation.

Complete evaporation retires a resolved particle by clearing its count and
charge after gas/energy accounting. A remaining nonvolatile or unmapped core
keeps the particle active. The resulting free slots are valid for subsequent
coagulation and wall loss; gas-phase charge transport is not modeled.

Native coagulation supports the existing CPU mechanisms, including turbulent
DNS and the charged strategy variants. Resolved merging requires **equal
positive active counts within a box**; unequal weights reject before selection.
Density steps preserve the existing number-density update while keeping per-bin
species masses and charge fixed. Their intended use is uniform composition
across radius or a single effective material with prescribed effective density,
rather than tracking radius-dependent composition. Effective density alone
does not provide per-species transport. When composition varies with size,
individual species and weighted charge need not be conserved, even if total-mass error
converges under grid refinement. This accepted limitation is characterized by
independent tests; conservative species transport is deferred. See
[coagulation contracts](coagulation_strategy_system.md).

Choose `particle_resolved` to track species composition that varies with radius
or more complex interactions involving individual particles' composition,
size, and charge. Resolved particles retain their own species masses as they
evolve and merge. Use the supported process models and their constraints
described here; resolved coagulation requires equal positive active counts.

Native CPU wall loss retains forward Euler for PMF/PDF and Bernoulli whole-
representative removal for resolved counts, including unequal weights.
Account for removed inventory as an explicit sink. M4 native CPU resolved removal
clears charge together with mass/count, so removed slots satisfy the activation
free-slot contract. Surviving and untouched inactive storage is preserved. This
amends M3 charge retention without changing the CPU removal law. GPU removal
also clears charge. See [wall-loss contracts](wall_loss_strategy_system.md).

## Direct/prepared Warp boundary

Transfers are explicit and preserve raw values and supported representation
tags. Direct condensation supports discrete density and resolved counts with
an explicit species binding; direct coagulation and wall loss support resolved
counts. Tagged PDF execution rejects. Coagulation retains its eleven supported
mechanism masks and equal-count restriction; no GPU DNS or unsupported mask
support follows from metadata.

Configuration and ownership checks happen during preparation. Frozen prepared
calls retain references and scalar semantics; callers must keep their payloads
valid and re-prepare after layout, mapping or semantic changes. Enqueue does
not revalidate, upload, allocate, read back, synchronize or rebind. Wall-loss
prepared calls also retain the explicit kernel reset flag: persistent replay
requires initialized caller-owned RNG and `initialize_rng=False`.

## Integration and migration boundary

The native integration suite verifies condensation deltas separately in
particles and mapped gas, actual resolved coagulation conservation at
V=.25/1/4, the retained fixed-composition density update and its characterized
limitation, and wall-loss sinks. It keeps gas
and environment protected across particle-only processes. The owning unit
suites provide the wider mechanism, control, rejection and stochastic matrices.

M4 owns high-level runnable composition, execution adapters, dilution/nucleation
orchestration and resident/checkpoint schema integration. Their unsupported
tagged-state boundaries currently reject. M5 owns broad tutorial/notebook
migration. Temporary facades and untagged dispatch remain for those consumers;
M6 owns deletion after downstream migration. This milestone does not authorize
a release or establish resident/GPU performance claims.

## Native source integration (M4)

`Nucleation(source_config, commit_config)` reads the currently held native
`Aerosol.gas` and `Aerosol.environment` on each equal substep. Its source config
requires an explicit `NucleationSpeciesMap` from
`particula.dynamics.nucleation.species_mapping`: full expected gas names and
unique `(gas_index, particle_index)` pairs. Gas and particle species widths may
be different. Positive injection composition entries must map to a particle
lane and have partitioning enabled; unmapped gas and particle-only species are
preserved. Native CPU source commits require one resolved box.

Potential rates and immutable finalization records retain number-density units.
At commit, admitted event density is multiplied by current physical volume.
`maximum_slot_weight` limits represented **counts** per newly activated slot.
Native final diagnostics expose `event_count_units="count"`; gas transfer and
mapped conservation residuals remain mass densities. Representative-volume
scaling reduces existing counts and provisional source counts with volume;
it preserves pre-existing particle density and gas concentration. Physical
volume evolution is a separate operation. Resampling retains count-weighted
species mass and charge under its existing bounded moment checks.

Each source attempt stages before writing, retains array identities, and
restores attempted particle/gas writes if a CPU writer fails. Earlier committed
substeps remain applied when a later attempt fails. The small
`docs/Examples/Nucleation/cpu_nucleation.py` example exercises this native API
at non-unit volume with inert gas in the unified container. Transitional
`LegacyNucleation`, available only in `particula.dynamics.particle_process`,
retains the old constructor environment and facade behavior for compatibility
tests until M6; it is not exported from `particula.dynamics`.

The direct GPU `nucleation_step_gpu` requires a separately prepared
`NucleationSpeciesBinding` for tagged resolved storage. Create it with
`prepare_nucleation_species_map` from
`particula.gpu.nucleation_species_mapping`, supplying retained full gas names
and the explicit source map. Preparation uploads only lane configuration and
pins exact carrier arrays; replacing storage or metadata requires preparation
again. Full-gas P2 outputs use gas width; resampling scratch uses particle width.

GPU source rates/final demand are densities. The existing direct algorithm
requires admitted density times volume to be an integral int32 event count and
activates one unit-count slot per event. It does not adopt the CPU weighted-slot
packing algorithm. Native representative scaling acts on count demand and
volume once, preserves gas concentration, and restores density units before
final gas removal. Prepared repeated enqueue performs no allocation, transfer,
readback or synchronization; device validation gates invalid live state before
primary writers, with explicit observation afterward. Standalone GPU
resampling and representative scaling likewise admit only resolved native
storage, retaining count-weighted moments or count/volume density respectively.

### Native direct communication and physical volume

The concrete GPU communication primitives accept tagged resolved counts.
Particle transport moves the same represented count out of the source and into
its destination, including unequal-volume boxes. Fixed-capacity planning retains
its exact population matching, ascending free-slot allocation and gated commit.
Count-weighted species mass and signed charge are conserved. PMF/PDF particle
transport is unsupported.

`volume_evolution_step_gpu` prescribes physical volume: native particle counts,
per-particle masses and charge remain unchanged, while every gas density scales
by the old/new volume ratio. This differs from representative-volume exhaustion,
which scales native counts and volume together while preserving gas density.
Invalid volumes or unrepresentable count densities gate primary writes. Callers
retain device placement, transfers and synchronization; prepared enqueue uses
pinned storage without allocation or host readback.

## Resident native metadata and diagnostics

Resident setup accepts tagged PMF and resolved populations and preserves their
raw storage. It pins the population type, complete gas names, and separate
particle-material and gas widths. Registry validation and capture signatures
reject changed semantics or bindings; dispatch does not infer or rebind maps.
Condensation and nucleation requests require their explicit uploaded mappings.
Gas updates, thermodynamics and gas-shaped scratch use the full gas width.

Native number diagnostics report number density (resolved counts divided by
physical volume). Total-species-mass diagnostics require an explicit
`CondensationSpeciesMap` in `ResidentDiagnosticRegistration.species_map` with
the session's gas order and every particle material mapped. Outputs are
extensive kg in gas order: gas concentration times volume plus mapped particle
inventory. Unmapped particle-only material is supported by process calls, but
this bounded gas-order total-mass diagnostic rejects it rather than dropping
its inventory. Two-snapshot diagnostics do not require a mass map.
