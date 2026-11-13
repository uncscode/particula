# Native aerosol construction and replacement

This guide describes native construction used by the supported migrated
examples. Native CPU processes and runnables consume these held containers;
see [dynamics migration](../migration/v0.3.0/dynamics.md) for configuration and runnable batching.
The [access walkthrough](ownership.md) demonstrates executable replacement and
rejection checks. See also [container foundations](index.md).

## Assemble three containers

```python
import particula as par

particles = par.particles.create_resolved_particle_data(
    masses=[[0.5e-18, 1.5e-18], [2e-18, 3e-18]],
    density=[1000.0, 1200.0],
    concentration=[8.0, 0.5],  # represented counts, including fractional weights
    volume=0.25,  # physical m^3
)
gas = (
    par.gas.GasDataBuilder()
    .set_names(["water", "inert", "organic"])
    .set_molar_mass([0.018, 0.028, 0.1])
    .set_partitioning([True, False, True])
    .set_concentration([1e-8, 7e-8, 3e-8])  # kg/m^3, in full gas order
    .build()
)
environment = par.gas.EnvironmentData(
    temperature=[298.15], pressure=[101325.0],
    saturation_ratio=[[0.7, 7.0, 1.3]],
)
aerosol = par.Aerosol(
    particles=particles, gas=gas, environment=environment,
)
assert aerosol.particles is particles

# Equivalent fluent construction; repeated builds share supplied containers.
builder = (par.AerosolBuilder().set_particles(particles)
           .set_gas(gas).set_environment(environment))
second = builder.build()
assert second.gas is gas
```

The constructor is keyword-only. All three builder inputs are required; no
containers or physics configuration are synthesized. Builder setters retain
references, and `build()` validates through the constructor. Optional `*_units`
builder-protocol arguments are ignored with a log warning because these are
already constructed containers, not quantities to convert.

Admission first checks container types, array structure, compatible box counts,
unique/nonblank gas names, and full gas/environment width; it then calls
`ParticleData.validate_representation()` for the explicit kind, float64 schema,
finite/nonnegative particle values, material density, volume and PDF grid.
All-false gas masks and unequal particle/gas species counts are valid. No gas
lane is dropped. Chemical mapping and gas/environment ordering provenance stay
process-owned: unnamed same-width permutations cannot be detected here.

## Replace state without partial publication

```python
# Compatible individual replacements retain candidate identity.
aerosol.particles = particles.copy()
aerosol.gas = gas.copy()
aerosol.environment = environment.copy()

# All three arguments are required, including for a coordinated layout change.
aerosol.replace_data(
    particles=particles, gas=gas, environment=environment,
)
```

Each operation validates the complete proposed triple before publishing any
references, including same-object assignments. Invalid candidates leave both
old and candidate state unchanged. Coordinated replacement can change box or
gas/environment widths when individual replacements would be incompatible.
It returns `None`; it selects caller-supplied state and does not rescale it to
conserve an old inventory. Validation atomicity is not a threading guarantee.

Reads return the actual held objects. Raw arrays remain writable, so direct
mutations are immediately visible and require subsequent validation before
interpretation. Callers request detached data explicitly through container
`copy()` methods. Derived particle population properties compute fresh values.
Replacing CPU state does **not** refresh/rebind prepared or resident GPU state;
its owners retain their original bindings and must explicitly recreate or
reprepare them through their existing APIs.

## Initialize particles without facades

All four functions are exported from `particula.particles` and return owned,
writable float64 arrays. They use the existing `ParticleDataBuilder` for final
admission. Existing native builder ownership and storage units are unchanged.

| Function | Inputs and storage |
|---|---|
| `create_particle_data_from_radii` | Single material, positive radii and scalar/per-node density weights; explicit `discrete` or `continuous_pdf`; mass = 4πρr³/3 |
| `create_lognormal_binned_particle_data` | Existing lognormal PDF/PMF generators; explicit kind; scalar/per-node charge |
| `create_resolved_particle_data` | Scalar or `(N,)` single-species mass, `(N,S)` or `(B,N,S)` mass; explicit density, optional counts, charge and physical volume |
| `create_lognormal_resolved_particle_data` | Existing sampled mixture, single material, one count per sampled slot |

Direct mass/species construction also remains available through
`ParticleDataBuilder`; there is no need for a facade or strategy selection.

### Distribution and volume rules

| Kind | Stored concentration | Volume | Bulk calculation |
|---|---|---|---|
| `discrete` | Number per bin per m³ | Exactly 1 m³ | Sum weights or mass × weights |
| `continuous_pdf` | Radius `dn/dr`, m⁻⁴ | Exactly 1 m³ | Trapezoids over explicit radius; integrate mass × PDF together |
| `particle_resolved` | Represented counts | Positive finite physical volume | Sum counts or mass × counts, divided by volume for density |

PDF nodes are positive, finite, strictly increasing metres with at least two
nodes. Integration uses the supplied interval without extrapolation. Zero PDFs
retain their grid. Empty discrete/resolved populations are supported; sampled
presets require a positive integer sample count. Non-unit binned volumes reject.
At initialization only, non-SI reference-volume conversion roundoff within four
float64 ulps of one is canonicalized to exactly one; direct SI admission stays
exact. Canonical SI units work without Pint. Other unit spellings and
conversions use the existing optional Pint conversion utility.

For resolved inputs, `concentration_units="count"` is the default and does not
rescale supplied weights. Explicit `"1/m^3"` or `"1/cm^3"` inputs multiply the
converted density by each box's volume once. Omitting concentration creates unit
counts and requires count units. For eight represented particles, densities at
V=0.25, 1 and 4 m³ are 32, 8 and 2 m⁻³; their extensive particle mass is unchanged.
Fractional weights are valid storage but do not imply support by every process.

### Defaults, shapes and sampling

Binned defaults are `logspace(-9, -4, 250)` metre nodes, modal scale radii
`[100e-9, 1e-6]` metres, GSD `[1.2, 1.4]`, modal densities `[1e10, 1e9]` m⁻³,
material density 1000 kg/m³, and zero charge. Supplied nodes/modes use
`radius_units`; omitted defaults remain SI. GSD must exceed one, because the
existing finite-width lognormal generator does not represent a delta function.
The existing PDF and PMF algorithms retain their normalization on the supplied
grid. Lognormal discrete PMFs require at least three positive, strictly
increasing nodes because the retained converter extrapolates its last bin
width from two intervals. Lognormal PDFs require two nodes. Direct radius
initialization still accepts two discrete bins with caller-supplied weights.
Binned kind is always explicit; legacy `pdf`/`pmf` aliases are not used.

Sampled defaults preserve 20,000 samples, volume 1 m³, and the same modal and
material defaults. Modal densities control relative sampling weights only.
Samples retain unit counts, so physical number density is sample count / volume,
not the sum of the modal input densities. The existing NumPy/SciPy global random
state is used; no new seed API or cross-backend replay guarantee is provided.
Deterministic input validation happens before sampling. Modal densities must
have a finite total: individually finite inputs such as `[1e308, 1e308]` reject
with `ValueError` before the sampler runs or global random state changes.

Resolved density is per species; charge is per particle. Scalar and singleton
forms broadcast. Redundant density population rows or charge species lanes may
reduce only when every discarded value is identical to the retained value.
Differing rows/lanes reject. Canonical `(B,N)` charge shape takes precedence;
no values are discarded in that form. Radius helpers require a single material;
use explicit species masses for mixtures. Inputs are never modified.

## Explicit execution boundaries

Native discrete and resolved particle uploads preserve their tag and raw
concentration values. Radius-PDF upload is unsupported; never strip metadata to
bypass validation. Gas names remain externally owned, and native condensation
and nucleation require explicit exact-carrier species bindings. Direct and
resident examples show their concrete preparation APIs and synchronization.

A CPU aggregate replacement cannot rebind existing device arrays or a prepared
resident request. Explicit restart uses a fresh same-device binding and schema-v5
canonical checkpoint bytes; CPU inspection is not continuation authority.
There is no implicit transfer, fallback, migration or recovery. CPU dynamics runnables iterate batched storage; direct strategies retain
their single-box contracts and existing numerical restrictions.

Legacy facades and conversion bridges have been removed from this development
tree. CPU-Warp population transfers and resident construction require explicit
supported distribution metadata; untagged checkpoint records reject before
restart setup. V0.3.0 final validation and review remain separate from release
publication.
