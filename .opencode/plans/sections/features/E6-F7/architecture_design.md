# Architecture Design

## High-Level Design

The shipped P1 strategy computes a scalar potential event rate. Shipped P2
consumes that already survival-adjusted rate with a duration and immutable
injection composition, then produces gas-admitted source records without any
state write. Shipped P3 consumes those immutable records in a detached,
all-box transaction before one validated caller write phase.

```text
mass concentration + molar mass + T + optional saturation + strategy
          |
validate scalar physical inputs and saturation presence/form
          |
validate float64 C conversion
          |
zero coefficient/C/survival? -> exact 0.0
          |
closed C/T checks; below-lower saturation -> exact 0.0
          |
evaluate J = A*C or K*C^2 [events m^-3 s^-1]
          |
 finite potential rate [#/m^3/s]
          |
          + P2: potential count = rate * duration
          |
          + per-event mass = molecule count * molar mass / N_A
          |
          + per-box min(participating gas / per-event mass)
          |
           + one admitted count, provisional demand, diagnostics
           |
           + P3: private ParticleData copy + E6-F6 policy resolution
           |
           + final equal-weight slots, scaled gas, conservation validation
           |
           + atomic particle/gas array write
```

`C = mass_concentration / molar_mass * N_A` is calculated with representation-
safe `np.float64` evaluation. Basic input validation and this conversion
precede zero paths; zero paths bypass only domain membership. Concentration and
temperature intervals are inclusive. A configured saturation below its lower
bound is an exact zero gate, while above its upper bound raises `ValueError`.

## Scientific Contract

- Activation: `J=A*C`; kinetic: `J=K*C^2`. `C` is precursor number
  concentration explicitly converted from `kg/m^3` using molar mass and
  Avogadro's constant. Coefficient units are explicit and normalized to SI.
- Inputs are finite/nonnegative and configured validity intervals are closed.
  Out-of-domain calls raise rather than extrapolate. Zero coefficient,
  precursor, survival, or an unsatisfied configured saturation gate is a no-op.
- Injection composition is nonnegative with at least one positive molecule
  count. Formation diameter is metadata checked against the documented
  convention; no hidden growth occurs.
- The empirical forms follow Kulmala et al. (2006) and Seinfeld & Pandis (2016).
  Vehkamäki et al. (2002) is context, not an implemented parameterization. A
  supplied survival factor may represent Kerminen & Kulmala (2002); it is never
  inferred silently.

## Data / API / Workflow Changes

- **Data Model:** No `ParticleData` or `GasData` schema change. Frozen records
   are `ClosedInterval`, `NucleationValidityDomain`, `InjectionComposition`,
   `FormationMetadata`, `PotentialEventData`, `SourceDemandData`,
   `SourceDiagnostics`, `ParticleSourceCommitConfig`, and
   `FinalizedSourceDiagnostics`. Record arrays are fresh, owned, and read-only.
- **API Surface:** Concrete symbols live only in
  `particula.dynamics.nucleation.nucleation_strategies` and
  `particula.dynamics.nucleation.particle_source`; the package and
  `particula.dynamics` do not re-export them.
- **Mutation Contract:** P1 returns a `float` potential rate. P2 returns only
  provisional demand and diagnostics; it validates gas read-only and does not
  mutate caller state. A participating species limits each box by the minimum
  inventory ratio; ties use the lowest original species index. P2 applies no
   slot, exhaustion, particle, or gas commit. P3 performs E6-F5/E6-F6 work
   only on a private particle copy; on success it writes existing caller
   `masses`, `concentration`, `charge`, `volume`, and gas concentration arrays.
   Representative-volume rows scale pre-existing particle and gas state before
   source removal, enforcing `particle_post + gas_post = scale * pre_total`.
- **Workflow Hooks:** P3 has shipped E6-F5/E6-F6 consumption. E6-F8 parity and
  E6-F9 integration remain future work.

## Security & Compliance

There are no network, permission, or persistence changes. P1 scientific safety
uses citations, units, explicit validity domains, finite validation, and
failure-before-mutation behavior. Documentation must not claim general
atmospheric predictiveness, inventory/conservation support, unimplemented
Vehkamäki/CNT physics, GPU parity, dynamic capacity, hidden transport, or
performance evidence.
