# Architecture Design

## High-Level Design

The shipped P1 strategy computes a scalar potential event rate only. Immutable
configuration records carry a closed scientific domain, future composition
metadata, and formation metadata; no finalizer, process, or state write exists.

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
  and `FormationMetadata`.
- **API Surface:** Concrete symbols live only in
  `particula.dynamics.nucleation.nucleation_strategies`; the package and
  `particula.dynamics` do not re-export them.
- **Mutation Contract:** Evaluation returns a `float` potential rate and does
  not mutate caller state.
- **Workflow Hooks:** E6-F5/E6-F6 consumption, E6-F8 parity, and E6-F9
  integration are future work.

## Security & Compliance

There are no network, permission, or persistence changes. P1 scientific safety
uses citations, units, explicit validity domains, finite validation, and
failure-before-mutation behavior. Documentation must not claim general
atmospheric predictiveness, inventory/conservation support, unimplemented
Vehkamäki/CNT physics, GPU parity, dynamic capacity, hidden transport, or
performance evidence.
