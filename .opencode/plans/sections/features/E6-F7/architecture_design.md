# Architecture Design

## High-Level Design

The strategy computes a potential event rate only. A pure finalizer turns that
rate into a gas-feasible source record. The process then asks E6-F5 and E6-F6
whether the complete record is representable. Gas and particle writes occur
only after every box passes validation and planning.

```text
GasData + T + strategy + dt
          |
validate closed scientific domain
          |
J = A*C or K*C^2 [events m^-3 s^-1]
          |
potential events = J * dt * volume * survival
          |
injection molecules/event * molar mass / N_A
          |
joint per-box/species gas-inventory limiter
          |
immutable source record + diagnostics
          |
E6-F5 discover slots -> E6-F6 exhaustion plan if needed
          |
all boxes feasible? no -> no writes | yes -> commit once
          |
particle activation + exactly matching gas depletion
```

For species `s`, `m_event,s = n_s*M_s/N_A`. Potential represented events are
`E_pot = J*dt*V*f_survival`. The shared admission factor is
`alpha=min(1,min_s(G_s/(E_pot*m_event,s)))` over participating species, and
`E_admit=alpha*E_pot`. Zero denominators are excluded. Particle represented
mass added for each species equals gas mass removed. One computational particle
may represent many events through its weight, but represented totals cannot
change.

## Scientific Contract

- Activation: `J=A*C`; kinetic: `J=K*C^2`. `C` is precursor number
  concentration explicitly converted from `kg/m^3` using molar mass and
  Avogadro's constant. Coefficient units are explicit and normalized to SI.
- Inputs are finite/nonnegative and configured validity intervals are closed.
  Out-of-domain calls raise rather than extrapolate. Zero time, coefficient,
  precursor, survival, or an unsatisfied configured saturation gate is a no-op.
- Injection composition is nonnegative with at least one positive molecule
  count. Formation diameter is metadata checked against the documented
  convention; no hidden growth occurs.
- The empirical forms follow Kulmala et al. (2006) and Seinfeld & Pandis (2016).
  Vehkamäki et al. (2002) is context, not an implemented parameterization. A
  supplied survival factor may represent Kerminen & Kulmala (2002); it is never
  inferred silently.

## Data / API / Workflow Changes

- **Data Model:** No required `ParticleData` or `GasData` schema change.
  Typed immutable CPU sidecars hold config, source records, and diagnostics:
  potential/admitted events, limiting species, gas removed, slot counts,
  exhaustion policy, and residual demand.
- **API Surface:** Add `NucleationStrategy`, activation/kinetic strategies,
  builders, `NucleationFactory`, source finalizer, and `Nucleation` runnable
  under `particula.dynamics.nucleation` and intended package exports.
- **Mutation Contract:** Success mutates participating gas concentrations and
  selected particle mass/concentration/charge; only configured E6-F6 scaling
  may change volume. Density, metadata, shapes, identities, requests, and
  unselected state remain unchanged.
- **Workflow Hooks:** E6-F5 supplies discovery/activation; E6-F6 supplies
  complete-demand exhaustion; E6-F8 ports the CPU contract; E6-F9 integrates it.
- **Failure Boundary:** Scientific, shape, capacity, policy, and conservation
  errors occur before any caller state is written. No partial box succeeds.

## Security & Compliance

There are no network, permission, or persistence changes. Scientific safety
requires citations, units, explicit validity domains, finite validation,
inventory limitation, deterministic packaging, conservation, and
failure-before-mutation tests. Documentation must not claim general atmospheric
predictiveness, unimplemented Vehkamäki/CNT physics, GPU parity, dynamic
capacity, hidden transport, or performance evidence.
