# Architecture Design

## High-Level Design

This feature is a documentation and schema-decision foundation. It does not
introduce new runtime containers, but it defines the contract downstream runtime
containers must follow.

```text
Current code inventory
  -> ParticleData / WarpParticleData field table
  -> GasData / WarpGasData field table
  -> scalar environment API and Atmosphere notes
      -> authoritative ownership decision record
          -> shape convention table
          -> CPU/GPU round-trip semantics
          -> downstream handoff map for E2-F2..E2-F9
```

## Data / API / Workflow Changes

- **Data Model:** No production data model changes are required in this feature.
  The deliverable is an authoritative schema table with owner, shape, dtype,
  mutability, CPU/GPU representation, and round-trip behavior.
- **API Surface:** No public API additions are planned. Documentation may link
  existing public exports such as `particula.particles.ParticleData`,
  `particula.gas.GasData`, and GPU conversion helpers.
- **Workflow Hooks:** Downstream E2 feature plans should cite the decision record
  before adding environment containers or changing kernel input contracts.

## Ownership Model to Decide

- `ParticleData`: owns particle masses, concentration/count representation,
  charge, particle material density, and currently per-box volume.
- `GasData`: owns species metadata and gas concentration, including names,
  molar mass, and partitioning eligibility.
- `EnvironmentData` (future): should own per-box thermodynamic state such as
  temperature, pressure, and selected humidity/saturation fields.
- GPU mirrors: should represent device-compatible views of the same state, with
  explicit exceptions for strings and device-only caches.

## Security & Compliance

No new permissions or external services are introduced. The main compliance risk
is scientific correctness: avoid undocumented lossy transfers, shape squeezing,
or ambiguous ownership that could produce silent numerical errors.
