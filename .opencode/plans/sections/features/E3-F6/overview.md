## Overview

### Problem Statement

`CondensationLatentHeat` now exposes a real per-step latent-heat energy
bookkeeping path, but users do not have a runnable documentation example that
shows the strategy inside the normal CPU condensation workflow. Existing docs
describe the API and vapor-property latent heat configuration, yet a reader can
miss the distinction between merely setting a latent heat property and running
`MassCondensation.execute()` to obtain `last_latent_heat_energy` from actual
mass transfer.

### Value Proposition

This feature adds a CPU-only example under `docs/Examples/Dynamics/Condensation/`
that builds a latent-heat strategy through documented public factories/builders,
advances an aerosol with `par.dynamics.MassCondensation`, and reports
per-step/cumulative latent heat energy. The example gives users a copyable
workflow for energy bookkeeping while avoiding unsupported GPU parity claims.

### User Stories

- As a particula user, I want a runnable latent-heat condensation example so
  that I can reproduce the full condensation workflow from the documentation.
- As a scientific developer, I want the example to show energy released from
  actual particle mass transfer so that I can validate diagnostics against my
  own simulations.
- As a maintainer, I want CPU-only guidance and validation commands so that the
  deferred Epic B/E1 documentation gap is closed without expanding GPU scope.

### Parent Epic Context

- Parent epic: `E3`
- Feature: `E3-F6 - Add a runnable CondensationLatentHeat documentation example`
- Sibling tracks already cover GPU coagulation/condensation infrastructure,
  direct-kernel docs, and Warp testing policy. This feature is intentionally a
  documentation/example feature and does not add GPU latent-heat parity.
