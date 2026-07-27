# Overview

**Problem Statement:** Particula has backend-selected condensation and Brownian
coagulation plus a resident GPU session, but no production scheduler that orders
all supported processes and state refreshes. Ad hoc loops can consume stale
temperature-dependent vapor pressure or saturation state, move gas at the wrong
time, reset stochastic resources, or violate the session's no-transfer contract.

**Value Proposition:** E7-F5 turns the shipped adapters and direct-process
boundaries into one validated, deterministic timestep. It executes supported
condensation, coagulation, dilution, wall loss, and nucleation against resident
state, applies environment and gas updates in dependency order, exposes bounded
diagnostic hooks, and performs no implicit conversion, synchronization, or
fallback. This preserves issue #1451 Track T5 and unblocks E7-F7, E7-F8, and
E7-F9.

**User Stories:**

- As a simulation user, I want the same declared process set to run in a stable
  order every timestep so results do not depend on registration order.
- As a GPU user, I want environment, derived thermodynamic, particle, gas,
  sidecar, and RNG state to remain resident between checkpoints.
- As a maintainer, I want invalid graphs and stale-state hazards rejected before
  launch so process physics remains delegated to its owning implementation.

Parent epic: E7. Scope authority: issue #1451, Track T5.
