# Overview

## Problem Statement

Particula exposes `get_volume_dilution_coefficient()` and
`get_dilution_rate()` as CPU free functions, but it has no process-level CPU
reference that applies dilution consistently to an `Aerosol`. Consequently,
the downstream direct GPU dilution feature, E6-F2, has no stable strategy,
runnable, validation, or mutation contract against which to prove parity.

## Value Proposition

E6-F1 defines the authoritative CPU behavior for chamber dilution: particle
number concentration and gas mass concentration decrease according to the same
validated coefficient while particle mass, charge, density, distribution,
representation volume, gas metadata, temperature, and pressure remain
unchanged. A composable `Dilution` runnable gives users a normal Particula
process API and supplies E6-F2 with a deterministic NumPy oracle.

## User Stories

- As a simulation user, I want to compose dilution with existing runnables so
  that a chamber timestep does not require manual concentration bookkeeping.
- As a physics developer, I want explicit units and no-op behavior so that CPU
  dilution results are reproducible and physically interpretable.
- As a GPU developer, I want a tested CPU reference so that E6-F2 can establish
  scalar and per-box parity without inventing different semantics.

**Parent epic:** [E6](../../epics/E6/vision_problem.md) — GPU Process
Completeness. This feature is issue track T1 and is the required predecessor of
E6-F2; E6-F9 consumes it during integrated closeout.
