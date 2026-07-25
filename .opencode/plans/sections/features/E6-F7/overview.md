# Overview

## Problem Statement

Particula lacked validated CPU boundaries for both bounded nucleation
potential-rate laws and the gas-inventory admission of their resulting source
demand. P1 establishes the scientific rate contract; P2 turns its
survival-adjusted rates into immutable, inventory-limited planning records
before future slot or commit work.

## Value Proposition

E6-F7 P1/P2 now provide CPU-only, unexported potential-rate strategies and a
pure particle-source finalizer. P2 derives one shared admitted event count per
box from participating gas inventories, records provisional per-species mass
demand and deterministic limiting diagnostics, makes all record payloads
defensively owned and read-only, and applies a bounded vectorized rounding
correction. It does not mutate gas or caller inputs, activate slots, plan
exhaustion, commit mass, export APIs, or claim conservation or GPU parity.

## User Stories

- As an aerosol modeler, I want cited scalar rate strategies with explicit
  units and closed bounds so that I do not accidentally extrapolate an
  empirical rate law.
- As a future process author, I want potential rates isolated from source and
  mutation behavior so that subsequent phases can add those contracts explicitly.
- As a future source-transaction author, I want gas-admitted demand and its
  limiting diagnostics isolated from state writes so that slot/exhaustion
  planning and atomic commits can be added explicitly.

Parent epic: **E6**. Track: **T7**. P1 shipped for issue #1430 and P2 shipped
for issue #1431; E6-F5/E6-F6 integration, direct GPU parity (E6-F8), and an
integrated consumer (E6-F9) remain deferred.
