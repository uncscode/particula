# Overview

## Problem Statement

Particula has theory and custom examples for nucleation but no supported CPU
process that converts a cited nucleation rate into fixed-shape particle source
records while depleting participating gas. Computing number production
independently from gas depletion can create mass, drive gas negative, or
silently discard demand when no particle slot is available.

## Value Proposition

E6-F7 provides the bounded NumPy reference for Epic E6: strategy-based
activation and kinetic sulfuric-acid rate laws, explicit validity domains and
injection composition, inventory-limited source finalization, and a
transactional CPU process built on E6-F5 slot activation and E6-F6 exhaustion.
It gives E6-F8 a deterministic scientific and conservation oracle without
promising general nucleation chemistry.

## User Stories

- As an aerosol modeler, I want a cited strategy with explicit units and bounds
  so that I do not accidentally extrapolate an empirical rate law.
- As a process author, I want gas depletion and particle activation committed
  together so that every box and species conserves represented mass.
- As a GPU developer, I want a deterministic CPU source-record oracle so that
  E6-F8 can validate direct Warp behavior independently.

Parent epic: **E6**. Track: **T7**. Required predecessors: **E6-F5** and
**E6-F6**. Direct downstream parity: **E6-F8**; integrated consumer: **E6-F9**.
