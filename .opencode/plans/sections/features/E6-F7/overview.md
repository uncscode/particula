# Overview

## Problem Statement

Particula lacked a validated CPU path from bounded nucleation potential rates
through gas admission into fixed particle slots without partial particle or gas
mutation. P1 establishes the scientific rate contract, P2 creates immutable
inventory-limited planning records, and P3 commits their represented demand
atomically after capacity policy resolution.

## Value Proposition

E6-F7 P1-P3 now provide CPU-only, concrete-module-only potential-rate,
inventory-admission, and source-commit boundaries. P3 consumes immutable P2
records, stages E6-F5 activation plus E6-F6 resampling/scaling on a private
`ParticleData` copy, scales pre-existing gas for representative-volume rows,
validates per-box/species scaled-domain conservation, then writes validated
particle and gas arrays atomically. The transaction remains unexported and
does not add a runnable, GPU parity, or a public API.

## User Stories

- As an aerosol modeler, I want cited scalar rate strategies with explicit
  units and closed bounds so that I do not accidentally extrapolate an
  empirical rate law.
- As a future process author, I want potential rates isolated from source and
  mutation behavior so that subsequent phases can add those contracts explicitly.
- As a process author, I want immutable gas-admitted demand committed through
  existing slot and exhaustion contracts so that capacity policy cannot cause
  partial writes or silent source truncation.

Parent epic: **E6**. Track: **T7**. P1 shipped for issue #1430, P2 for #1431,
and P3 for #1432. Builders/runnable work, direct GPU parity (E6-F8), and an
integrated consumer (E6-F9) remain deferred.
