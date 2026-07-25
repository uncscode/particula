# Overview

## Problem Statement

Particula had no validated, concrete CPU boundary for evaluating bounded
nucleation potential-rate laws. The shipped P1 boundary establishes the
scientific rate contract before any future source, inventory, or slot work.

## Value Proposition

E6-F7 P1 now provides CPU-only, unexported activation and kinetic potential
rate strategies with immutable validated domains, composition, and formation
metadata. It establishes strict scalar validation, overflow-safe SI conversion,
and exact zero/saturation-gate ordering without claiming source construction,
conservation, or GPU parity.

## User Stories

- As an aerosol modeler, I want cited scalar rate strategies with explicit
  units and closed bounds so that I do not accidentally extrapolate an
  empirical rate law.
- As a future process author, I want potential rates isolated from source and
  mutation behavior so that subsequent phases can add those contracts explicitly.

Parent epic: **E6**. Track: **T7**. P1 shipped for issue #1430; E6-F5/E6-F6
integration, direct GPU parity (E6-F8), and an integrated consumer (E6-F9)
remain deferred.
