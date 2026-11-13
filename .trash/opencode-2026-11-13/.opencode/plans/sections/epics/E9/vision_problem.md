# Vision and Problem

E9 plans the v0.3.0 data-native migration requested in issue #1602. This is
planning, not a declaration that implementation or release validation is done.

## Current problems

1. `ParticleData`, `GasData`, and `EnvironmentData` coexist with legacy state
   facades, while `Aerosol` still groups an `Atmosphere` and a
   `ParticleRepresentation` (`particula/aerosol.py:37–49`).
2. Facades supply scientific configuration and normalization, not merely names.
   For example, `representation.py:566–581` divides concentration by volume.
   Mechanical field substitution risks incorrect units and scientific results.
3. Builders, CPU processes, adapters, examples and regression fixtures retain
   facade dependencies. Deleting those first would break supported workflows
   and can erase enduring behavioral coverage.
4. Species order, whole-container replacement, and state ownership need one
   explicit contract rather than multiple implicit compatibility paths.

## Vision

A flat `Aerosol` directly holds `ParticleData`, unified partitioning and
nonpartitioning `GasData`, and `EnvironmentData`. Users retain runnable `|`
composition and useful initialization/preset capabilities. Advanced users can
inspect held containers by identity and perform validated individual or
coordinated replacement without duplicate authoritative storage. Scientific
strategies belong to processes, not a new state facade.

## Why now

Issue #1602 identifies the refreshed v0.2.13 checkout as the baseline and
v0.3.0 as the intentional breaking boundary. Six strictly serial maintenance
tracks migrate consumers, tests and examples before deletion. Epic I follows
this migration; it is neither included here nor an inbound dependency.
