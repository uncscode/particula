# Overview

## Problem Statement

Epic C closes with CPU examples and GPU hardening work, but future Epic D GPU
latent-heat parity needs a deterministic CPU integration-level reference before
any GPU parity claim can be evaluated. Existing latent-heat checks are strong at
the unit level, and `condensation_particle_resolved_test.py` already proves the
particle-resolved condensation integration style, but there is no default
integration test that combines particle/gas mass conservation with latent-heat
energy bookkeeping through the public CPU runnable path.

## Value Proposition

E3-F7 adds a CPU-only baseline that future GPU work can compare against without
changing production GPU behavior. The baseline should be fast enough for default
integration tests, deterministic, and explicit about two invariants:

- water inventory is conserved between particles and gas; and
- `CondensationLatentHeat.last_latent_heat_energy` matches the mass transfer
  times a constant latent heat strategy.

This gives Epic D a stable acceptance target while preserving the constraint
that no GPU latent-heat production parity is claimed in this feature.

## User Stories

- As a GPU implementer, I want a CPU latent-heat integration baseline so that I
  can compare future GPU latent-heat behavior against a reviewed reference.
- As a maintainer, I want the baseline to run in the default integration suite so
  that conservation regressions are caught before Epic D work begins.
- As a documentation reader, I want the roadmap to identify this as CPU-only so
  that I do not mistake it for completed GPU parity.

## Parent Epic Context

- Parent epic: E3.
- Feature: E3-F7, dependent on E3-F6 runnable CPU latent-heat example.
- Sibling features provide RNG hardening, mixed-scale coagulation diagnostics,
  benchmark/usage boundaries, direct-kernel docs, Warp pytest policy, and the
  E3-F6 latent-heat example that this baseline can reference.
