# Overview

## Problem Statement

Epic C closes with CPU examples and GPU hardening work, but future Epic D GPU
latent-heat parity still needs a deterministic CPU integration-level reference
before any GPU parity claim can be evaluated. Existing latent-heat checks are
strong at the unit level, and `condensation_particle_resolved_test.py` already
proves the particle-resolved condensation integration style. P1 now fills the
remaining gap with a default integration baseline for the public CPU
latent-heat runnable path, while leaving full conservation and energy-closure
assertions for follow-up work.

## Value Proposition

E3-F7 now has shipped P1 and P2 coverage in
`particula/integration_tests/condensation_latent_heat_conservation_test.py`,
and shipped P3 documentation updates in
`docs/Features/Roadmap/data-oriented-gpu.md` and
`docs/Features/condensation_strategy_system.md`. The implemented slice still
uses only public `particula` APIs, a constant latent-heat strategy, and
`MassCondensation.execute()`, but now also proves the CPU reference fixture
conserves whole-run water inventory, that `last_latent_heat_energy` matches
the final-step transferred water mass times the explicit latent-heat constant,
and that the roadmap/feature docs point future Epic D work to this CPU-only
diagnostic baseline without implying shipped GPU parity.

This leaves the feature with a stronger executable CPU reference baseline for
future GPU parity work while preserving the constraint that no GPU latent-heat
production parity is claimed in this feature.

## User Stories

- As a GPU implementer, I want a CPU latent-heat integration baseline so that I
  can compare future GPU latent-heat behavior against a reviewed reference.
- As a maintainer, I want the baseline to run in the default integration suite
  so that public CPU latent-heat runnable regressions are caught before Epic D
  work begins.
- As a follow-up implementer, I want the reviewed fixture to support strict
  conservation and energy-equality checks without changing production code or
  widening scope beyond the CPU integration test.

## Parent Epic Context

- Parent epic: E3.
- Feature: E3-F7, dependent on E3-F6 runnable CPU latent-heat example.
- Sibling features provide RNG hardening, mixed-scale coagulation diagnostics,
  benchmark/usage boundaries, direct-kernel docs, Warp pytest policy, and the
  E3-F6 latent-heat example that this baseline can reference.
