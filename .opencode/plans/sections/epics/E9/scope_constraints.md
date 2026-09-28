# Scope and Constraints

## In scope

- Contracts/accessors for particle, gas and environment containers, including
  derived quantities, concentration/volume semantics and species ordering.
- Flat `Aerosol`, validation of individual and coordinated replacement,
  data-native construction and retained useful preset capabilities.
- Native CPU condensation, coagulation and neutral/charged wall-loss strategy
  paths; migration of associated scientific fixtures and assertions.
- Runnable composition, dilution of both gas categories, nucleation
  orchestration, CPU adapters and integrated workflows.
- All supported affected examples, paired notebooks, API references, advanced
  state-access examples and final migration/release documentation.
- Last-stage removal of obsolete facades, `Atmosphere`, obsolete builders,
  facade bridges, compatibility branches, exports, messages and facade-only
  tests, after recording replacement coverage for enduring behavior.
- Shared distribution_type metadata and consistent per-simulation-volume
  storage, including radius PDFs; CPU/direct/prepared/resident normalization
  corrections and transfer/checkpoint metadata preservation. See appendix D1–D2.

## Out of scope

New physics, general multi-box CPU execution, renaming containers, changing
precision/array layouts beyond the approved distribution metadata,
permanent compatibility facades, new GPU public
APIs, graph-capture redesign, hidden transfers/fallbacks, performance studies
and Epic I implementation. Do not remove unrelated warnings or container
transfer helpers merely because they mention conversion.

## Constraints

Target v0.3.0 with no fixed calendar deadline or new external dependencies.
All six maintenance tracks execute strictly in sequence with completed-and-
validated handoffs. Tests ship beside each owning implementation change,
not in a deferred testing track. Repository test/lint/documentation policies
apply; focused assertions precede the untargeted coverage runner and explicit
adjacent CPU wall-loss tests are required. No arbitrary coverage gates or
local `-Werror` flags are introduced. No additional compliance constraints
were supplied.

The 2026-09-27 maintainer review amends E9 and all six children together;
the appendix records the bounded changes to issue #1602's original constraints.
