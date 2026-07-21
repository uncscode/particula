# Overview

- **Problem Statement:** E6-F7 establishes an inventory-limited CPU nucleation
  reference, but device-resident simulations still lack a direct Warp operation
  that turns the same bounded rates into fixed-shape particle sources. Moving
  state to the host would violate ownership, transfer, and composition goals.
- **Value Proposition:** A low-level GPU step can finalize gas-feasible demand,
  consume E6-F5 slot discovery and E6-F6 exhaustion policy, and commit matching
  gas depletion and particle mass without resizing arrays, hidden transfers, or
  CPU fallback. Independent parity and conservation evidence makes that path
  scientifically auditable.
- **User Stories:**
  - As a GPU simulation author, I want nucleation to operate on caller-owned
    Warp state so repeated process sequences require no intermediate host copy.
  - As a scientific user, I want the direct step to match the E6-F7 CPU oracle
    and conserve every box/species inventory so GPU results remain trustworthy.
  - As a library maintainer, I want fixed-shape sidecars and fail-before-write
    validation so invalid calls cannot partially mutate simulation state.
