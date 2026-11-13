# Dependencies

## Upstream

- **E4-F1 (required):** Supplies fixed-shape numeric thermodynamic
  configuration and on-device vapor-pressure refresh. E4-F3 must invoke that
  refresh during every substep and preserve its validation-before-mutation
  contract.
- Existing issue #1272 candidate evidence and fixed fp64 GPU container contracts
  define the baseline behavior to promote.

## Parallel Sibling

- **E4-F2:** Activity and effective surface-tension physics may proceed in
  parallel. E4-F3 must provide a loop location where those calculations can be
  refreshed after convergence, but must not require E4-F2 to ship.

## Downstream

- **E4-F4:** Converges E4-F1, E4-F2, and E4-F3 with latent-heat correction and
  energy diagnostics.
- **E4-F5:** Adds gas-coupled updates and conservation after fixed-four behavior.
- **E4-F6:** Builds full device-aware parity, conservation, mutation, and
  graph/autodiff-readiness evidence.
- **E4-F7:** Publishes support contracts and examples after implementation gates.

## Phase Ordering

E4-F1 must ship before E4-F3 starts. Within E4-F3, P1 buffer contracts precede
P2 integration; P2 precedes P3 production validation; P4 documents the verified
behavior last. E4-F2 and E4-F3 converge only at E4-F4.
