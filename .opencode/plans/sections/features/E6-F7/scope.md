# Scope

Deliver a CPU-only, particle-resolved nucleation reference that computes a
bounded empirical event rate, converts admitted events into fixed-shape
multi-species source records, limits those records by available gas, and commits
gas depletion plus E6-F5/E6-F6 particle activation as one validated operation.

## In Scope

- A `NucleationStrategy` interface and activation-type `J=A[H2SO4]` and
  kinetic-type `J=K[H2SO4]^2` strategies.
- SI-normalized inputs and outputs: gas mass concentration in `kg/m^3`, molar
  mass in `kg/mol`, temperature in K, time in s, and `J` in `#/m^3/s`.
- Configured closed validity ranges for precursor concentration, temperature,
  and optional saturation gate; out-of-domain evaluation fails closed.
- Fixed injection composition as molecules per event (or exactly equivalent
  species mass), formation-size metadata, and optional explicit survival factor.
- Per-box/species inventory finalization: admitted events do not exceed rate
  demand, gas availability, or completely representable slot demand.
- CPU E6-F5 slot activation and E6-F6 exhaustion integration, including default
  resampling and optional representative-volume scaling semantics.
- Strategy builder/factory exports, `Nucleation` runnable with substeps,
  diagnostics, fast tests, citations, and documentation.

## Out of Scope

- Full Vehkamäki binary parameterization, classical-nucleation free-energy
  solver, ion-induced/heterogeneous nucleation, cluster dynamics, chemistry, or
  automatic coefficient calibration.
- Hidden survival/growth correction; Kerminen-Kulmala correction applies only
  through a caller-supplied documented factor.
- GPU kernels (E6-F8), integrated scheduling (E6-F9/Epic G), dynamic storage,
  hidden transfers, backend selection, graph capture, differentiability, or
  performance claims.
- Silent clipping of unsupported environmental inputs, partial multi-box
  commits, or silent loss of slot-exhausted source demand.
