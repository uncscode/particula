# Open Questions

All E6-F1 planning questions were resolved on 2026-07-21 from the existing
dilution helpers, container semantics, and process conventions.

1. [x] **Finite-step rule:** use the exact first-order solution
   `c_new = c * exp(-alpha * dt)` for both particle and gas concentrations.
   This is nonnegative for all valid inputs and is E6-F2's CPU parity oracle.
2. [x] **Strategy construction:** the strategy accepts one precomputed
   coefficient. Callers derive `alpha=Q/V` with
   `get_volume_dilution_coefficient`; volume and flow are not duplicate mutable
   strategy state.
3. [x] **Rate return shape:** return a typed two-tuple in fixed
   `(particle_rate, gas_rate)` order. Existing dual-container process APIs use
   tuples, so a new result hierarchy is unnecessary.
4. [x] **Container mutation API:** update the underlying
   `ParticleData.concentration` storage in place with the representation-volume
   conversion, and use the supported gas concentration setter. Do not add a
   setter to the deprecated facade or call distribution-merging
   `add_concentration()`.
5. [x] **Builder/factory surface:** omit builders and factories. A single
   coefficient-configured strategy does not justify the multi-variant factory
   surface used by wall loss.
6. [x] **Free-function broadcasting compatibility:** preserve native NumPy
   broadcasting for scalar/scalar, scalar/array, equal-shape arrays, and other
   broadcast-compatible numeric arrays. Non-broadcastable shapes and invalid
   physical values raise; validation must not accidentally add list support.
