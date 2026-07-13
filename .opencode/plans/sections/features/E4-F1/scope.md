# Scope

E4-F1 supplies the thermodynamic model contract, constant/Buck Warp formulas,
and the refresh hook used by GPU condensation. It establishes the primitive
that E4-F3 can invoke before each of its future four substeps.

## In Scope

- Numeric, fixed-shape, species-indexed model modes and parameters.
- Validation of mode, parameter bounds, dtype, shape, species count/order, and
  Warp device before launch or output mutation.
- Constant and Buck vapor-pressure implementations using `wp.float64`.
- On-device output with shape `(n_boxes, n_species)`.
- Refresh from normalized current temperature immediately before the existing
  condensation mass-transfer launch.
- Repeated-call, direct-temperature, and `WarpEnvironmentData` parity tests.
- Early failure when required thermodynamic configuration is absent or invalid,
  subject to the explicit compatibility decision tracked in open questions.

## Out of Scope

- Activity and surface-tension physics (E4-F2).
- Four-substep production orchestration and scratch management (E4-F3).
- Latent heat (E4-F4), gas coupling/conservation (E4-F5), broad readiness
  evidence (E4-F6), and final user examples/support matrix (E4-F7).
- Porting CPU vapor-pressure strategies other than constant and Buck.
- Moving vapor pressure into CPU `GasData` or storing Python strategy objects,
  strings, or species names in Warp data.
