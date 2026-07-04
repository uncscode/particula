## Child Plans

### Feature Tracks

| ID | Feature Plan | Status | Notes |
|----|--------------|--------|-------|
| E2-F1 | Container schema and ownership conventions | Draft | Define authoritative ownership and shapes for particle, gas, and environment state. |
| E2-F2 | CPU `EnvironmentData` container | Draft | Add per-box temperature, pressure, humidity/saturation state, validation, and docs. |
| E2-F3 | `WarpEnvironmentData` and transfer helpers | Draft | Add Warp struct, conversion helpers, exports, and CPU/GPU round-trip tests. |
| E2-F4 | `GasData` / `WarpGasData` schema reconciliation | Draft | Resolve names, partitioning, vapor pressure ownership, and round-trip semantics. |
| E2-F5 | Scalar-to-per-box environment migration | Draft | Preserve scalar kernel compatibility while enabling per-box environment arrays. |
| E2-F6 | Mass representation and precision study | Draft | Evaluate current fp64 mass representation and document recommendations. |
| E2-F7 | Condensation stiffness characterization | Draft | Characterize timestep stiffness and recommend integration foundations. |
| E2-F8 | CPU dynamics container boundaries | Draft | Clarify where data containers are supported and where CPU paths remain single-box or legacy. |
| E2-F9 | Foundation docs and examples | Draft | Publish container, transfer helper, shape, limitation, and roadmap handoff docs. |

### Maintenance Tracks

Maintenance Tracks: none
