## Child Plans

### Feature Tracks

| ID | Feature Plan | Status | Notes |
|----|--------------|--------|-------|
| E2-F1 | Container schema and ownership conventions | Shipped | Defined authoritative ownership and shapes for particle, gas, and environment state. |
| E2-F2 | CPU `EnvironmentData` container | Shipped | Added per-box temperature, pressure, species-resolved `saturation_ratio`, validation, and docs. |
| E2-F3 | `WarpEnvironmentData` and transfer helpers | Shipped | Added Warp struct, conversion helpers, exports, and CPU/GPU round-trip tests. |
| E2-F4 | `GasData` / `WarpGasData` schema reconciliation | Shipped | Resolved names, partitioning, vapor pressure ownership, and round-trip semantics. |
| E2-F5 | Scalar-to-per-box environment migration | Shipped | Preserved scalar kernel compatibility while enabling per-box environment arrays. |
| E2-F6 | Mass representation and precision study | Shipped | Evaluated current fp64 mass representation and published the recommendation report. |
| E2-F7 | Condensation stiffness characterization | Shipped | Characterized timestep stiffness and recommended integration foundations. |
| E2-F8 | CPU dynamics container boundaries | Shipped | Clarified where data containers are supported and require explicit errors for unsupported multi-box CPU paths. |
| E2-F9 | Foundation docs and examples | Shipped | Published container, transfer helper, shape, limitation, and roadmap handoff docs. |

### Maintenance Tracks

Maintenance Tracks: none
