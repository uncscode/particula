# E2-F6 Dependencies

## Internal Dependencies

- **E2-F1:** Required. Provides schema foundation and terminology for the
  particle data model used as the study baseline.
- **E2 parent epic:** Supplies the broader data-model and numerical foundation
  scope for issue #1172.
- **E2-F2 through E2-F5:** Inform environment and condensation boundary choices,
  but this feature should avoid taking hard dependencies on unfinished schema
  changes except where documented.

## Code Dependencies

- `particula/particles/particle_data.py` for baseline mass storage and derived
  quantity APIs.
- `particula/particles/particle_data_builder.py` for baseline dtype coercion.
- `particula/gpu/warp_types.py` and `particula/gpu/conversion.py` for current
  GPU dtype and transfer behavior.
- `particula/dynamics/condensation/mass_transfer.py` for conservation-limited
  CPU reference calculations.
- `particula/gpu/kernels/condensation.py` for current GPU condensation behavior
  and fp64 kernel baseline.

## External Dependencies

- NumPy for CPU candidate projections and precision analysis.
- SciPy only if existing condensation helpers require it for reused cases.
- NVIDIA Warp and CUDA are optional for GPU throughput data; the study must be
  skip-safe when CUDA is unavailable.

## Ordering Constraints

No production schema or dtype change should depend on this feature until the
final report is complete and accepted.
