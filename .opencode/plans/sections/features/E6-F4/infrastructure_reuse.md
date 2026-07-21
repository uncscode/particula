# Infrastructure Reuse

- **E6-F3 neutral foundation:** Extend the planned configuration, preflight,
  coefficient buffers, active-slot predicate, fixed-slot clearing, and RNG
  lifecycle in `particula/gpu/kernels/wall_loss.py`; do not fork a second
  removal entry point. See E6-F3 architecture lines 5-64 and phase details.
- **CPU charged oracle:** Preserve `_electrostatic_factor`,
  `_resolve_electric_field`, `_drift_term`, `_combine_coefficients`, and
  `compute_coefficient_from_arrays` in
  `particula/dynamics/wall_loss/wall_loss_strategies.py:638-933`.
- **Neutral CPU oracle:** Reuse the spherical/rectangular coefficients selected
  by `_neutral_coefficient` at
  `particula/dynamics/wall_loss/wall_loss_strategies.py:596-636`; E6-F4 must
  compose charged terms around E6-F3's corresponding device result.
- **Charge physics:** Port the CPU call to `get_coulomb_enhancement_ratio`, its
  diagonal selection, absolute value, `[-50, 50]` clipping, exponentiation, and
  per-zero-charge factor of one from
  `particula/dynamics/wall_loss/wall_loss_strategies.py:658-671`.
- **Field physics:** Match scalar/vector magnitude, potential divided by the
  geometry scale, mobility based on `ELEMENTARY_CHARGE_VALUE`, signed drift,
  minimum radius/scale guards, and final nonnegative finite clipping at
  `particula/dynamics/wall_loss/wall_loss_strategies.py:673-752`.
- **CPU regression fixtures:** Reuse semantic cases in
  `particula/dynamics/wall_loss/tests/wall_loss_strategies_test.py:424-645`,
  especially neutral fallback, zero-potential image charge, sign-sensitive
  field drift, potential contribution, rectangular vectors, and survival bounds.
- **Warp data contract:** Read caller-owned `WarpParticleData.charge` from
  `particula/gpu/warp_types.py`; retain its shape, dtype, device, and identity
  for survivors, and clear it only when E6-F3 removes that fixed slot.
- **Environment and transfer patterns:** Follow `particula/gpu/conversion.py`
  and existing direct condensation/coagulation kernels for explicit transfers,
  scalar/per-box normalization, same-device validation, and failure-before-
  allocation ordering.
- **RNG and deactivation prior art:** Reuse persistent per-box `wp.uint32`
  sidecar and slot-clearing conventions from
  `particula/gpu/kernels/coagulation.py`; never store RNG in a container.
- **Backend matrix:** Use `particula/gpu/tests/cuda_availability.py` so Warp CPU
  is required evidence and CUDA remains optional.
