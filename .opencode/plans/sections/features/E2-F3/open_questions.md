# Open Questions

## Resolved Answers

1. Expected CPU path is `particula.gas.EnvironmentData`. The field list is
   `temperature`, `pressure`, and `saturation_ratio` unless E2-F2 records a
   direct implementation blocker.
2. `saturation_ratio` is stored as an explicit array, not computed lazily from
   temperature, pressure, and gas state in the first implementation.
3. `to_warp_environment_data` should default to `device="cuda"` for consistency
   with existing GPU transfer helpers.
4. Keep conversion explicitly per-field for review clarity. Introduce a shared
   schema field tuple only if E2-F2 already defines one and tests benefit from
   reusing it.
5. No additional environment fields are required for E2-F3. Downstream kernel
   tracks must justify any new fields in their own scope.
