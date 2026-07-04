# Open Questions

## Resolved Answers

1. Expected CPU path is `particula.gas.EnvironmentData`. The field list is
    `temperature`, `pressure`, and `saturation_ratio` unless E2-F2 records a
    direct implementation blocker.
2. `saturation_ratio` is stored as an explicit array, not computed lazily from
    temperature, pressure, and gas state in the first implementation.
3. `WarpEnvironmentData` shipped as a thin schema struct only; helper defaults,
   copy semantics, and sync semantics remain deferred until conversion phases.
4. Field declaration stayed explicit in `particula/gpu/warp_types.py`, with
   direct tests for attribute presence and deterministic stored values.
5. No additional environment fields were required for `E2-F3-P1`. Downstream kernel
   tracks must justify any new fields in their own scope.
