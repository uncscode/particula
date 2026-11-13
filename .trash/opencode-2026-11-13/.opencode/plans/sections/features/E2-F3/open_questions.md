# Open Questions

## Resolved Answers

1. Expected CPU path is `particula.gas.EnvironmentData`. The field list is
    `temperature`, `pressure`, and `saturation_ratio` unless E2-F2 records a
    direct implementation blocker.
2. `saturation_ratio` is stored as an explicit array, not computed lazily from
    temperature, pressure, and gas state in the first implementation.
3. `WarpEnvironmentData` first shipped as a thin schema struct, `#1193`
   established `to_warp_environment_data(data, device="cuda", copy=True)`, and
   `#1194` completed the reverse boundary with
   `from_warp_environment_data(gpu_data, sync=True)`. The documented
   `sync=False` path requires manual `wp.synchronize()` before `.numpy()` use.
4. Field declaration stayed explicit in `particula/gpu/warp_types.py`, with
   direct tests for attribute presence and deterministic stored values.
5. No additional environment fields were required for `E2-F3-P1`, `E2-F3-P2`,
   or `E2-F3-P3`. Downstream kernel tracks must justify any new fields in their
   own scope.
6. Public helper exports now ship from `particula.gpu`; `#1195` also closed the
   remaining helper-surface follow-up by adding device-aware parity coverage
   and docs that make the explicit transfer boundary, shape rules, and
   round-trip example unambiguous. Only higher-level runtime integration
   remains future work outside this feature.
