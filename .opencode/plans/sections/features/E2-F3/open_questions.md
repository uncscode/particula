# Open Questions

1. What is the final `E2-F2` CPU `EnvironmentData` module path and exact field
   list?
2. Are derived humidity/saturation fields always stored as arrays, or are any
   computed lazily from temperature, pressure, and gas state?
3. Should `to_warp_environment_data` default to `device="cuda"` for consistency
   with existing helpers, or should this new helper default to `"cpu"` during
   early migration? The recommended default is consistency with existing GPU
   helpers.
4. Should a schema field tuple be shared between CPU validation, conversion,
   and tests, or should conversion remain explicitly per-field for review
   clarity?
5. Do downstream kernel tracks require additional environment fields beyond
   those planned in `E2-F2`?

These questions should be resolved before or during `E2-F3-P1`.
