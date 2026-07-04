# Open Questions

- Should `from_warp_gas_data()` require explicit `name` input, warn when using
  placeholders, or preserve the current placeholder behavior as a documented
  contract?
- Should non-binary `WarpGasData.partitioning` values raise an error before
  casting to CPU boolean values?
- Should missing `vapor_pressure` in `to_warp_gas_data()` remain a zero-filled
  default, become an explicit required argument for condensation paths, or emit
  a warning?
- Should GPU vapor pressure ever be returned to CPU as a sidecar, or should it
  remain intentionally discarded because CPU `GasData` does not own behavior?
- What final decisions from `E2-F2` and `E2-F3` constrain vapor-pressure
  ownership when temperature or environment state is involved?
