# Open Questions

- [ ] Confirm ideal activity support is limited to molar fraction for E4-F2;
  mass- and volume-fraction ideal strategies would remain CPU-only.
- [ ] Select the composition-dependent surface contract: global volume-weighted
  tension or phase-aware volume weighting with fixed-shape `phase_index`.
- [ ] Define numeric mode values and whether omitted E4-F2 configuration maps
  explicitly to legacy unit-activity/static-surface behavior.
- [ ] Record scientific `rtol` and `atol` per formula and coupled fixture before
  implementation acceptance; invariants remain tighter than parity tolerances.
- [ ] Decide whether scratch composition values are recomputed per launch or
  supplied through caller-owned reusable buffers for E4-F3 repeated substeps.

No question may be resolved by silently choosing stale/zero E4-F1 pressure,
adding strings/strategy objects to Warp data, or introducing host recomputation.
