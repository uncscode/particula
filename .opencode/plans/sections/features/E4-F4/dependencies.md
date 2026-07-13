# Dependencies

**Upstream:**
- **E4-F1:** numeric thermodynamic configuration and current per-substep
  saturation-pressure refresh; stale pressure is forbidden.
- **E4-F2:** activity/effective surface tension and the shared
  activity-adjusted Kelvin surface pressure.
- **E4-F3:** exactly four substeps, stable fp64 scratch, bounded applied
  transfer, and whole-call transfer accumulation.
- CPU condensation modules define equations, units, and issue #1272 signs.

**Downstream:**
- **E4-F5:** gas mutation and full particle-plus-gas conservation.
- **E4-F6:** complete Warp CPU/CUDA parity and exit-bar evidence.
- **E4-F7:** final documentation and release closure.

**Phase ordering:** P1 formulas/validation precede P2 integration; P2 precedes
P3 diagnostics; P4 closes integration and docs. E4-F1/F2/F3 must be available
before P2 finalizes. Existing NumPy, Warp, and pytest are sufficient.
