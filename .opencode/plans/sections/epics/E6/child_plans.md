# Child Plans

### Feature Tracks

| ID | Feature Plan | Status | Notes |
|----|--------------|--------|-------|
| E6-F1 | CPU dilution strategy and runnable reference | Draft | Freeze particle/gas concentration semantics; add co-located tests and exports. |
| E6-F2 | Direct GPU dilution with CPU parity | Draft | Depends on E6-F1; fixed-shape scalar/per-box inputs with no hidden transfers. |
| E6-F3 | Neutral spherical/rectangular GPU wall loss | Draft | Port coefficient/removal physics and persistent-RNG behavior from CPU references. |
| E6-F4 | Charged GPU wall loss with neutral fallback | Draft | Depends on E6-F3; preserve image-charge, field, and zero-charge fallback semantics. |
| E6-F5 | CPU/GPU fixed-slot activation and diagnostics | Draft | Define active predicates, deterministic free-slot discovery, and caller-owned counts. |
| E6-F6 | Slot exhaustion, resampling, and volume scaling | Draft | Depends on E6-F5; resampling-first default and optional representative-volume scaling. |
| E6-F7 | CPU nucleation and particle-source process | Draft | Depends on E6-F5 and E6-F6; inventory-limited gas-to-particle source reference. |
| E6-F8 | Direct GPU nucleation process | Draft | Depends on E6-F5, E6-F6, and E6-F7; fixed-shape parity and conservation. |
| E6-F9 | Integrated validation, documentation, and closeout | Draft | Depends on E6-F1 through E6-F8; example, diagnostics, roadmap links, and exit bar. |

The table order is authoritative and preserves issue tracks T1 through T9.

### Maintenance Tracks

Maintenance Tracks: none
