# Child Plans

### Feature Tracks

| ID | Feature Plan | Status | Notes |
|----|--------------|--------|-------|
| E6-F1 | CPU dilution strategy and runnable reference | Shipped | Freeze particle/gas concentration semantics; add co-located tests and exports. |
| E6-F2 | Direct GPU dilution with CPU parity | Shipped | Depends on E6-F1; fixed-shape scalar/per-box inputs with no hidden transfers. |
| E6-F3 | Neutral spherical/rectangular GPU wall loss | Shipped | Port coefficient/removal physics and persistent-RNG behavior from CPU references. |
| E6-F4 | Charged GPU wall loss with neutral fallback | Shipped | Depends on E6-F3; preserve image-charge, field, and zero-charge fallback semantics. |
| E6-F5 | CPU/GPU fixed-slot activation and diagnostics | Shipped | Defines active predicates, deterministic free-slot discovery, and caller-owned counts. |
| E6-F6 | Slot exhaustion, resampling, and volume scaling | Shipped | Depends on E6-F5; resampling-first default and optional representative-volume scaling. |
| E6-F7 | CPU nucleation and particle-source process | Shipped | Depends on E6-F5 and E6-F6; inventory-limited gas-to-particle source reference. |
| E6-F8 | Direct GPU nucleation process | Shipped | Depends on E6-F5, E6-F6, and E6-F7; fixed-shape parity and conservation. |
| E6-F9 | Integrated validation, documentation, and closeout | Shipped | P1-P4 and the E6 closeout are complete. |

The table order is authoritative and preserves issue tracks T1 through T9.

### Maintenance Tracks

Maintenance Tracks: none
