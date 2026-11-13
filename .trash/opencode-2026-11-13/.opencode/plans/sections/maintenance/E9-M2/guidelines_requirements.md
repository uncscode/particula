# Guidelines and Requirements

1. Retain exactly three authoritative containers in native Aerosol; whole reads
   and accepted replacements preserve supplied identity.
2. Validate structure then particle representation before any reference swap.
   Same-object assignments revalidate. Rejected old/candidate arrays, metadata
   and references remain unchanged; do not chain public setters.
3. Keep full ordered gas names/masks and gas/environment widths. Particle
   species may differ. Do not infer chemical identity from equal widths or
   claim detection of unnamed permutations; process mappings stay downstream.
4. Initializers allocate owned arrays and normalize only explicit input units.
   PMF/PDF remain densities at V=1; resolved density input converts to counts
   once. Preserve sampled unit weights, defaults and the existing RNG path.
5. Reject lossy density/charge reductions. Preserve scalar/singleton and
   redundant forms, explicit empty slots and supported fractional weights.
6. Existing raw data-builder ownership is unchanged. Direct array edits remain
   possible and require revalidation. No aggregate access caches stale values.
7. Keep all temporary legacy consumers inventoried with migration and M6
   removal owners. Preserve old execution until its owning track migrates it.
8. No automatic GPU/resident rebind, hidden transfer/fallback, tagged GPU/PDF
   support or new multi-box CPU capability is established by construction.

Use co-located behavioral tests, independent mass-scale tolerances, full-package
coverage with unchanged thresholds, normal lint/type gates and strict docs.
Each of P1–P6 is a separate serial PR with tests in the phase that changes code.
Full tutorial workload migration is M5-owned, not a prerequisite falsely claimed
from bounded construction checks.
