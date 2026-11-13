# Dependencies

## Upstream
- **E4-F1:** Required numeric thermodynamic configuration, validation pattern,
  and current-temperature `vapor_pressure[n_boxes, n_species]` refresh.
- **E1/E2/E3:** Existing GPU data, kernel, and execution foundations inherited
  through E4-F1.

## Downstream
- **E4-F4:** Converges E4-F2 physics with E4-F3's fixed four-substep execution.
- **E4-F5 through E4-F7:** Later energy, runnable, parity, and documentation
  tracks rely on the stable supported-mode contract.

## Phase Ordering

P1 and P2 establish independently tested formulas; both gate P3 integration.
P3 gates P4 coupled parity and documentation. E4-F2 may proceed in parallel
with sibling E4-F3 only after E4-F1 ships.

External dependencies remain existing NumPy, pytest, and optional Warp; no new
package is planned.
