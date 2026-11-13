# Dependencies

## Required predecessors
- **E4-F3:** supplies exactly four fixed substeps, current-state refresh, stable
  fp64 scratch ownership, and accumulated whole-call transfer semantics.
- **E4-F4:** supplies thermal correction and latent-energy accounting that must
  consume the finalized conserved transfer.
- Transitively, E4-F1/F2 define refreshed vapor pressure, activity, Kelvin, and
  surface-tension inputs used by each coupled substep.

## Downstream
- **E4-F6** consumes the gas-coupled production path for cross-device parity and
  evidence; E4-F5 must complete first.
- **E4-F7** may close documentation/production claims only after downstream
  gates pass.

## Phase Ordering

P1 validates the partitioning gate before P2 can finalize inventory-limited
transfer. P2 gates P3, which applies that finalized transfer to gas and
particles across the four substeps. P4 verifies the production hook against
the conserved path, and P5 documents only the verified ownership and limits.
E4-F4's finalized transfer and energy semantics must remain available before
P3 begins; this prevents gas mutation from bypassing the upstream accounting
contract.

## External
No new runtime dependency. NumPy defines the CPU reference and Warp supplies
CPU/CUDA execution. CUDA remains optional and must skip cleanly when absent.
