# Dependencies

## Upstream

- **E4-F1:** On-device vapor-pressure refresh and numeric thermodynamic configuration; establishes species ordering and validation contract.
- **E4-F2:** Supported activity and effective surface-tension models plus explicit unsupported modes.
- **E4-F3:** Exactly four fixed substeps, stable-shape fp64 scratch, accumulated transfer, and validation-before-mutation behavior.
- **E4-F4:** Latent-heat temperature feedback, signed whole-call energy diagnostics, and bookkeeping criteria.
- **E4-F5:** Gas partitioning, deterministic inventory limiting, gas/particle coupling, and per-box/per-species conservation.
- **E4-F6:** Mandatory Warp CPU parity, optional CUDA evidence, graph-readiness boundaries, and final tolerances.
- **Issue 1272 guardrails:** Existing documentation assertions remain authoritative until all six implementation/evidence tracks satisfy their exit criteria.

## Downstream

- E4 epic completion and roadmap publication depend on E4-F7 accurately presenting the verified exit bar.
- Future high-level GPU `Aerosol`/`Runnable` integration may link to this low-level contract but is not enabled by it.
- User support and release notes depend on the canonical troubleshooting and reproduction commands.

## Phase Ordering

All E4-F1 through E4-F6 behavior and evidence was verified stable before P4
publication. P1 preceded P2 so the example targets the canonical contract, and
P2 preceded P3 command publication. Issue #1317 completed P4 only after
inspecting the E4-F1--E4-F6 `phase_details.md` records and P1--P3 entries above,
then passing `pytest particula/tests/condensation_latent_heat_docs_test.py -q
-Werror` (22 passed). Warp CPU remains the baseline and optional CUDA remains
additive evidence with clean skips; high-level integration remains downstream.
