# Example Tasks

1. Replace `aerosol.atmosphere` reads in scientific runnable `rate`/`execute`
   with M2-held native state; verify environment forwarding and exact identity
   in the same PR. Demonstrate full-cycle sequence ordering with two substeps.
2. Replace dilution's two-facade gas list with all lanes of unified `GasData`.
   Test interleaved partitioning flags and volume 4 m^3 against an independent
   exponential oracle without touching per-particle masses.
3. Inject native dilution commit failure after particle concentration changed.
   Prove restored particle/gas state, unchanged held identities and original
   exception propagation, including the existing rollback-failure boundary.
4. Replace nucleation facade topology/cache synchronization with the approved
   native admission contract. With depleted precursor gas, prove later equal
   substeps use the new rate, and a failing second attempt preserves the first.
5. Migrate real CPU condensation/coagulation adapter fixtures to M2 construction;
   retain profile rejection and dispatch-once tests and assert no conversion,
   retry, fallback or prepared-GPU rebinding.
6. Close the assertion/compatibility ledgers and publish narrow developer
   contracts plus final integration evidence for M5. Do not preempt M5 tutorial
   migration or M6 deletion while closing M4.
