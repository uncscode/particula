# Purpose and Justification

E9-M6 implements issue #1602 track T6, the LAST maintenance track of parent
E9. It removes obsolete APIs only after native contracts, construction,
scientific consumers, runnables and supported examples have migrated and passed
their owning gates. The classifier identifies maintenance-only work, with no
feature/research tracks or diagnostic expansion.

The required sequence is completed-and-validated
`E9-M1 -> E9-M2 -> E9-M3 -> E9-M4 -> E9-M5 -> E9-M6`.
Draft status, a merged PR or a pending validation run is not permission to
start the next track. M6 produces v0.3.0 readiness evidence, not a release tag
or authorization to start Epic I.

Keeping ParticleRepresentation, GasSpecies, Atmosphere and temporary bridges
after migration preserves competing state-access paths and misleading public
contracts. Removing them prematurely can silently change volume normalization,
species alignment, process configuration, ownership and failure behavior.
Deleting their scientific tests would hide precisely those failures.

This plan therefore requires a consumer/compatibility audit before deletion,
traceable preservation of enduring tests, positive native and negative removed
API checks, and final-revision scientific, GPU, example and documentation
evidence. CPU-Warp transfer helpers are supported boundaries, not legacy
bridges. Missing required evidence blocks readiness; this draft reports no
implementation or execution success.
