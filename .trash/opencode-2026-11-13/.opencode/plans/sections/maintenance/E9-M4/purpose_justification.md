# Purpose and Justification

E9-M4 implements T4 of issue #1602 under parent E9: migrate CPU runnable
state access, composition, dilution, nucleation orchestration and existing CPU
adapters to the flat `Aerosol` holding `ParticleData`, `GasData` and
`EnvironmentData`. This is maintenance, not new physics or a new execution API.

M1 defines quantities and alignment, M2 supplies native construction, and M3
supplies native scientific strategies. Those foundations alone do not migrate
`particle_process.py`, legacy dilution gas groups, or CPU adapter fixtures.
Ignoring this layer would leave supported composed simulations dependent on
facades that M6 must delete, or silently change volume normalization, gas
selection, environmental authority and failure behavior.

Preserve original aggregate/container identities, single-box admission,
equal sequential substeps and current-gas coupling. Dilution must still include
nonpartitioning gas. Nucleation must remain atomic per attempted transaction,
not retrospectively roll back earlier successful substeps. Migrate enduring
behavioral tests alongside each implementation change, not during deletion.

Strict sequence: completed-and-validated E9-M1 → E9-M2 → E9-M3 → E9-M4 →
E9-M5 → E9-M6. M4 cannot begin implementation until the M3 gate is accepted;
M5 examples/docs cannot begin until M4 is completed and validated. This draft
does not claim any implementation, passing runtime checks or release readiness.
