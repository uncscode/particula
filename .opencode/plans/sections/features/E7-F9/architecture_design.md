# Architecture Design

## High-Level Design

E7-F9 adds an evidence layer around the shipped E7 execution system. Diagnostic
reducers are typed scheduler hooks writing caller/session-owned same-device
buffers. Integration fixtures invoke only the public backend/session boundary
and compare checkpoint snapshots with independent CPU/NumPy oracles. A closeout
matrix maps each issue #1451 and roadmap exit criterion to a test, example, and
reproduction command.

```text
CPU reference inputs + BackendRequest + TimestepPlan
                         |
             E7-F1..F8 public execution contracts
                         |
                  setup/upload once
                         |
        +----------------+----------------+
        | repeated canonical resident steps|
        | communication -> environment     |
        | -> derived state -> processes    |
        | -> device diagnostic reductions  |
        +----------------+----------------+
                         |
             explicit checkpoint/finalize
             sync + restore + metadata/RNG
                         |
      independent oracle comparisons + evidence matrix
                         |
        example + support contract + Epic G closeout
```

## Data / API / Workflow Changes

- **Data Model:** P1 implements the closed, concrete-only six-operation
  diagnostics protocol in `particula.execution.diagnostics`: two preserved
  snapshots followed by total species mass, particle-number concentration,
  latent-heat energy, and conservation residual. Outputs use fixed `(B, S)` or
  `(B,)` same-device float64 storage; no core container or checkpoint schema
  changed.
- **API Surface:** Registrations, plans, reducer kernels, and validation remain
  concrete direct-import seams. P1 added no `particula.execution` or top-level
  exports, public result surface, or user-facing documentation.
- **Diagnostics semantics:** Total species mass is box volume times
  concentration-weighted particle mass plus gas concentration; number is the
  per-box particle concentration sum. Latent energy copies the supplied signed
  P2-finalized energy ledger. Residual is total mass minus baseline and source
  ledger plus sink ledger. `gpu_resources.py` now performs operation-specific
  same-device float64 schema, capacity, and alias preflight for registrations.
- **Workflow Hooks:** The concrete diagnostics executor is directly callable
  today after its exact plan preflight. References to E7-F5 barrier placement
  describe only future scheduler integration. Normal-step reducers launch
  on-device and never call `.numpy()`, conversion helpers, checkpoint, or
  implicit synchronization.
- **Private scheduler correction:** P3 routes the canonical `nucleation` node
  directly to `ResidentNucleationAdapter.execute(...)` and records it as an
  ordinary thermal completion. It does not route nucleation through the
  thermodynamic-consumer path, whose consumer set remains condensation and
  diagnostics. This fixes private dispatch only; the schedule and public API
  are unchanged.
- **Checkpoint:** Host-visible diagnostics are observed only at explicit
  checkpoint/finalization unless the caller requests a named diagnostic readback
  boundary. The versioned payload includes E7-F4 state and E7-F8 stream records.
- **Evidence:** Every matrix row records backend, fixture, tolerance, command,
  pass/skip rule, and artifact. Warp CPU is required when Warp is installed;
  CUDA is optional and a clean skip is not counted as failure.

## Security & Compliance

No network, credential, or regulated-data behavior is added. Validate dimensions,
dtype, device, aliases, finite values, schema versions, bounded payload sizes,
and source/sink ledger compatibility before mutation. Checkpoints contain data,
not executable objects or dynamic imports. Diagnostics and failures report
metadata/reason codes rather than dumping simulation arrays. No test or example
may imply silent fallback, hidden transfer, mandatory CUDA, or unsupported scope.
