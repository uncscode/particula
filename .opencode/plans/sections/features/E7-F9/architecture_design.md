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

- **Data Model:** Add only bounded diagnostic descriptors/results if E7-F5 does
  not already supply them. Results use fixed `(n_boxes,)`, `(n_boxes, n_species)`,
  or explicitly documented scalar shapes and same-device float64 storage. Freeze
  checkpoint schema/version and evidence metadata; do not change core containers.
- **API Surface:** Expose user-relevant diagnostic requests/results through the
  deliberate `particula.execution` boundary. Keep Warp kernels, status arrays,
  scratch records, fixture builders, and closeout tooling private/test-only.
- **Diagnostics semantics:** Total species mass includes concentration-weighted
  particle inventory and gas amount using box volume. Number is particle number
  concentration. Energy follows E7-F2 latent-heat sign/unit conventions.
  Conservation residuals require an explicit baseline and source/sink ledger.
- **Workflow Hooks:** Diagnostics execute only at declared E7-F5 barriers after
  relevant updates. Normal-step reducers launch on-device and never call
  `.numpy()`, conversion helpers, checkpoint, or implicit synchronization.
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
