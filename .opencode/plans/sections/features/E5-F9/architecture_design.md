# Architecture Design

### High-Level Design

```text
E5-F1..F6 implementation contracts
          + E5-F7 release evidence
          + E5-F8 carry-forward artifacts
                    |
                    v
P1 support contract ---- P2 direct example
                    \    /
                     P3 artifact/ID matrix (E5, E5-F1..F9)
                              |
                      P4 release gate (fail closed)
                               |
              all pass? -- no --> remain in pre-closeout state
                          yes --> mark E5 shipped / Epic F active
```

Documentation describes the final low-level interface; it does not wrap or
change `coagulation_step_gpu`. The example owns CPU fixtures, performs explicit
conversion, lazily imports the direct step, initializes caller-owned collision
and RNG sidecars once, runs the supported configuration, synchronizes before
host observation, and restores a CPU checkpoint.

### Data / API / Workflow Changes

- **Data Model:** No container or plan schema change. Existing particle fields,
  collision buffers, and `rng_states` ownership remain authoritative.
- **API Surface:** No new library API. Publish the existing lazy import from
  `particula.gpu.kernels` and a standalone documentation example.
- **Workflow Hooks:** P4 read child/phase status and validation results as a
  fail-closed release predicate. Roadmap status edits are outputs of a passing
  gate, never inputs used to infer success.

### Security & Compliance

No new network, credential, or privilege surface is introduced. Documentation
tests must resolve repository-local links without executing arbitrary linked
content. The example must not hide device transfers, suppress validation
errors, silently fall back to CPU physics, or report a kernel run when Warp is
unavailable. Closeout must not falsify child completion or optional CUDA skips;
Warp CPU failures when Warp is installed were blockers.
