# Architecture Design

## High-Level Design

Profiling is an evidence consumer around the final resident implementation, not
a scheduler mode. One validated upstream fixture is prepared once. Warmup and
capture setup occur outside measured regions. Host timers record enqueue/graph
launch separately from synchronized elapsed samples; vendor tools collect
kernel-level evidence in separate runs to avoid treating profiler perturbation
as benchmark timing. A deterministic analyzer joins both artifact families by
workload and machine IDs and emits bounded recommendations.

```text
E8-F5 fixture + E8-F6 matrix + qualified CUDA device
                         |
             validate budget and provenance
                         |
           setup / capture / warmup (untimed)
                         |
          +--------------+----------------+
          |                               |
 host timing run                    profiler run
 uncaptured dispatch                Nsight Systems timeline
 captured graph launch              Nsight Compute metrics
 synchronized elapsed               per-kernel duration/count
          |                               |
          +---------- raw artifacts ------+
                         |
       normalize schema; never invent missing metrics
                         |
      contribution ranking + evidence-linked decisions
                         |
        machine-bounded report and follow-up issues
```

## Data / API / Workflow Changes

- **Data model:** Add frozen, concrete test-support records for schema version,
  workload dimensions, replay mode, machine/software identity, timing method,
  raw samples, kernel metrics, unavailability, bottleneck ranking, and
  recommendation provenance. Durations use integer nanoseconds where possible;
  percentages are derived, never primary evidence.
- **Artifact model:** Keep normalized JSON under `.artifacts/benchmarks` and
  reference external profiler exports by safe relative filename plus checksum.
  Do not commit bulky binary reports or absolute local paths.
- **API surface:** No package or top-level export. Extend only opt-in pytest
  benchmark/profiling support and documented external commands. `--benchmark`
  remains the sole collection-affecting switch.
- **Workflow hooks:** Consume E8-F3 resource identities, E8-F4 captured replay,
  E8-F5 validated fixtures, and E8-F6 scaling/timing/memory artifacts. Feed
  bounded findings to Epic H documentation and explicitly created follow-ups.
- **Timing contract:** Host launch samples and synchronized elapsed samples use
  identical work and step counts, but remain separately labeled. Kernel metrics
  come from dedicated profiler runs and are not numerically mixed with
  unprofiled wall-clock samples.

## Security & Compliance

The workflow requires no network, credential, permission, or public API change.
Metadata is allow-listed and must omit environment secrets, usernames, absolute
paths, device pointers, array payloads, RNG words, and unbounded exception text.
Artifact paths reject traversal and symlink escape. External commands are
documented for a human operator and never shell-expand values from artifacts.
Missing CUDA, permissions, profiler binaries, counters, or compatible metrics
produces explicit unavailable evidence and no CPU fallback.
