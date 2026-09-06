# Risk Register

| Risk | Likelihood | Impact | Mitigation | Owner |
|------|------------|--------|------------|-------|
| Timer or synchronization boundaries favor one launch mode | Medium | High | Share fixture/step code; warm both modes; record raw samples and explicit synchronization; review timing diagram | P2 implementer |
| Large matrix causes OOM or destabilizes developer machines | Medium | High | Checked preflight against configured/device budget; run box rows sequentially; emit unavailable rows before allocation | P3 implementer |
| Logical bytes are confused with allocator-reserved or peak bytes | Medium | High | Preserve provenance and separate columns/scenarios; publish unexplained deltas and allocator caveats | P4/P5 implementers |
| E8-F3 roles are double-counted by handwritten formulas | Medium | High | P4 consumes one caller-supplied logical-byte aggregate, retains zero-byte communication selection, and tests uniqueness/reconciliation | P4 delivered |
| Device-memory probe is unavailable or asynchronous | Medium | Medium | Record probe method and availability; synchronize at boundaries; never infer missing observed values | P5 implementer |
| Tape projection is read as measured Epic I evidence | Medium | High | Label every tape row `projected`, state formula/assumptions, and keep it out of observed totals | Documentation owner |
| Benchmark results are mistaken for universal guarantees | High | Medium | Record hardware/software/date/command and machine-bounded caveats; avoid CI speedup thresholds | Feature owner |
| Artifact contains environment data, pointers, or payload values | Low | High | Allowlist scalar metadata, sanitize paths/errors, and test forbidden-field absence | P1 implementer |
| CUDA/capture absence is masked by Warp CPU fallback | Low | High | Reuse canonical availability checks, assert exact CUDA device, and serialize explicit unavailable status | P2/P3 implementers |
| Artifact schema drifts before E8-F8 closeout | Medium | Medium | Version schema, canonicalize records, and add backward/required-field contract tests | Feature owner |
