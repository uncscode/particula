# Appendix

## Evidence and references

| Source | Drafting evidence / use |
|---|---|
| Issue #1602 via workflow issue state | Agreed T1 scope and replacement invariants |
| E9 `implementation_strategy`, `dependency_map`, `open_questions` sections | Strict serial gates and unresolved M1 decisions |
| `particula/particles/particle_data.py:58–80,166–232` | Distribution-dependent concentration documentation, existing derived properties and deep copy |
| `particula/particles/representation.py:566–592` | Getter divides raw concentration by volume before summation |
| `particula/gas/gas_data.py:54–80,82–147` | Ordered gas metadata, kg/m^3, constructor coercion and copy |
| `particula/gas/environment_data.py:22–43,45–83,94–129` | Environment authority and constructor copies; not a read-only validator |
| `particula/gpu/conversion.py` | Protected transfer boundary; no deletion authorization |
| `.opencode/guides/testing_guide.md`, `pyproject.toml` | Focused/full-suite split, markers, warning policy and wall-loss exclusion |
| `.opencode/plans/templates/maintenance/` | Canonical twelve-section format; embedded M23 examples used as formatting prior art |

Only E9 maintenance shells were found under the canonical maintenance section
directory during discovery; those are not completed implementation examples.

## Implementation ledgers to complete

- Contract row: field/helper, actual consumer, distribution convention, raw
  units, physical units, conversion equation, shape, order, mutation and test.
- Species row: gas index/name/flag, particle lane or explicit absence,
  environment lane, process-parameter index and mismatch rejection rule.
- Helper row: existing/reused/added, module/import, caller need, copy/view
  behavior, invalid-input behavior and adjacent test node.
- Replacement case: operation, held/candidate triple, admission result,
  identity guarantees and owning M2 acceptance test.
- Validation row: revision/date, command, exit status, counts/availability,
  report location and reviewer. All implementation outcomes currently pending.

## Drafting limitations

This run populates E9-M1 only. No production code or other plans are edited.
The available editor is `apply_patch`, so full-file replacements are used
instead of the unavailable `write` tool. Returned canonical relative paths
were checked for the E9-M1 prefix and traversal, and files are accessed under
the exact worktree. No independent lstat/symlink audit tool is available.
No schema command is used. Implementation tests, lint and docs execution are
future gates, not results of this planning-only run.
