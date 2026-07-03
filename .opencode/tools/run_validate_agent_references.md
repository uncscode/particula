# run_validate_agent_references

Validation-safe wrapper for repository agent-reference checks.

## Preferred wrapper

- Use `run_validate_agent_references` only in the two allowlisted validator agents: `docs-validator` and `adw-validate`.

## Compatibility status

- Preferred narrow wrapper for validator-oriented agent flows.
- CI and manual shell flows should continue using `scripts/validate_agent_references.sh`.

## Direct fields

- `cwd`
- `baselinePath`

Keep `cwd` explicit. It is the safety boundary for repository/worktree-root confinement.
Keep `baselinePath` explicit too; it is the only supported suppression input and
must point to git-tracked, committed, clean, reviewable JSON under
`.opencode/guides/`.

## Bounded `options` tokens

- None.

## Examples

```json
{ "cwd": "/path/to/worktree" }
{ "cwd": "/home/user/repo" }
{ "cwd": "/path/to/worktree", "baselinePath": ".opencode/guides/agent-reference-validation-baseline.json" }
```

## Notes

- `cwd` must resolve to the current repository/worktree root exactly.
- `baselinePath` must be repo-relative and resolve under `.opencode/guides/`.
- `baselinePath` must also be git-tracked and free of local modifications.
- The wrapper only runs `scripts/validate_agent_references.py` via `python3`.
- The wrapper fails closed when the validator script has local modifications.
- Baseline entries are auditable suppressions only: approved matches can remain
  visible in output, but new/unbaselined violations still fail the run.
