# P3 native builder validation

P2 prerequisite: `19ed66933`, PR #1626, validated before P3 implementation.
The public builder now assembles three native containers and delegates to
Aerosol admission. Repeated builds share the supplied containers. The old
builder and its historical tests remain concrete legacy-only. Exact consumers
and native successors are recorded in the legacy and capability ledgers.

Repository `.venv`, 2026-11-10 (America/Denver):

- Focused native/legacy construction, replacement and quick-start: 51 passed.
- Untargeted full runner, host GPU/cache access: 6,815 passed, 19 skipped,
  92% coverage; configured 80% gate passed.
- Source examples plus infrastructure (`pytest examples_tests scripts/tests
  -q -Werror --no-cov`), host access: 277 passed, 1 skipped.
- Default lint runner: Ruff and mypy passed.
- Strict MkDocs and validation-only strict documentation commands: passed.
- `scripts/build_particula_plugins.py --sync` and `--check`: passed.
- Three changed notebook code-cell pairs synchronized from Python with Jupytext;
  outputs preserved. Full tutorial execution remains M5-owned.

The helper retains released-0.2.x behavior only when the temporary legacy
module is absent; unrelated import errors propagate. The generated OpenCode
payload is synchronized from the canonical script.
Logs: `/tmp/e9-m2-p3-{full,focused,lint,examples-infrastructure,docs,docs-validation}.log`.
