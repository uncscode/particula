# P2 replacement validation

P1 prerequisite: `d5ec8ffb2`, PR #1625; P1's full gate passed before P2 began.
P2 adds all three setters and required-all-three `replace_data`, sharing one
private preflight/publication method with construction. No sequential setter
publication, normalization, GPU rebinding, or extra state authority is added.

`aerosol_replacement_test.py` covers six individual identity cases, three
individual box-layout rejections followed by coordinated success, eight
read-only failure cases with shared arrays, three same-object revalidations,
and coordinated species-layout replacement retaining old external bindings.

Repository `.venv` results, 2026-11-10 (America/Denver):

- Focused constructor/replacement/structural tests: 83 passed.
- Untargeted `.opencode/tools/run_pytest.py` with host device/cache access:
  6,803 passed, 19 skipped, 92% coverage; configured 80% gate passed.
- `.opencode/tools/run_linters.py`: Ruff and mypy passed.

Property setters were moved adjacent to getters to satisfy mypy during the
full run; this changes declaration order only. The final focused tests and
lint/type checks passed after that edit. Final M2 validation will rerun the
complete suite at the committed implementation revision.
Local logs: `/tmp/e9-m2-p2-{full,focused,lint}.log`.
P2 is implemented and validated; merge/release and M3 admission are not claimed.
