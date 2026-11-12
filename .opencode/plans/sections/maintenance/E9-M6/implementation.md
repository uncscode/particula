# M6 implementation record

## Entry checkpoint — 2026-11-11

- Authorized objective: complete the approved fifteen-commit plan, clean/update
  documentation including full publication, and create one PR for #1607.
- Branch: `codex/e9-m6-legacy-removal`.
- Local base and freshly queried upstream main:
  `fb25078b6c0e8136ed06feeda59f0928b85c1b34` (M5 merge).
- Preserved the six existing planning-file modifications and reconciled the
  accepted plan: remove untagged execution/checkpoint admission; retain tagged
  v5 and explicit transfer functions; require both strict publication builds.
- Initial lexical inventory: `removal_inventory.json`, 95 matching files and
  740 test functions in matching Python files. Every disposition is pending;
  this is a search seed, not approval to delete tests or complete P1. Dynamic
  aliases, metadata conditionals and generated references still need expansion.
- Started the untargeted baseline with repository `.venv` on PATH:
  `python .opencode/tools/run_pytest.py --timeout 1200`.
  Output: `/tmp/m6-baseline-pytest.log`. Running session at handoff: 12235;
  verify that live handle before restarting. No passing result claimed yet.
- Planning diff whitespace check passed. No production changes or commits yet.

## Next work

Complete baseline collection and assertion-level inventory, starting with CPU
condensation native successors and surviving untagged CPU behavior. Map each
removed assertion before deleting its old fixture. Then execute serial commits
2–15 from `phase_details.md`. Preserve full required gates; do not shrink the
objective to facade imports alone. Readiness and PR creation remain pending.

## Condensation audit checkpoint

- Preserved 144 original strategy test nodes and 268 assertion/check expressions
  in `condensation_assertions.json`. This records the exact checks, not just
  function names; all successor/disposition reviews remain pending.
- Native baseline command (repository `.venv`):
  `pytest particula/dynamics/condensation/tests/native_step_test.py particula/dynamics/condensation/tests/native_step_admission_test.py particula/dynamics/condensation/tests/native_staggered_test.py particula/dynamics/condensation/tests/species_mapping_test.py -q --no-cov`
- Result: **331 passed in 0.65s**, exit 0. Log:
  `/tmp/m6-condensation-baseline.log`. Assertion evidence only, not coverage.
- Source inspection confirms native entry points already own mapped physics,
  staged inventory finalization and staggered private working state. Remaining
  legacy branches include unwrap/mixed-type guards, strategy fallback, raw-weight
  rate scaling, facade cache publication and separate old stepping paths.
- The untargeted baseline session 12235 remained active at the latest poll.

## Native execution guard preparation

Introduced shared `forbid_legacy_facades` test fixture. It inspects only
already-loaded facade classes to block cached constructors and blocks fresh
imports through `sys.modules`, without importing APIs slated for deletion.
Three native integration/adapter/full-loop suites now use that guard and retain
all original numerical assertions. Two guard regressions cover both already
loaded and absent legacy modules: **2 passed**. Scoped Ruff check/format passed.

Initial integration run: **59 passed, 4 skipped, 4 failed**. All failures were
Warp CPU cache writes to read-only `/home/kyle/.cache/warp`; rerun started with
`WARP_CACHE_PATH=/tmp/particula-m6-warp`, log
`/tmp/m6-native-guards-writable-cache.log`. No failed run is called passing.
The ongoing untargeted baseline lacks this cache override; inspect its outcome
and rerun with writable cache after it finishes. Test-guard files were edited
while that process was active, so do not label its outcome a pristine-base
revision result; final validation will run on a frozen executable revision.

## Native guard validation and publication preparation

- Writable-cache native integration/adapter/full-loop/guard run completed:
  **65 passed, 4 skipped in 44.24s**, exit 0. Log:
  `/tmp/m6-native-guards-writable-cache.log`.
- Repaired malformed Google section bullets and continuation indentation in
  106 production source files identified by M5's publication warnings. Each
  before/after AST was compared after stripping docstring nodes and was equal.
  No numerical code, signatures or runtime behavior changed in those files.
- Full package Ruff check passes and all 552 Python files pass format check.
- A full strict publication build is active (session 12505), output
  `/tmp/m6-publication-pass1.log`, site `/tmp/particula-m6-full-site`.
  Rendering still reports notebook Div warnings; publication is not yet clean.
- Documentation preparation was pulled forward while the original full-suite
  process runs. Keep it separate from API removal commits. Original baseline
  session 12235 remains live; do not restart until terminal state is observed.

## Publication parser audit

A direct Griffe traversal parsed **2710 production docstrings** without
rendering notebooks (test modules and aliases excluded). First repair audit:
147 warnings; second repair audit: **99 warnings**. Logs:
`/tmp/m6-griffe-audit.log`, `/tmp/m6-griffe-audit2.log`.
This is not equivalent to full publication success. Remaining issues include
stale parameters, missing types in unannotated APIs, and section indentation.

All **107** uncommitted package files were compared against HEAD after stripping
docstring AST nodes: executable ASTs are identical. Ruff check and format check
pass (552 files). Publication session 12505 and baseline session 12235 were
still live at last observation; preserve their handles. Current source changes
mean an in-flight publication run is diagnostic rather than final evidence.

Resume parser cleanup from `/tmp/m6-griffe-audit2.log`, retain scientific prose,
and then return to serial removal. Do not claim inventory P1 complete merely
because initial lexical and assertion snapshots exist. No facade production
API has been removed yet; this remains a required major outstanding deliverable.

## Parser cleanup completion (publication still pending)

Direct Griffe audit now parses **2710 docstrings with zero warnings**:
`/tmp/m6-griffe-audit4.log`. Fixed stale argument names/units against signatures,
missing documented types, continuation indentation and empty/incorrect sections.
This does not prove full publication rendering or visual correctness. Final AST
comparison and strict rendering remain required before documentation closeout.

Original untargeted run terminated exit 1: **7559 passed, 501 failed, 678 skipped**;
coverage gate passed. Cache configuration was defective as demonstrated by the
focused native-loop failures. Full rerun with `WARP_CACHE_PATH=/tmp/particula-m6-warp`
is active in session **67715**, log `/tmp/m6-baseline-writable-cache.log`.
Do not classify all original failures as cache-only without the corrected run.
Publication diagnostic session **12505** remains to be polled.

## First production consumer removal

- Committed docstring-only repair as `54a7d3257`: 113 files; final AST check
  against its parent confirmed executable AST identity. Direct Griffe parser
  was clean; this is not a final rendered-publication acceptance claim.
- First full publication diagnostic terminated exit 1 with 138 warnings. It
  began before later repairs; fresh build session **57066** uses repaired
  sources, log `/tmp/m6-publication-pass2.log` and site
  `/tmp/particula-m6-full-site-clean`.
- Removed Equilibria's legacy aggregate import/type union. Native result
  recognition remains; generic solver mappings/phase-result behavior unchanged.
- Strengthened adjacent return-identity test: separate real native input and
  output aggregates replace the old same-stub assertion, which could pass via
  the unknown-result fallback. No existing scientific assertion was deleted.
- Equilibria suites plus native CPU adapter suite: **69 passed in 0.10s**;
  log `/tmp/m6-equilibria-removal2.log`. Scoped mypy log:
  `/tmp/m6-equilibria-mypy.log`. Ruff and whitespace checks passed.
