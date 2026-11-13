# M34-P1 Findings Artifact (Baseline Audit)

Issue: #2777  
Phase: M34-P1 (audit-only baseline capture)  
Generated in worktree: `825fbf37`

## Decision Summary

- Accepted findings: 2
- Deferred findings: 1
- Scope: wrapper contract baseline evidence only (no production behavior fixes)

---

## M34-P1-F01

- **Decision:** accepted
- **Affected wrapper/tool:** `adw_spec_read` compatibility-split wrapper tests
- **Drift class:** input validation drift
- **Concrete file path(s):**
  - `.opencode/tools/adw_spec_read.ts` (adw_id validation/normalization entrypoint)
  - `.opencode/tools/__tests__/adw_spec_read.test.ts` (baseline repro assertion)
- **Repro input payload:** `{ command: "read", adw_id: "bad" }`
- **Observed behavior:** returns deterministic validation text for invalid `adw_id`.
- **Expected contract behavior (baseline repro target):** test currently (intentionally) expects a stricter phrase fragment (`"exactly 8-character lowercase hex"`) that does not match baseline contract text.
- **Wrapper-level behavior:** wrapper currently emits `ERROR: 'adw_id' must be an 8-character hex string (e.g., "abc12345").`
- **Runtime/CLI behavior boundary:** runtime `adw spec` is not invoked; failure is pre-spawn wrapper validation text.
- **Deterministic envelope/assertion fragment(s):** `ERROR:` and `8-character hex`.
- **Rationale:** high-confidence, deterministic, no environment dependence; useful baseline-fail guard for validation-message contract hardening.

## M34-P1-F02

- **Decision:** accepted
- **Affected wrapper/tool:** `platform_operations` compatibility delegation tests
- **Drift class:** diagnostic precedence/envelope drift
- **Concrete file path(s):**
  - `.opencode/tools/platform_operations.ts` (required arg validation path for `pr-comments`)
  - `.opencode/tools/__tests__/platform_operations_compat_comment_pr_review.test.ts` (baseline repro assertion)
- **Repro input payload:** `{ command: "pr-comments" }`
- **Observed behavior:** wrapper returns deterministic missing-argument envelope requiring `issue_number` before spawn.
- **Expected contract behavior (baseline repro target):** test currently (intentionally) expects a delegated failure envelope fragment (`"Failed to execute 'adw platform pr-comments'"`) that should not appear for pre-spawn validation.
- **Wrapper-level behavior:** wrapper validates command-required issue token and returns `ERROR: 'issue_number' is required for command 'pr-comments'`-style message.
- **Runtime/CLI behavior boundary:** runtime delegation path should not execute in this invalid-input case.
- **Deterministic envelope/assertion fragment(s):** `ERROR:` and `issue_number` required phrase.
- **Rationale:** high-confidence contract boundary check; locks precedence between wrapper validation and delegated execution errors.

## M34-P1-F03

- **Decision:** deferred
- **Affected wrapper/tool:** `adw_plans` patch/cwd guardrails (candidate)
- **Drift class:** argument normalization drift
- **Concrete file path(s):**
  - `.opencode/tools/adw_plans.ts`
- **Repro input payload:** candidate payloads around `patch` normalization and `cwd` path guards.
- **Observed behavior:** no reproducible deterministic drift confirmed within this slice without broadening scope.
- **Expected contract behavior:** keep deterministic `ERROR:` envelopes for invalid `cwd`/`patch` payload shapes.
- **Wrapper-level behavior:** guardrails exist but evidence for a currently broken baseline contract was not high-confidence in this slice.
- **Runtime/CLI behavior boundary:** unresolved; deferred until targeted fix phase or additional reproducer evidence.
- **Rationale:** avoid speculative failing tests; preserve bounded, reproducible findings only.

---

## Mapping Constraints Verification

- All accepted findings map to at least one concrete `.opencode/tools/` path.
- Drift classes used are from the allowed set:
  - `input validation drift`
  - `diagnostic precedence/envelope drift`
  - `argument normalization drift`
- Each accepted finding explicitly separates wrapper-level behavior from runtime/CLI behavior.

## Handoff Notes

- This artifact is baseline evidence for M34-P2/M34-P3 fixes.
- Intentional failing tests are limited to accepted findings F01/F02.
- No production wrapper logic changes are included in M34-P1.
