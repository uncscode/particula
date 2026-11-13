# M35-P1 Findings Artifact (Tooling Reliability Audit)

Issue: #2846  
Phase: M35-P1 (analysis-only accepted-finding inventory)  
Generated in worktree: `416287b9`

## Decision Summary

- Accepted findings: 2
- Deferred findings: 1
- Scope: reviewable audit artifact only (no production behavior changes)
- Persisted workflow messages: none present in this ADW state; evidence is limited to plan docs, wrapper code, and existing wrapper tests.

---

## M35-P1-F01

- **Decision:** accepted
- **Affected wrapper/tool:** `adw_spec` split/compat `adw_id` preflight parity
- **Drift class:** input validation drift
- **Source link/signal:** issue #2843 audit scope; `.opencode/tools/adw_spec.ts:239-250`; `.opencode/tools/adw_spec_read.ts:32-48`; `.opencode/tools/adw_spec_messages.ts:30-47`; `.opencode/tools/__tests__/adw_spec.test.ts:25-35`; `.opencode/tools/__tests__/adw_spec_read.test.ts:26-38`; `.opencode/tools/__tests__/adw_spec_write.test.ts:25-35`
- **Concrete file path(s):**
  - `.opencode/tools/adw_spec.ts`
  - `.opencode/tools/adw_spec_read.ts`
  - `.opencode/tools/adw_spec_write.ts`
  - `.opencode/tools/adw_spec_messages.ts`
- **Expected contract:** whitespace-only `adw_id` should fail deterministically before spawn with the same split/compat classification and message family.
- **Likely implementation seam:** shared `adw_id` validation/normalization logic or coordinated wrapper-local preflight updates across the split and compatibility surfaces.
- **Parity scope:** split+compat
- **Owner phase:** `M35-P2`
- **Serialization note:** no shared helper is currently imported across these wrappers, but the same contract is duplicated across four files, so the follow-on fix should land as one parity slice.
- **Rationale:** current evidence shows the compatibility wrapper treats whitespace-only `adw_id` as invalid hex while split wrappers classify the same input as missing/blank required input. That is a concrete pre-spawn validation-order mismatch, not a runtime CLI difference.

## M35-P1-F02

- **Decision:** accepted
- **Affected wrapper/tool:** `adw_plans` compatibility vs split diagnostic redaction behavior
- **Drift class:** diagnostic precedence/envelope drift
- **Source link/signal:** `.opencode/tools/adw_plans.ts:121-139,738-775`; `.opencode/tools/adw_plans_read.ts:77-95`; `.opencode/tools/adw_plans_mutate.ts:88-106`; `.opencode/tools/__tests__/adw_plans_compat_required_args.test.ts:55-90`; `.opencode/tools/__tests__/adw_plans_read_required_args.test.ts:54-83`; `.opencode/tools/__tests__/adw_plans_mutate_required_args.test.ts:68-96`
- **Concrete file path(s):**
  - `.opencode/tools/adw_plans.ts`
  - `.opencode/tools/adw_plans_read.ts`
  - `.opencode/tools/adw_plans_mutate.ts`
  - `.opencode/tools/adw_plans_contract_shared.ts`
- **Expected contract:** command-failure diagnostics should preserve stderr→stdout precedence while redacting absolute filesystem paths consistently across compatibility and split surfaces.
- **Likely implementation seam:** extract or align the `sanitizeOutput`/diagnostic helper path used by `adw_plans` wrappers so command-failure envelopes share the same redaction behavior.
- **Parity scope:** split+compat
- **Owner phase:** `M35-P3`
- **Serialization note:** if the fix is implemented by introducing a shared diagnostic helper for the plans family, serialize that helper touch instead of parallel edits across compat and split wrappers.
- **Rationale:** the compatibility wrapper strips control characters only, while the split wrappers also redact path-like substrings to `<path>`. Existing tests already lock preflight diagnostics and stderr-first precedence, so the remaining high-confidence gap is family-level command-failure redaction parity.

## M35-P1-F03

- **Decision:** deferred
- **Affected wrapper/tool:** `adw_issues_batch_*` split wrappers parity coverage
- **Drift class:** coverage gap / argument normalization risk
- **Source link/signal:** `.opencode/tools/adw_issues_spec.ts:326-517`; `.opencode/tools/adw_issues_batch_init.ts`; `.opencode/tools/adw_issues_batch_read.ts`; `.opencode/tools/adw_issues_batch_write.ts`; `.opencode/tools/adw_issues_batch_log.ts`; `.opencode/tools/adw_issues_batch_summary.ts`; no matches for `.opencode/tools/__tests__/adw_issues_batch*.test.ts`
- **Concrete file path(s):**
  - `.opencode/tools/adw_issues_spec.ts`
  - `.opencode/tools/adw_issues_batch_init.ts`
  - `.opencode/tools/adw_issues_batch_read.ts`
  - `.opencode/tools/adw_issues_batch_write.ts`
  - `.opencode/tools/adw_issues_batch_log.ts`
  - `.opencode/tools/adw_issues_batch_summary.ts`
- **Expected contract:** split wrappers should preserve compatibility-wrapper command validation, optional-input omission, and deterministic failure-envelope behavior.
- **Likely implementation seam:** direct Bun coverage for the split wrappers first; only then decide whether any shared helper extraction or contract fixes are needed.
- **Parity scope:** split+compat
- **Deferred follow-up:** add targeted split-wrapper Bun suites in a later regression slice after a concrete reproducer or parity assertion target is selected.
- **Defer reason:** this is evidence of missing direct split-wrapper coverage, not evidence of a demonstrated runtime regression. The absence of `adw_issues_batch*.test.ts` files should be treated as coverage evidence only.

---

## Mapping Constraints Verification

- All accepted findings include stable IDs, source references, affected paths, expected contracts, likely seams, parity scope, and explicit routing.
- Accepted findings route to `M35-P2` or `M35-P3`; shared-helper serialization is called out only where the seam would overlap.
- Deferred work is separated from accepted work and includes an explicit defer reason.
- Compatibility-surface findings identify the matching split-wrapper relationship.
- Issue-batch coverage notes are labeled as coverage evidence, not as proven behavioral regressions.

## Acceptance Criteria Coverage Check

- Reviewable artifact under existing repository review path: **covered** by this file.
- Stable finding IDs, source references, affected paths, expected contracts, seams, parity scope, and accept/defer decisions: **covered** in each finding block.
- Accepted findings routed to `M35-P2`, `M35-P3`, or shared-helper serialization: **covered** in accepted finding metadata.
- Deferred findings clearly separated with brief reasons: **covered** in `M35-P1-F03`.
- Analysis-only unless a helper is required: **covered**; no helper code or user-facing docs were added in this slice.
- Issue-batch test absence labeled as coverage evidence only: **covered** in `M35-P1-F03`.

## Handoff Notes

- Canonical artifact path: `.opencode/tools/__tests__/fixtures/feedback_findings/M35-P1-findings.md`
- Accepted findings are intentionally bounded to one validation-parity slice (`M35-P2`) and one diagnostics-parity slice (`M35-P3`).
- No production wrapper logic, no helper code, and no README/user-doc updates are included in `M35-P1`.
