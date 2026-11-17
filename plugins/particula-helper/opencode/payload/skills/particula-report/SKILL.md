---
name: particula-report
description: "Investigate and prepare a developer-facing GitHub issue for a researcher's suspected Particula bug, unexpected result, or perceived scientific error. Use for reporting concerns, not implementing fixes or submitting code PRs."
---

# Investigate and report a concern

Help the researcher communicate an actionable concern without requiring them
to prove a bug or develop a fix. Read `../../references/scientific-practice.md`
and `../../references/sources.md` for scientific checks and version-matched
source discovery. Use `../../references/issue-report.md` for the report format.

## Investigate within the researcher's environment

1. Establish the intended calculation, observed discrepancy, expected behavior
   and research impact. Record the selected interpreter, Particula version,
   installation source/commit and relevant dependency/device details. Use
   `../../scripts/doctor.py` when local execution is available; inspect its
   output before including any of it in a public report.
2. Search version-matched documentation and the owning installed source. Follow
   relevant callers and inspect nearby tests when available. Use CodeGraph in
   an indexed checkout where available; otherwise use ordinary source search.
   A full development checkout is optional. Cite source paths and symbols,
   preferably commit-pinned links. Separate installed-release evidence from
   current upstream code; never infer that a main-branch fix is installed.
3. Reduce to a small runnable example with synthetic or shareable inputs when
   feasible. Preserve the original study and environment. Check relevant units,
   population measures, physical assumptions, conservation, convergence, seeds
   and backend support. Do not silently upgrade, switch backends, alter the
   research data or patch the library to make the discrepancy disappear.
4. Search open and closed issues and relevant PRs/fixes in
   https://github.com/uncscode/particula using the error, symbols and symptom.
   Read likely matches before calling something a duplicate. Record useful
   links and whether any fix applies to the researcher's version. If online
   access is unavailable, explicitly mark the duplicate search as not checked.
5. Separate observed results, hypotheses and confirmed findings. Report what
   was actually run and its outcome. If reproduction or source access is
   unavailable, say so and identify the next useful diagnostic step. Stop once
   the report scopes the concern sufficiently for developer triage; a fix or
   exhaustive investigation is not required.

## Review and submission

Draft a concise symptom-based title and the complete issue body using the
shared format. Retain its headings; mark missing information as unknown or
not tested and omit irrelevant detail. Do not automatically classify an
unconfirmed concern as a confirmed bug or apply a bug label.

Show the researcher the destination, title and complete sanitized report for
review. Ask for approval of that report before posting, even if the initial
request was to file an issue. A matching issue should receive a proposed
evidence-bearing comment for approval instead of an automatic duplicate issue.
If a fix already exists, explain the applicable version and let the researcher
decide whether additional reporting is useful; do not upgrade their environment.

After approval, use available authenticated GitHub tools to create the issue
in `uncscode/particula`, or add the approved comment to the agreed issue.
Approval applies to the reviewed content and destination; substantive changes
need another review. Check the returned result and provide the issue/comment
link. If posting fails ambiguously, check for an existing submission before
retrying. Without GitHub write access, provide the exact copyable Markdown and
the repository's new-issue link; do not claim it was submitted.
