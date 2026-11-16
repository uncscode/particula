# Literature scheduler: continuous discovery and comparison

Status: proposed; no automation has been created or enabled by this plan.
Placement: independent of the [development stages](README.md).

## Purpose

Periodically identify aerosol papers with accessible data or sufficiently
specified simulations that Particula could reproduce. Select a suitable study,
develop a reproducible comparison, validate the work, and create a PR. Use the
scientific requirements in
[Stage 3B](stage-03b-scientific-comparisons.md) and retain a small candidate queue
and evidence for selecting future physics work.

When activated, each selected study iteration follows the same
[worktree, branch, and PR workflow](README.md#worktree-branch-and-pr-workflow)
as the staged plans. The difference is that **each new iteration creates a
fresh worktree and branch**, rather than reusing a single scheduler branch.

## Proposed cadence and activation

- Start with one search per week; consider twice weekly if the candidate yield
  and review capacity justify it. The cadence is a proposal, not a saved setting.
- Choose the day/time in America/Denver and the delivery destination when
  activating the scheduler. Reuse or update an existing matching automation
  rather than creating a duplicate.
- Set a persistent candidate/iteration record location accessible from all
  worktrees, a maximum number of active studies, and per-study runtime,
  CPU/GPU, and storage budgets. Keep the record outside disposable worktrees.
  These are activation settings, not unlimited resource allocations.
- At activation, use the application's automation controls and confirm the
  resulting saved schedule. No scheduler script or operating-system cron job
  is part of this repository plan.
- The scheduled instructions must include the full search-to-PR workflow
  below, not only a request for a shortlist. Editing this plan alone does not
  activate a schedule or start an implementation run.

## Search and screening scope

Search primary papers, publisher supplements, author-maintained data/code
repositories, and archived model inputs. Include older reproducible reference
studies as well as new publications; recency alone is not the selection goal.

Initial topics are condensation/evaporation, coagulation, chamber growth and
loss, charged particles/ion interactions, and organic/water thermodynamics.
Reassess topics against the current capability and support guides at each run.
An unsupported capability must be labeled as a gap rather than inferred from a
similar function name or an earlier development plan.

For each candidate, check:

1. Accessible data or reference output, with reuse conditions and provenance.
2. Sufficient initial conditions, forcing, material properties, and numerical
   or experimental setup to make a meaningful comparison.
3. Physics and observables that match supported Particula capabilities, or a
   clearly bounded missing capability.
4. A defined figure, table, time series, or metric to reproduce.
5. Scientific value, approximate effort, and likely ambiguity or uncertainty.

## Screening report and iteration record

Return at most three strong candidates per run, or state that no suitable new
candidate was found. For each include the citation and source links, scientific
question, target result, data/code availability, required physics, compatibility
gaps, and a short recommendation. Separate verified information from inference
and do not claim to have reproduced results during screening.

Maintain a deduplicated record of previously screened/selected candidates in
the agreed destination. Distinguish ready now, needs bounded development,
insufficient information, in progress, PR open, and already evaluated. Record
the DOI or stable paper identifier, iteration ID, target comparison, worktree
path, branch/base revision, and PR URL when available. Check this record and
existing worktrees/open PRs before starting work. Revisit a completed candidate
only when new evidence or new model support changes its suitability.

Use a single scheduler writer or an atomic claim in that shared record so
overlapping invocations cannot select the same study simultaneously. Save the
claim before creating the worktree and record progress after each durable
milestone. Reconcile interrupted claims with existing branches and PRs before
retrying; recover the same iteration rather than silently abandoning it.

## Execution: one new worktree per selected iteration

1. **Select a bounded study.** From the screened candidates, select at most
   one new study per scheduler run. Define its target result, input provenance,
   comparison metrics, uncertainty, and expected computational cost before
   implementation. Prefer supported physics; identify any bounded corrections
   needed for the comparison. If no candidate is suitable, report that outcome
   without creating an empty development branch or PR.
   Check available study capacity and the configured compute/storage budgets;
   keep additional candidates queued when capacity is exhausted. If a study
   reaches its budget, preserve its work and report the remaining requirement.
2. **Create the iteration's worktree and branch.** Follow the shared workflow
   to verify remotes and the current upstream base. Assign a unique iteration
   ID and create a new worktree with a branch such as
   `codex/literature-<YYYYMMDD-HHMMSS>-<paper-slug>`. Keep each study's changes,
   tests, and evidence in that worktree. Do not reuse a previous study's
   worktree or branch, or use the main checkout for its implementation.
3. **Develop the comparison.** Implement the executable example, input/data
   provenance, analysis, and relevant tests. Include bounded library corrections
   when needed, with independent regressions and the required backend checks.
   Use the maintained example/output workflow and scientific boundaries in
   Stage 3B. A substantial missing capability becomes a separately scoped
   development plan rather than an unsupported substitute model.
4. **Validate and interpret.** Run the comparison and relevant repository
   checks. Preserve inputs, commands, revision, numerical settings, outputs,
   uncertainty, and limitations in the maintained example and validation
   records. Distinguish agreement, measured disagreement, and insufficient
   evidence; do not tune acceptance criteria to manufacture agreement.
5. **Deliver a PR.** Once the scoped study and required checks are complete,
   commit and push the iteration branch and create its own PR. Include the
   paper citation, target result, reproduction instructions, observed outcome,
   validation, and remaining limitations. A scientifically useful disagreement
   can be a completed study; it must not be described as successful physical
   validation. Return the PR URL and update the iteration record.

An interrupted run resumes the same iteration's worktree and branch. A newly
selected study, or a justified new evaluation of a previously completed paper,
gets a new iteration ID, worktree, branch, and PR. Do not launch a duplicate
while that study is already in progress or has an open PR; continue its existing
work and report its status. Retain incomplete work and record blockers without
claiming completion.

## Handoff and success criteria

A selected candidate applies Stage 3B's comparison specification and declared
acceptance metrics in its own iteration. A feature gap becomes a scoped
proposal under
[thermodynamics](stage-03a-thermodynamics.md),
[later chemistry](stage-06-chemistry-model.md), or the appropriate numerical
plan. Selection does not imply successful validation.

The scheduler is useful when selected studies produce reviewable PRs containing
reproducible comparisons and clear scientific conclusions. Each completed study
must have its own recorded worktree/branch identity, evidence, and PR. Evaluate
candidate quality and completed comparisons, not the number of papers found.
