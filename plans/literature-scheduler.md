# Literature scheduler: continuous discovery

Status: proposed; no automation has been created or enabled by this plan.
Placement: independent of the [development stages](README.md).

## Purpose

Periodically identify aerosol papers with accessible data or sufficiently
specified simulations that Particula could reproduce. Supply a small, useful
candidate queue for [scientific comparisons](stage-03b-scientific-comparisons.md)
and evidence for selecting future physics work.

This is a discovery and screening activity. It does not automatically launch
simulation campaigns, alter the library, contact authors, or commit to new
features.

## Proposed cadence and activation

- Start with one search per week; consider twice weekly if the candidate yield
  and review capacity justify it. The cadence is a proposal, not a saved setting.
- Choose the day/time in America/Denver and the delivery destination when
  activating the scheduler. Reuse or update an existing matching automation
  rather than creating a duplicate.
- At activation, use the application's automation controls and confirm the
  resulting saved schedule. No scheduler script or operating-system cron job
  is part of this repository plan.

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

## Expected report

Return at most three strong candidates per run, or state that no suitable new
candidate was found. For each include the citation and source links, scientific
question, target result, data/code availability, required physics, compatibility
gaps, and a short recommendation. Separate verified information from inference
and do not claim to have reproduced results during screening.

Maintain a deduplicated record of previously screened/selected candidates in
the agreed destination. Distinguish ready now, needs bounded development,
insufficient information, and already evaluated. Revisit a candidate only when
new evidence or new model support changes its suitability.

## Handoff and success criteria

A selected candidate enters Stage 3B with a comparison specification and
declared acceptance metrics. A feature gap becomes a scoped proposal under
[thermodynamics/chemistry](stage-03a-thermodynamics-and-chemistry.md) or the
appropriate numerical plan. Selection does not imply successful validation.

The scheduler is useful when its shortlist leads to reproducible studies and
clear scientific decisions without flooding the development queue. Evaluate
candidate quality and successful handoffs, not the number of papers found.
