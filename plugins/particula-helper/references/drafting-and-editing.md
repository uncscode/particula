# Draft and edit for a Particula reader

The agent responsible for the page owns the brief, the final wording and the
file edits. Use the following role briefs as assignments for host subagents.
For a new study, worked example or substantial guide or theory revision,
delegate a draft and a fresh editorial read when delegation is available.
For an existing passage that only needs editing, send it directly to the
editor. A small correction can use the relevant check inline. If the host
lacks delegation or a worker is unavailable, the owner can draft or review
inline. Use an available independent reader for a fresh edit after drafting.

## Literature reader

A reader question may come from the user, a previous independent literature
analysis, or a reader agent launched for this study. Use a supplied question
or analysis when it fits. When the study calls for a question, launch a
literature reader with the topic, available primary sources and study scope.
Let the reader examine the paper, supplement or author data independently
before seeing the draft's interpretation. In a how-to guide, the task the
user wants to complete can supply the brief directly.

Return the question a researcher could answer, why the answer matters, and
the exact figure, table, equation or data field that motivated it. Identify
the observable, units, conditions and comparison needed. Say what the
source establishes and what the proposed study would have to establish.
If no comparable observations are available, identify a narrower question
the evidence can answer or record the missing evidence. Carry the question
forward as the agent's synthesis, with attribution for the underlying data.

## Drafting agent

The page owner selects and self-approves a brief using the question, intended
reader, page type, scope and available evidence. Give the drafter the brief,
the relevant source locations and any existing text worth keeping. For a long
page, a section drafter can work from the section's purpose, the neighboring
argument and the subset of evidence that belongs there. Have drafters return
clean prose, sources for factual claims and any unresolved gaps to the owner.

Lead a literature comparison with the answer the data support. Name who
measured or calculated each quantity, under which conditions, and what a
reader can infer from the difference. Introduce a worked example's code with
the decision it implements and interpret the result beside its output. For a
guide, start with the reader's task and end with an observable check. For
theory, connect the assumptions to the regime where the equation applies.
Use the page's existing formatting and citation conventions.

Let the scientific finding carry the contribution. Start from what the
original study established, then show what the comparison reveals about a
physical assumption, a prediction or a proposed solution to an open problem. Name
Particula where it identifies a calculation the reader needs to distinguish
or reproduce. Give the scientific consequence its own sentence when it matters.

For example, “In the Π Chamber experiments, the salt-water case had more
cloud droplets than the fresh-water case. With interior forcing held fixed
in the simulations, the predicted difference changed from 362 more droplets
per cm³ with a drier salt-water floor to 379 fewer with a slightly
supersaturated floor.” Floor humidity was an assumption in these runs. A
simulation could now vary it between those limits to locate the reversal.
That calculation would establish the model's sensitivity. The actual floor
humidity would remain unknown.

For literature studies, draft the opening alongside the TOC figure. Use the
first descriptive heading to introduce the original paper. Give the analysis
its own scientific heading and organize its comparisons as subsections.
Write the opening and closing paragraphs with three reader questions in mind.

| Reader question | What the drafter should give them |
| --- | --- |
| What did the comparison show? | In the summary, identify the observable, who measured or predicted it, the decisive condition, and the magnitude or direction of the result. For a qualitative or inconclusive result, state the distinction the evidence could resolve and what remains open. |
| What can I conclude or do with that result? | Answer the opening question, explain why the finding matters for a particular use, and name the assumption or uncertainty that bounds that use. Bring together the strongest evidence already shown in the body. |
| What would change that conclusion? | For a literature study, specify an available dataset to reanalyze or a simulation that tests the remaining uncertainty. Name the observable and the result that would favor one interpretation, reveal a limitation or justify extending the finding. Mark it as proposed. |

For example, two models can agree closely while both overpredict measured
water activity. A reader choosing a model needs to know that switching between
them leaves the discrepancy. If the authors propose refitting one model, a
useful follow-up is to compare its residuals with the original model on
independent compositions at the same temperature. A smaller residual on the
fitting data alone leaves transfer to those other compositions open.

## Editing agent

Give the editor the draft, its brief, adjacent context and access to the
same evidence. Have the editor first say what answer or action a reader
would take away. If that differs from what the evidence supports, fix the
claim or its placement before polishing the sentences. Use the five editing
specialties where they address a difficulty in the passage.

| Specialty | What to look for in Particula prose |
| --- | --- |
| Clarity | Name the actor and observable. Can the reader tell which curve the instrument measured and which the model predicted? |
| Continuity | Carry a recognizable quantity into the next sentence. Replace an ambiguous “this increase” with the quantity that increased. |
| Coherence | Give a paragraph one job. Keep a result with the figure that establishes it and move an untested explanation into the interpretation. |
| Concision | Cut repeated setup and generic claims. Retain conditions, units, uncertainty and qualifiers that limit the result. |
| Cadence | Read dense passages aloud. Split competing comparisons and put the decisive finding where the reader can absorb it. |

For a difficult passage the editor can delegate selected specialties in
parallel against the same text. Reviewers return the passage location, the
reader's difficulty and a proposed repair. The owner reconciles competing
suggestions once, so a later wording change cannot silently undo an earlier
scientific correction.

### Read the study summary and ending on their own

Read the TOC figure and its unheaded 100–200 word summary together. Check
that the following background uses fewer than 300 words to explain the paper's
motivation. Count prose separately, excluding headings, figure credits and
citation labels. Then skim just the headings. Can a reader tell what the
paper investigated, which question the simulations tested, and what to take
away or do next? Replace generic labels with those answers. Check that the
individual analyses sit beneath their parent heading. The scientific
conclusion leads into reproducibility information, study resources or both,
followed by any footnotes.

Read the opening question, summary and last scientific paragraph, including any
future-work passage, before consulting the methods or reproduction notes.
In your own words, answer the three reader questions above. Check those
answers against the most relevant figure, table or source. If you could
answer only because you already knew the paper, identify the missing link
and return replacement prose for that passage.

Look for a sentence praising the software or calling a factor “important.”
Could a reader tell what changed, relative to which reference, and why that
affects the interpretation? Rewrite it around the actual result. If the
calculations reproduced an existing finding, say under which conditions it
held. Claim an extension only when the analysis tested one.

Repair a conclusion that merely repeats the headline by adding the scientific
consequence and its limit. Replace a list of possible extensions with the next
test that would change an interpretation. Check what that test could settle.
A measurement that challenges an assumption may still leave several physical
explanations open. If the evidence gives no basis for a follow-up, state the
unresolved question plainly. Keep the scientific ending
easy to find before setup commands, file inventories or project milestones.
The owner applies the repair and repeats the short read before self-approval.

## Self-approve the revision

Use the request as authority for in-scope editorial decisions. Select the
outline, read relevant context and apply warranted changes without waiting for
the user to approve each stage. Re-read the title, opening, captions and
conclusion against the brief. Trace changed claims to the primary source,
saved numerical output or documented API contract. For example, “consistent
with wall loss” needs evidence of consistency; “caused by wall loss” also
needs evidence distinguishing alternative causes.

Narrow or omit an unsupported claim and record a consequential gap in the
handoff. Stop revising when the page answers the question as far as the
evidence allows. Leave clean copy in the maintained source and use the diff
for review. Follow the owning skill's checks for executable cells, saved
outputs and links. Editorial self-approval covers the copy. Publication,
research review and external actions follow the scope of the original task.
