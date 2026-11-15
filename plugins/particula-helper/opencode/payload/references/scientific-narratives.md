# Write a scientific narrative

Use for public examples and theory explanations. Choose the reader's scientific
question and the capability they should leave with before organizing code:
explain a mechanism, distinguish representations, select an approximation,
interpret a diagnostic, or adapt a calculation. A list of APIs is not an outcome.

## Worked examples

Make these answers apparent in natural prose; identical headings, classroom
objectives boxes, and quizzes are not required:

1. What question motivates the calculation, and why would a researcher care?
2. What can the reader do or decide afterward? What knowledge is assumed?
3. What physical system, parameters, units, initial conditions, and exclusions
   make the calculation interpretable? Explain important parameter choices.
4. Why does each meaningful code stage advance the question? Place explanations
   beside the relevant stages. Avoid line-by-line narration of obvious Python.
5. What should the reader notice in each important result? Identify a specific
   trend or comparison, the mechanism or hypothesis, and uncertainty or limits.
6. What answer does the evidence support? Distinguish a calculation that ran,
   conservation, numerical convergence, and physical validation. Never invent a
   quantitative observation or infer missing numbers from a plot.
7. How could the reader adapt it? Give a concrete question, parameter or model
   change, diagnostic to compare, and additional evidence needed. Mark unrun
   extensions as proposals rather than results.

Preserve the source-to-result contract. Prose-only edits must not accidentally
change executable cells or output association. Explain figures near the figure;
when a function's output is emitted at invocation, place interpretation there.
End by answering the opening question, not by repeating the steps performed.

Example design: "Why can two plots represent the same population but look
different?" leads from density to bin integration and coordinate transformations.
The outcome is choosing the correct measure and preserving integrated number.
A useful extension is comparing an instrument spectrum with model input while
checking diameter/radius and linear/logarithmic coordinates. Avoid a generic
claim that a successful conversion establishes measurement accuracy.

## Other page roles

Theory: physical question, assumptions, equations and units, limiting behavior,
consequences, sources, and links to worked calculations. Keep long operational
instructions in the owning Guide. Scientific comparisons must identify reference
provenance and distinguish model differences from implementation errors.

How-to guidance: a concrete task, prerequisites, necessary choices, and a result
check. API reference: precise contracts and concise examples. Do not impose the
worked-example narrative on every reference entry or release evidence record.

## Review and evidence

Read the opening, code stages, figures, and conclusion as one argument. Check
that they describe the same calculation. A heading called Summary is not proof
of closure. Ask whether an unfamiliar reader can explain the result, its domain
of applicability, and one reasonable next investigation. Topic links should
supply prerequisites and next concepts rather than interrupting every paragraph.

These are adaptations of [Diátaxis](https://diataxis.fr/), the Carpentries'
[backward design](https://carpentries.github.io/lesson-development-training/lesson-design.html)
and [scientific narrative guidance](https://carpentries.github.io/lesson-development-training/narrative.html),
and [computational essays](https://arxiv.org/abs/1909.12697). The latter reports a
physics-education pilot; it is supporting motivation, not evidence that this
Particula framework has already improved reader outcomes.
