---
name: particula-reaction-discovery
description: Discover precursor-compatible aerosol reaction networks by screening discrete pathways, fitting rates with automatic differentiation and testing candidates in CUDA chamber replays. Use for primary chemistry, product aging, cross-reactions, and evidence for competing mechanisms.
---

# Discover and test aerosol reaction networks

Start with the measured precursors, accessible product or phase measurements, forcing, and the residual the new chemistry should explain. For joint fits across experiments, keep intrinsic chemistry shared and bound run-specific inputs such as injection uncertainty. For setting up multiple chamber observables and physical controls use [chamber analysis](../particula-chamber-analysis/SKILL.md).

## Search one addition at a time

1. Replay the incumbent and define a score per experiment and observable. Give the incumbent the same refit opportunity as each extension. Batch inexpensive bulk trajectories over candidates, rates, runs and doses on CUDA. Include zero rate as an exact nested control. A bounded multistart or population search can supply several starting points for local gradients.
2. Construct a discrete bank of precursor-compatible reactions or small connected sequences. Label species, phase, stoichiometric coefficients, rate law, units, bounds, product volatility and any oxidant reservoir. Test secondary and tertiary chemistry where the products can form. For example, let a pinene-derived pool dimerize, then let its dimer oxidize or cleave into daughters of different volatilities. Test a connected motif together when its downstream rate has zero leverage before the intermediate exists. Track molecules or carbon consistently through fragmentation, oxidation and phase transfer. Declare the external oxygen source or the budget a lumped yield actually closes.
3. Fit continuous parameters for each proposed topology. Check that switching off its added channels reproduces the parent trajectory. Record the proposed equation, fitted change, and residual it explains before searching the primary literature. Compare substrate, phase, oxidant, conditions, products and rate units. Record a direct analogue, family resemblance, or unresolved hypothesis. Treat a fitted effective rate as a parameter of the modeled pools. Keep the unrestricted best fit and the literature-supported subset available as separate comparisons.
4. Admit additions on paired gain relative to refitting the parent under matched bounds, optimizer effort and numerical settings. Carry the incumbent forward as an eligible solution. Repeat within the requested budget. Stop when gains fail to exceed refit variability or numerical resolution, or when the observables cease to discriminate. Retain the more parsimonious explanation when fits are indistinguishable.

## Use gradients where the objective supports them

For a fixed topology write the continuous tendency as `dc/dt = S v(c, k)`. Bind the species order, observation operator and units. Use bounded or transformed rates with explicit zero-channel controls. On a smooth, supported subproblem, differentiate the integration and loss with respect to rates and shared chemistry. For linear first-order aging over a fixed interval, a differentiable matrix exponential propagates the inventories. State which sources, phase partitioning and population state stay fixed. A CUDA forward solver alone establishes no gradient support. Verify the installed backend, AD operation, precision and device for each step. Consult the selected checkout's GPU support matrix and gas/resolved chemistry guides via [version-matched sources](../../references/sources.md).

Check a feasible direction against centered finite differences over several step sizes, or a one-sided difference at a bound. Verify the zero-rate limit and compare an isolated reaction with its analytic solution. Document the differentiated loss if a smooth surrogate replaces a maximum, event threshold or discrete dose selection. Score proposals under the original loss.

For draws that decide births, wall deposition or collision pairs, use shared random numbers to compare proposals. Options include a conditional gradient for continuous aging at fixed inventories, or a score-function estimate of a smoothed search objective over parameter perturbations. A score-function gradient with fixed sampled losses differentiates their sampling distribution. Check it against the corresponding identity. Replay the proposed parameters through the full stochastic model and compare on fresh seeds and finer resolution before reporting the gain.

## Try a two-reaction CUDA fit

From this skill directory, run `python scripts/fit_aging.py` with NumPy, SciPy and CUDA-enabled PyTorch. The synthetic example starts with one unit of dimer carbon, fits oxidation and cleavage to species-resolved samples, and compares against a zero-cleavage control. It checks the AD directional derivative and carbon conservation. Swapping the rates gives identical suspended carbon at every time despite different intermediates. Use this check when deciding whether total mass alone can resolve a proposed chain.

## Decide what the measurements identify

Refit after removing a channel. Profile rates jointly with competing rates, wall loss and uptake. Compare product flux, surviving inventories, and instrument-visible signals. A dimer oxidation/cleavage rate swap can leave total mass unchanged while changing the products. Report this ambiguity and the composition or volatility measurement that could resolve it. Deliver the selected equations and parameters, literature crosswalk, baseline and control scores by run, gradient checks, and commands for replay. Keep trial populations and optimizer checkpoints local unless requested.

In Manchester the discrete search preceded the AD refinements. Warp differentiated local kinetic and search objectives. A separate PyTorch calculation differentiated 50 downstream rates through conditional aging on CPU. The particle trajectories and final candidate comparisons ran on CUDA. These are reusable examples of distinct derivative scopes. [Search controls](https://github.com/uncscode/particula/blob/7d4fefbfcc593d6e5aafb9d6ecb4e8aa2f4d3645/studies/voliotis-2022/joint/chains/METHODS.md) · [AD and particle replay](https://github.com/uncscode/particula/blob/7d4fefbfcc593d6e5aafb9d6ecb4e8aa2f4d3645/studies/voliotis-2022/joint/resolved/METHODS.md)
