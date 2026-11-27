# How the six fits were scored and checked

Each simulated minute contributes concentrations in the original 70 water-equivalent Welas bins. The search objective puts 65% of its weight on error across those bins and 20% on integrated counts in six diameter bands. Analyzer vapor and imager size and habit evidence receive 10% and 5% respectively, with the available terms renormalized when a channel is missing. Within the optical term, the represented time stages and eight logarithmic size groups receive equal weight. The six experiments receive equal weight in the shared fit. The reported optical errors isolate the spectral term from the rest of the objective.

The shared and individual searches each used two separated starts. Automatic differentiation of a local approximation to the objective proposed changes after paired perturbations of the particle solver. Leading candidates were reranked at 2 s. The selected parameters were checked at 1 s with doubled CCN, salt and ice size resolution. The 70 observation bins stayed fixed. Representative CPU/CUDA runs agreed to within 0.000001 decades in whole-record optical error. Across retained primary and comparator replays, maximum absolute water and particle budget residuals were 0.00134 g/m³ and 0.0357 cm⁻³.

## How well early parameters describe the last ten minutes

![Measured and simulated size distributions over the final ten valid minutes of six expansions](../../../images/jeju-chamber/forward-distributions.png)

*Figure 1. All curves average the same ten valid measured minutes in each run. The physical models start from their initial pre-cloud spectra and continue with later measured forcing. Their parameter selection uses the earlier minutes. A separate simple baseline holds the final observed pre-cutoff spectrum constant.*

The mean late error was 0.934 decades with pooled fits, 0.970 with independent fits and 0.971 when only aerosol and initial conditions varied by run against common physics. Persistence gave 0.263 decades. The model and its fixed instrument assumptions had previously been studied with these six records. These tests measure transfer along each observed expansion under known forcing. An independently measured experiment would be needed to assess wider predictive skill.

A detection-floor check and a restriction to occupied measurements test how the apparent improvement depends on low counts. Removing empty bins also removes the penalty for producing particles where the counter measured none. Both views are informative about different parts of the distribution. [Sensitivity to low counts and neighboring minutes](spectra.md)

[Six-run results and conclusions](index.md)
