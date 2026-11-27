# The droplet peak and the missing 5–10 µm particles

In unsalted ambient air (run 1), the measured mean spectrum and the simulation fitted across all six runs peaked in the same 14.9 µm bin. Fitting run 1 alone moved the peak up one bin to 16.0 µm. The sulfate–dust fill (run 5) proved harder to reconstruct. Over its 41 measured minutes, the optical counter averaged 388 particles/cm³ at 5–10 µm. The shared fit gave 141 particles/cm³ in that band and the individual fit gave 40.8. Both undercounted particles throughout the windows around minutes 20 and 30. [Jeju experiment](https://doi.org/10.1029/2026GL123395) · [Archived measurements](https://doi.org/10.6084/m9.figshare.31744108.v1)

## Similar total counts can conceal a shift in particle size

![Measured, pooled and individual mean size distributions for the six runs](../../../images/jeju-chamber/mean-distributions.png)

*Figure 1. Each step is one Welas optical diameter bin. Black shows measured number per decade of diameter at chamber conditions. Orange dashes show the shared fit and solid blue the individual fit, averaged over the same valid minutes in each run. Integrating across a band gives its number concentration. The logarithmic vertical axis omits zero.*

In run 2 the shared fit nearly recovered the measured total (298 versus 300 particles/cm³), yet placed the peak at 10.4 µm, above the measured 8.36 µm. At 5–10 µm, it gave only 126 particles/cm³ against 209 measured. The peak also shifted upward in run 5, from 7.78 µm measured to 9.65 µm in the shared fit and 11.15 µm in the individual fit.

## The deficit in run 5 persists across adjacent minutes

![Six chamber runs at minutes 12, 20 and 30 with measured and two full-record fits](../../../images/jeju-chamber/minute-distributions.png)

*Figure 2. Each column shows the one-minute mean starting at the indicated time, on common axes. Minute 12 precedes nominal salt addition and minute 20 follows it in the seeded runs. Run 1 received no added salt. By minute 30 the six expansions have reached different temperatures. The simulated spectra come from uninterrupted, fine-grid fits to the full records. Liquid and ice can occupy the same optical diameters. [Experiment clocks and injection provenance](methods.md) · [Size–time spectra and signed residuals](../../../images/jeju-chamber/six-spectra.png)*

| Run and minute | Measured total (cm⁻³) | Individual total (cm⁻³) | Measured 5–10 µm (cm⁻³) | Pooled 5–10 µm (cm⁻³) | Individual 5–10 µm (cm⁻³) |
|---|---:|---:|---:|---:|---:|
| 2, minute 20 | 669 | 688 | 479 | 188 | 198 |
| 5, minute 20 | 1,173 | 511 | 698 | 117 | 14.9 |
| 5, minute 30 | 795 | 111 | 599 | 47.5 | 17.1 |

*Table 1. Total number and number at 5–10 µm in three of the one-minute distributions above. Even where the individual fit recovers the total in run 2, it undercounts particles in this band.*

Moving the comparison one minute either way leaves the run-5 deficit in both fits. During minutes 19–21 the individual fit gives 8.2–20.3 particles/cm³ at 5–10 µm, against 388–928 measured. At minutes 29–31 it gives 9.8–18.6, against 575–668 particles/cm³ measured. These ranges describe changes between minutes. They locate the discrepancy in time and size but leave the composition of the missing particles unresolved. [Material and restart tests](dust.md)

## The gain from individual fits depends on which bins are scored

The individual fits have lower average optical error under all four scoring rules. The gain for a particular run depends on how the calculation treats sparse measurements. We kept the fitted trajectories fixed and varied the small concentration added to each measured and simulated bin before taking the absolute log₁₀ difference. A larger addition gives small counts less influence. A fourth calculation scores only bin–minutes where measured density reached 0.1 particles/cm³ per decade. It therefore omits the penalty for simulated particles in empty bins. Each calculation weights the represented stages and diameter groups equally. The reference calculation reproduced the archived scores at both resolutions.

| Bins scored | Floor (cm⁻³ per decade) | Runs with lower individual error on the fine grid | Runs with a gain exceeding the paired grid shift¹ | Six-run mean gain (decades) |
|---|---:|---:|---:|---:|
| All original bins | 0.03 | 6/6 | 3/6 | 0.0212 |
| All original bins | 0.10 | 6/6 | 4/6 | 0.0181 |
| All original bins | 0.30 | 5/6 | 2/6 | 0.0152 |
| Measured occupied bins | 0.10 | 5/6 | 2/6 | 0.0402 |

¹ Gain is shared-fit minus individual-fit optical error. A gain exceeds the numerical shift when the smaller gain on the two grids exceeds the absolute change between them. Refinement halves the timestep from 2 s to 1 s and doubles particle resolution. Both grids use the same 70 measurement bins.

Runs 3 and 5 illustrate how a small advantage can disappear. On the fine grid, raising the concentration floor from 0.10 to 0.30 changed the run-3 gain from +0.0035 to −0.0003 decades. In run 5, restricting the reference score to occupied bins changed the fine-grid gain from +0.0183 to −0.0008 decades. The occupied-bin score even reversed direction with refinement, from +0.0026 decades on the standard grid. The choice of score can change which fit ranks first. Both fits still undercount the measured 5–10 µm population. [Underlying chamber fits and checks](fitting.md)

[Six-run results and conclusions](index.md)
