# Why temperature differences broaden the chamber spectrum

Particles circulating through regions at different temperatures can grow to different sizes before Welas samples them. In an early nine-box experiment, this mechanism reproduced the average width of the observed droplet band. Size-dependent wall loss and collisions in a well-mixed chamber recovered less of that width. The comparisons below ask whether each mechanism can broaden the optical spectrum under the measured chamber conditions.

## Wall loss and collisions in a well-mixed chamber

We measured width as the logarithmic distance between the diameters below which 10% and 90% of the particles fell, using optical counts from 2 to 40 µm. Across the same 163 minutes, the original one-box fit gave a mean width of 0.204 decades and Welas gave 0.393. Refitting with Particula’s size-dependent wall loss raised the simulated width to 0.256 decades. That fit also increased the error in particle number.

Adding Brownian collisions with van der Waals attraction, differential settling and turbulent shear gave a refitted width of 0.263 decades. To isolate what collisions did, we switched them on and off at identical fitted settings. Turning them on slightly narrowed the band on all three numerical grids. The six independent chamber fits also spanned more than five orders of magnitude in the eddy rate controlling particle loss at the walls. A size-resolved deposition measurement would help determine which rates are physically plausible.

## Different cooling histories widen the sampled distribution

Across 166 minutes common to the circulation comparison, the simulated mean width rose from 0.205 decades in one box to 0.395 in nine exchanging boxes, against 0.393 measured. The nine-box case held particle microphysics fixed and selected a circulation that favored agreement in width. Removing the temperature contrast between boxes erased most of the broadening. A separate fit that also constrained particle number improved the width error but increased the error in counts above 40 µm. Matching the width alone therefore leaves uncertainty at the large-particle end.

The temperature probes constrain how different the boxes can be. After minute 18, the median difference between the measured wall and air means was 0.70 K, with the wall warmer. In a later comparison, we imposed the observed difference between the mean and coldest air readings in both 18-cell models. The refined mean spectral error was 0.659 decades in two dimensions and 0.654 in three. Only runs 3 and 6 favored the three-dimensional circulation. The six-run reconstruction uses the two-dimensional model.

![Nine-box circulation and local temperature differences at two stages of run 2, with the simulated heat budget](../../../images/jeju-chamber/chamber-processes.png)

*Figure 1. Chamber-mean thermometry constrains the heat budget. The arrows show the prescribed flow, and the local temperature differences and exchanges of heat come from the model. The experiment did not resolve the velocity field or temperatures on individual wall faces.*

## Would another run’s circulation recover the missing particles?

In the final independent fits, circulation speeds ranged from 0.012 to 0.208 m/s. We substituted each run’s fitted transport settings into the other five, holding the receiving run’s aerosol and microphysics fixed. Across the 30 substitutions, median optical error rose by 0.029 decades on the standard grid. Run 5 was already the poorest fit at 1.188 decades. Borrowing any other run’s transport settings raised its error by only 0.013–0.026 decades. None of those substitutions recovered its spectrum. A tracer pulse sampled at the optical inlet and a second height, alongside local temperature, would help determine which circulation histories the real chamber permits. [Run-5 source and restart tests](dust.md) · [Chamber measurements](https://doi.org/10.6084/m9.figshare.31744108.v1)

[Six-run results and conclusions](index.md)
