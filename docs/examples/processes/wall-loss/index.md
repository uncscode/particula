# Wall loss: distinguish removal, redistribution, and geometry

A chamber spectrum can change because particles deposit on walls, merge, grow,
or leave with the airflow. Start with a single mechanism and an interpretable
budget before fitting a combined model to measurements.

| Question | Example | What to inspect |
| --- | --- | --- |
| Did I enter chamber dimensions and coefficients in the right units? | [Builders and factory](wall_loss_builders_factory.py) | Converted geometry and the distinction between s⁻¹ coefficients and concentration rates |
| How does size-dependent removal change a spherical chamber spectrum? | [Spherical wall loss](spherical_wall_loss_strategy.py) | 100-minute decay, per-bin removal, unchanged individual masses |
| How is a rectangular decay step checked? | [Rectangular wall loss](rectangular_wall_loss_strategy.py) | 30-minute trajectory, positivity substeps, repeated-Euler reference |
| What changes with geometry or an added process? | [Geometry and process comparison](wall_loss_tutorial.py) | Nearly matched chamber volumes, separate wall-only and coupled cases |
| Which rates contribute to a combined chamber trajectory? | [Functional forward calculation](chamber_forward_simulation.py) | Instantaneous rate decomposition and the limits of a single combined run |

The [dilution example](../dilution/cpu_dilution.py) provides a useful analytic
baseline: ideal dilution reduces abundance uniformly without shrinking surviving
particles. Wall loss can be size selective, so total-number decay alone cannot
identify the mechanism.

For your chamber, supply measured geometry and justified transport parameters.
Check timestep and radius-grid sensitivity, compare size-resolved survival,
and account for external material loss. A nonnegative trajectory or a successful
configuration check does not establish agreement with experimental data.
