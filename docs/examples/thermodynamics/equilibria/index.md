# From phase properties to equilibrium aerosol mass

A prescribed humidity does not uniquely specify aerosol composition without a
thermodynamic model and an inventory of available material. These examples show
the two parts of that calculation: how water and an organic component mix, and
how a volatility distribution partitions between gas and condensed phases.

1. [Follow BAT phase properties](activity_part1.py). Compare activities with
   activity coefficients, inspect the phase-transition weight, and evaluate
   phase water fractions at prescribed water activity. The O:C comparison holds
   molar mass and density fixed; it is not a chemical-aging simulation.
2. [Solve multicomponent liquid–vapor partitioning](equilibria_part1.py). Combine
   those properties with five surrogate volatilities and fixed organic loading.
   Read aerosol water and condensed organic mass against their separate axes.

For a shorter introduction to the strategy interface, begin with the
[compact partitioning example](../activity/equilibria_example.py). For the
meaning of activity itself, use [ideal activity](../activity/ideal_activity_example.py)
before interpreting phase behavior.

Water activity corresponds to RH/100 under the flat-interface equilibrium
assumption used here. These examples do not calculate equilibration times,
particle curvature, or the kinetics of forming a second liquid phase. A useful
research extension is to compare equilibrium mass across several organic
loadings, assess solver residuals, and test the predicted humidity response
against observations before coupling it to time-dependent condensation.
