# Thermodynamics examples

What controls the equilibrium pressure above a particle, and how does that
pressure affect its composition? Start with activity for mixing, add curvature
for a small particle's surface, then follow partitioning to predict condensed
mass. These are complementary pieces of a model, with different assumptions.

| Your question | Start here | What to look for |
| --- | --- | --- |
| Why is equal mass not equal activity? | [Ideal molar activity](activity/ideal_activity_example.py) | Mole fraction and the flat-interface pressure reduction |
| Which activity model should I compare? | [Activity models together](activity/activity_tutorial.py) | Ideal, κ, and BAT assumptions and matched-component comparisons |
| How does hygroscopicity change water activity? | [κ activity](activity/kappa_activity_example.py) | Composition dependence and sensitivity to the solute parameter |
| What does organic nonideality add? | [BAT strategy](activity/bat_activity_example.py) | Organic activity at fixed O:C and molar mass |
| How do I express those rules in code? | [Strategy, builder, and factory](particle-properties/activity_tutorial.py) | Separate model choice from construction interface |
| What changes along a composition path? | [Activity functions](particle-properties/activity_functions.py) | Mole-, volume-, and mass-based rules on a common coordinate |
| How does radius change equilibrium pressure? | [Surface and Kelvin effect](particle-properties/particle_surface_tutorial.py) | Surface tension, species molar volume, and an imposed mixture rule |
| Can water and organic material occupy different phases? | [BAT phase properties](equilibria/activity_part1.py) | Activities, phase compositions, and phase-transition weights |
| How much organic material condenses? | [Compact partitioning example](activity/equilibria_example.py) | Available mass, volatility, and associated water |
| How does humidity affect a volatility distribution? | [Multicomponent partitioning](equilibria/equilibria_part1.py) | Separate organic and water concentration responses |

The [activity collection](activity/index.md) offers short entry points; the
[equilibrium collection](equilibria/index.md) develops the phase-property to
partitioning connection. Each worked page includes downloadable Python source
and saved results. The [theory](../../theory/thermodynamics/index.md) supplies
model equations and assumptions; the [activity guide](../../guide/processes/activity.md)
explains supported interfaces.

For a research study, choose the material-property model and validate its domain
before interpreting a growth calculation. Equilibrium composition is a useful
reference for [condensation examples](../processes/index.md), but does not tell
you how quickly particles reach that state.
