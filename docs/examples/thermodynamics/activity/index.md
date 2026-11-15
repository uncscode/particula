# Activity and partitioning examples

Activity connects mixture composition to equilibrium vapor pressure. A useful
first question is whether changing a mixing assumption materially changes the
pressure reduction for the component you care about. The examples below build
from that comparison to water uptake and condensed organic mass.

| Follow this question | Example | Scope of the answer |
| --- | --- | --- |
| How does mass become a mole-fraction activity? | [Ideal activity](ideal_activity_example.py) | A two-component reference and partial-pressure calculation |
| How does a solute's κ affect water activity? | [κ activity](kappa_activity_example.py) | A salt-like example and a separate generic parameter sweep |
| How far can organic activity depart from ideal mixing? | [BAT activity](bat_activity_example.py) | A fixed-descriptor binary mixture at three compositions |
| How do these approaches fit together? | [Activity model walkthrough](activity_tutorial.py) | Matched-component comparisons, water uptake, and partitioning |
| How does activity enter an equilibrium mass calculation? | [Equilibrium partitioning](equilibria_example.py) | Three low-volatility surrogates and a water-activity sweep |

Start with ideal activity if mole fractions and partial pressure are unfamiliar.
Then choose κ for the water-uptake question or BAT for water–organic nonideality.
These models have different domains; a difference between curves is a sensitivity
result, not proof that one is accurate for a particular aerosol.

For the same physical model expressed through different construction interfaces,
use [strategies, builders, and factories](../particle-properties/activity_tutorial.py).
For the meaning of alpha/beta phase outputs and a broader volatility range,
continue to [equilibrium examples](../equilibria/index.md). The
[activity guide](../../../guide/processes/activity.md) and
[activity theory](../../../theory/thermodynamics/activity-theory.md) provide API
contracts and mathematical context.
