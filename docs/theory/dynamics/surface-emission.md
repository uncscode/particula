# Wind-driven emission from a finite dust bed

How much dust a gust carries depends on how long the wind exceeds the erosion threshold and how much erodible material remains. In the two-gust dune experiment, a second gust of equal strength lifts fewer fine grains because the first exhausts that class. The source transfers dry particulate mass in kilograms between a finite surface patch and the airborne population. [Two-gust example](../../studies/examples/particle-processes/gust_dust_emission.py)

## From wind at a reference height to vertical mass flux

Sample the east and north components of the horizontal air velocity, $u$ and $v$ in m/s, at a declared physical height above the local bed. The dimensionless drag coefficient $C_D$ links this reference wind to a friction velocity $u_*$ and hence to the magnitude of surface shear stress $\tau$ for air density $\rho_a$.

$$
u_* = \sqrt{C_D}\sqrt{u^2+v^2}, \qquad |\tau| = \rho_a u_*^2 .
$$

The user supplies an onset threshold $u_{*t,i}>0$ in m/s and a vertical emission coefficient $E_i$ in kg m⁻² s⁻¹ for each emitted class $i$. This example uses the illustrative law

$$
F_i = E_i \max\!\left(\frac{u_*^2}{u_{*t,i}^2}-1,\,0\right).
$$

$F_i$ is airborne dry mass emitted per unit erodible surface area and time. A threshold can depend on the size of surface grains, their cohesion and gravity. The Shao–Lu analysis treats loosely arranged spherical grains on a dry, bare surface. Saltating parent grains and the aerosols they liberate can have different diameters. Here the input classes specify the *emitted* aerosol, with an independently prescribed threshold and flux coefficient. Applying the law to field conditions requires size-resolved flux and surface observations. [Shao and Lu (2000)](https://doi.org/10.1029/2000JD900304)

## Integrating a gust and depleting the bed

For patch $p$ with erodible slope area $A_p$ in m² and available bed mass $B_{pi}$ in kg, the accepted mass during an interval is

$$
\Delta M_{pi} = \min\!\left[B_{pi},\ A_p\int_{t_0}^{t_1} F_i(t)\,dt\right],
\qquad B'_{pi} = B_{pi} - \Delta M_{pi}.
$$

Use the physical area covered by erodible material, including the represented crosswind thickness of a 2-D slice. Each patch samples its local wind at the horizontal center of its bottom cell. With componentwise linear wind between sounding records, $u^2+v^2$ is quadratic in time. The integral splits at sounding times and at crossings of the threshold. For example, a steadily reversing wind can pass below threshold and resume emitting during one output interval.

The calculation integrates the flux before advancing the interval. Both airborne representations inject its integral at the left endpoint, an operator-splitting approximation for the positions and fates of newly emitted material. Refine accepted intervals to resolve the emission timing relative to deposition and boundary crossings. The offline source precomputes the schedule on the recorded grid. Resolved mode reserves one weighted slot for each nonzero patch–class–interval combination before any mass moves. The checkpoint records the accepted interval, stable record IDs and remaining patch masses.

## Closing the dry-material budget

For a spherical emitted grain of diameter $d_i$ and density $\rho_i$, its dry mass is $m_i=\pi\rho_i d_i^3/6$. A release represents $\Delta N_{pi}=\Delta M_{pi}/m_i$ grains. Fractional weights are meaningful here. If material $s$ contributes fraction $f_{is}$ of the grain mass, its released inventory is $\Delta M_{pi} f_{is}$.

In binned mode the bottom cell receives represented count and dry mass by material. The prescribed field flow then transports those inventories. In resolved mode the same dry mass funds a weighted record at a specified physical height above the bed, with a prescribed downward slip speed. Its subsequent fate can be airborne, deposited or escaped. Deposited grains remain in a separate reservoir. With no other sources, local mass exchange or imports, initial dry bed kg of each class equal current bed, airborne, deposited and exported kg summed together. [Fixed-grid and parcel guide](../../guide/processes/parcels.md)
