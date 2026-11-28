# Which dry particles can activate at a given water supersaturation?

At 0.2% water supersaturation, a spherical seed with hygroscopicity $\kappa=0.3$ and diameter-basis Kelvin length $A=2.1$ nm has a critical dry diameter of 104.54 nm. The water droplet at the top of its equilibrium Köhler curve is already about 702 nm across. Use the first diameter when selecting dry seeds from a population. The second describes the amount of growth needed to reach the peak.

## The equilibrium barrier

Let $D_d$ be the volume-equivalent diameter of the dry, non-water material and $D$ the wet diameter. Assume additive material volumes, a wettable surface, and constant temperature, surface tension and effective $\kappa$ during water uptake. At each wet diameter, dissolved material lowers water activity while surface curvature raises equilibrium vapor pressure. Their product gives the ratio of equilibrium water vapor pressure to saturation vapor pressure over flat pure water at the same temperature. [Petters and Kreidenweis (2007), Eq. 6](https://doi.org/10.5194/acp-7-1961-2007)

$$
S_{\rm eq}(D;D_d)=
\frac{D^3-D_d^3}{D^3-D_d^3+\kappa D_d^3}
\exp\!\left(\frac{A}{D}\right),
\qquad A=\frac{4\sigma M_w}{RT\rho_w}.
$$

Here $\sigma$ is the solution–air surface tension (N m⁻¹), $M_w$ the water molar mass (kg mol⁻¹), $R$ the gas constant (J mol⁻¹ K⁻¹), $T$ the temperature (K) and $\rho_w$ the liquid water density (kg m⁻³). Both diameters and $A$ have units of metres. Particula's Kelvin **radius** is $A/2$. Using that radius in the diameter formula halves the predicted dry threshold.

For a seed of known dry size, the critical ratio is the highest $S_{\rm eq}$ over $D>D_d$. A seed below the critical dry size encounters a peak higher than the available vapor ratio as it takes up water. A larger seed of the same composition has a lower barrier. At a prescribed ratio $S>1$, define the critical dry diameter by equality between the peak and $S$. Equality counts as the threshold in an equilibrium population calculation.

## Invert the peak for dry diameter

At any crossing $S_{\rm eq}=S$ with $\kappa>0$, set $L=\ln S$ and $t=A/D-L$. Taking the logarithm of the activity term gives $e^t-1=\kappa D_d^3/(D^3-D_d^3)$. Substituting $D=A/(L+t)$ expresses the dry diameter of that crossing entirely in terms of $t>0$.

$$
d(t)=\frac{A}{L+t}
\left(1+\frac{\kappa}{e^t-1}\right)^{-1/3},
\qquad D_{d,\rm crit}=\max_{t>0}d(t).
$$

Why take the largest crossing diameter? At fixed wet diameter, adding dry material lowers the water activity. A seed larger than every crossing therefore has $S_{\rm eq}<S$ at every point on its growth path. For positive $\kappa$, $d(t)$ approaches zero at each endpoint, so the largest interior maximum sets the threshold.

At $\kappa=0$, a film of pure water covers the insoluble seed. The forward curve reduces to $\exp(A/D)$ and its upper limit is reached as $D$ approaches $D_d$. This gives the exact threshold $A/\ln S$. For positive $\kappa$ and a critical droplet much larger than its dry seed, expanding the water activity instead gives the dilute approximation $D_{d,\rm crit}\simeq A[4/(27\kappa L^2)]^{1/3}$. The approximation diverges at $\kappa=0$ while the full threshold stays finite for each fixed $S>1$. [Petters and Kreidenweis (2007), Fig. 1 and Eq. 10](https://doi.org/10.5194/acp-7-1961-2007)

## From one dry seed to a CCN spectrum

For a mixed seed, let $V_j=m_j/\rho_j$ be the dry volume of material $j$. With explicit material hygroscopicities, the volume mixing rule gives $\kappa_{\rm eff}=\sum_j V_j\kappa_j/\sum_j V_j$. The same dry volumes give $D_d=(6\sum_j V_j/\pi)^{1/3}$. Exclude condensed water from both sums. Adding only water then increases wet diameter while leaving dry diameter and $\kappa_{\rm eff}$ fixed. [Petters and Kreidenweis (2007), Eq. 7](https://doi.org/10.5194/acp-7-1961-2007)

For a population represented by counts $n_i$ in sample volume $V_{\rm air}$, sum the counts of occupied particles with positive dry volume that meet their individual thresholds. Repeating this calculation over prescribed saturation settings gives the equilibrium CCN spectrum in m⁻³.

$$
N_{\rm CCN,eq}(S)=\frac{1}{V_{\rm air}}
\sum_i n_i\,\mathbf{1}\!\left[D_{d,i}\ge D_{d,\rm crit}(\kappa_i,S,A)\right].
$$

When composition varies across particles, evaluating a single mean $\kappa$ can move particles on either side of the cutoff. Recording the weighted sum captures the effect of these differences at the sampled settings. To evaluate another saturation setting later, save the individual dry sizes, composition and represented counts together with the sample volume and Kelvin length or its inputs. To predict an instrument count, model the sampled size distribution, inlet hydration, growth during the residence time and optical detection. [Worked example with live and saved populations](../../studies/examples/thermodynamics/activity/equilibrium_ccn_spectrum.py) · [Guide to designing a size-resolved CCN experiment](../../guide/processes/ccn-retrieval.md)
