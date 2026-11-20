# %% [markdown]
# An evaporation model must explain both how quickly vapor leaves a drop and where the energy for that evaporation comes from. Testing those predictions together is useful when interpreting drop experiments or choosing heat and mass transfer models for sprays: a plausible evaporation rate can still be incompatible with the measured temperature.
#
# Here we compare published transport correlations with a **continuously fed pentane drop evaporating into warm, dry air**. The reported surface temperature and dimensionless evaporation rate do not agree with this reconstruction. The convection-only model predicts a cooler drop and a lower flux; sustaining the reported-flux estimate would require about **10 kW m⁻² of additional net heat**. That is a diagnostic of unresolved experimental inputs or model assumptions, not a measurement of an unreported heat source.
#
# ## What the experiment measures
#
# [Chuchottaworn, Fujinami and Asano (1984)](https://doi.org/10.1252/jcej.17.7) suspended volatile liquid drops from a supply tube in a stream of dry air. They adjusted the liquid feed so that drop size remained approximately constant, then took that steady feed rate as the evaporation rate. They also measured drop and surrounding gas temperatures, drop dimensions and air velocity. The authors estimated surface area from an ellipsoidal shape and found no appreciable temperature variation inside the drops in their measurements.
#
# This review examines one reported pentane condition accompanying a measured gas-temperature profile. We use its surface temperature, flow descriptors and dimensionless evaporation rate to test heat–mass consistency. We do not compare calculated spatial temperature profiles with the measured profile, or claim validation against the paper's full evaporation dataset.
#
# ![Conceptual experiment: a continuously fed pentane drop receives heat from warm air while vapor leaves its surface. The model tests vapor transport and latent heat supply separately.](../../../../images/pendant-evaporation/experiment-overview.svg)
#
# *The illustration explains the experiment and model comparison. It is not an apparatus drawing to scale. Feed enthalpy, tube conduction and radiation are possible net heat inputs that must be constrained separately.*
#
# ## Model setup and goals
#
# The goal is to predict a steady **surface temperature and evaporation flux without fitting either reported value**, then identify which assumptions could change the answer.
#
# | Quantity | Value used | Role in the comparison |
# | --- | --- | --- |
# | Liquid | Pure pentane | Vapor pressure determines the surface vapor concentration. |
# | Ambient temperature, $T_\infty$ | 326 K | Reported warm-air condition. |
# | Surface temperature, $T_s$ | 263.6 K | Reported comparison target; allowed to vary when finding equilibrium. |
# | Diameter, $d$ | 1.61 mm | Reported horizontal diameter, used here as a spherical length and area proxy. |
# | Surface Reynolds number, $\mathrm{Re}$ | 60 at the reported temperature | Anchors the gas speed for each property configuration. |
# | Mass injection ratio, $\phi$ | 0.080 | Reported dimensionless rate from which we reconstruct a flux. |
# | Prandtl number, $\mathrm{Pr}$ | Calculated locally; 0.86 in a separate sensitivity | Tests the impact of the reported heat-transport input. |
# | Pressure, $P$ | 101325 Pa | Standard-atmosphere approximation to the paper's atmospheric-pressure experiments. |
#
# We represent dry air as nitrogen, oxygen and argon, with negligible incoming pentane vapor. The surface vapor is saturated. Gas speed stays fixed as surface temperature changes; its inferred value is 0.290–0.304 m s⁻¹ across property methods. Reynolds number therefore changes along each curve.
#
# The convection-only baseline assumes a steady drop whose incoming heat is used entirely for latent evaporation. Net heat from the liquid feed, support tube and radiation is set to zero. The supplied liquid maintains the drop size; this is not a freely shrinking drop calculation.
#
# All properties come from the locally installed thermo and chemicals databases through the study's chemical-property workflow. Saturation vapor pressure, latent heat, diffusivity, viscosity, conductivity and heat capacity vary with temperature. Sixteen combinations of collision data, viscosity treatment and gas conductivity mixing test method sensitivity. Properties are evaluated at surface conditions, following the paper's stated property convention. The paired drop height and original per-case property records have not been recovered, so the authors' ellipsoidal geometry is not reconstructed.
#
# ## Two constraints must meet at one temperature
#
# ### Vapor transport determines the mass flux
#
# Let $j_m$ denote outward mass flux in $\mathrm{kg\,m^{-2}\,s^{-1}}$. The reconstructed 1984 mass-transfer correlation is
#
# $$
# j_m(T_s)=
# \frac{\rho_s D B_M}{d}
# \left(2+0.37\,\mathrm{Re}^{0.61}\mathrm{Sc}^{0.51}\right)
# \left(\frac{\rho_\infty}{\rho_s}\right)^{1/3}
# g(B_M).
# $$
#
# Here $D$ is the mixture-averaged vapor diffusivity, $\rho_s$ and $\rho_\infty$ are surface and ambient gas densities, $\mu$ is surface gas viscosity, and $U_\infty$ is the free-stream gas speed. With vapor mass fraction $\omega$, the dimensionless inputs are
#
# $$
# B_M=\frac{\omega_s-\omega_\infty}{1-\omega_s},
# \qquad
# \mathrm{Re}=\frac{\rho_s U_\infty d}{\mu},
# \qquad
# \mathrm{Sc}=\frac{\mu}{\rho_s D}.
# $$
#
# Evaporation drives an outward gas flow, often called *blowing*, which modifies heat and mass transport. We compare the authors' blowing correction with a Stefan film expression:
#
# $$
# g_{1984}(B)=\frac{1}{0.3+0.7(1+B)^{0.88}},
# \qquad
# g_{\mathrm{film}}(B)=\frac{\ln(1+B)}{B}.
# $$
#
# Both tend to one as $B\rightarrow0$. We substitute the film expression on the **same mass-transfer baseline and density correction**, so this comparison isolates the correction. It is not a separate exact solution for the pendant geometry.
#
# The reported injection ratio provides a second flux estimate:
#
# $$
# j_{\mathrm{reported}}
# =\phi\,\rho_s U_\infty
# =\frac{\phi\,\mathrm{Re}\,\mu}{d}.
# $$
#
# This estimate inherits our property and spherical-diameter assumptions. It is not an independently recovered absolute-flux measurement.
#
# ### Heat supply limits the flux that convection can sustain
#
# The Ranz–Marshall heat-transfer baseline gives
#
# $$
# \mathrm{Nu}_0=2+0.6\,\mathrm{Re}^{1/2}\mathrm{Pr}^{1/3},
# \qquad
# \mathrm{Pr}=\frac{\mu c_p}{k},
# \qquad
# q_0''=\frac{k\,\mathrm{Nu}_0}{d}(T_\infty-T_s),
# $$
#
# where $k$ is gas thermal conductivity, $c_p$ is mass-specific gas heat capacity, and $q_0''$ is heat flux in $\mathrm{W\,m^{-2}}$ before a blowing correction.
#
# The [companion theory paper](https://doi.org/10.1252/jcej.17.1) defines a distinct thermal transfer number. At a steady state supplied only by convection, $q''=jL$, where $L$ is latent heat in $\mathrm{J\,kg^{-1}}$. Thus,
#
# $$
# B_H=\frac{j c_p(T_\infty-T_s)}{q''}
# \quad\Longrightarrow\quad
# B_H=\frac{c_p(T_\infty-T_s)}{L}.
# $$
#
# Using this **thermal** number, the heat-equivalent evaporation flux is
#
# $$
# j_h(T_s)=\frac{q_0''}{L}
# \begin{cases}
# 1, & \text{no blowing correction},\\
# g_{1984}(B_H), & \text{1984 correction},\\
# g_{\mathrm{film}}(B_H), & \text{Stefan film correction}.
# \end{cases}
# $$
#
# The predicted steady temperature satisfies
#
# $$
# \boxed{j_m(T_s)=j_h(T_s).}
# $$
#
# Away from an intersection, $j_h$ describes the evaporation that the assumed heat supply could sustain. It does not give the actual heat flux at an independently imposed evaporation rate. Applying the companion blowing correction to the Ranz–Marshall heat baseline is an explicit model combination, rather than a reproduction of the companion paper's full numerical solution.
#
# ## Run the reproducible comparison
#
# This example requires Python 3.12+, Particula, NumPy, SciPy, Matplotlib,
# thermo, chemicals, and Warp with an NVIDIA CUDA device. Use the downloadable
# ZIP to include the helper modules. Run from its extracted directory, or
# from the repository root. This workload uses steady algebraic correlations,
# not the native time-dependent condensation solver.
#
# The study helper computes temperature-dependent properties on CPU, evaluates
# the correlations and reverse derivatives on CUDA, and brackets equilibrium
# temperatures on CPU. It runs 16 property configurations over 321 temperature
# points, checks NumPy/CUDA agreement and derivatives, and compares three
# temperature-grid resolutions. The reported-Prandtl sensitivity adds 96 CPU
# root solves. The retained report feeds each figure below.

# %%
import matplotlib.pyplot as plt
from scripts.pendant_evaporation.plots import (
    plot_curves,
    plot_heat_budget,
    plot_sensitivity,
)

# The documentation replot workflow supplies verified saved_data.
# A normal script run always performs the real CUDA calculation.
if "saved_data" in globals():
    report = globals()["saved_data"]
else:
    from scripts.pendant_evaporation.study import simulate

    report = simulate()

print(
    {
        "device": report["summary"]["device"],
        "CUDA_curve_points": report["summary"]["CUDA_curve_points"],
        "CPU_CUDA_relative_error": report["summary"]["cpu_cuda_relative_error"],
        "finest_grid_error_K": report["summary"][
            "grid_max_temperature_error_K"
        ]["321"],
    }
)

# %% [markdown]
# ## The coupled model predicts a cooler drop and less evaporation
#
# As surface temperature increases, the mass-transfer prediction rises because more pentane vapor is available at the surface. The heat-supported flux falls because the temperature difference driving convection becomes smaller. Their crossing therefore selects a steady state.

# %%
plot_curves(report)
plt.show()

# %% [markdown]
# *Left: the reported-condition flux estimate lies above the transport predictions at the same temperature. Right: black points mark the 16 coupled solutions using the 1984 correction for heat and mass. Lines show the first property configuration; shading spans all configurations. Method ranges are not measurement error bars or confidence intervals.*
#
# With the 1984 correction on both processes, the predicted temperature is **258.59–260.36 K**, compared with the reported **263.6 K**. Predicted equilibrium flux is **0.01141–0.01226 kg m⁻² s⁻¹**. The reported injection ratio instead implies **0.03814–0.04000 kg m⁻² s⁻¹** under our conversion.
#
# At the reported temperature, changing the mass blowing correction to the Stefan expression increases predicted mass flux by only **4.94%**. That is much smaller than the factor **2.51–2.65** between the reported-condition flux and the 1984 mass-transfer prediction.
#
# ## Property choices shift the result without resolving the discrepancy
#
# The reconstructed mixture gives $\mathrm{Pr}=0.694$–$0.797$. We also impose the reported $\mathrm{Pr}=0.86$ in the heat correlation while retaining the independently reconstructed airflow and mass transport.

# %%
plot_sensitivity(report)
plt.show()

# %% [markdown]
# *Each row changes the mass and heat blowing corrections. Points are property-method variants and horizontal segments span their range. “No correction” applies only to the heat-transfer blowing correction. The dashed vertical line is the reported temperature; its experimental uncertainty has not been recovered.*
#
# Using the reported Prandtl number moves the 1984/1984 intersections to **258.89–261.09 K**. Even omitting the heat blowing correction gives **260.49–262.74 K**. These sensitivities move the prediction toward the reported temperature, but they do not recover both temperature and flux.
#
# The spread shown here covers selected property and correlation choices. It does not include all uncertainty in the pendant shape, gas conditions, measurements or heat exchange with the apparatus. Numerical intersection error is below **0.0001 K** on the finest retained grid, much smaller than those physical uncertainties.
#
# ## The heat budget reveals the missing constraints
#
# At the reported surface temperature, the latent heat is approximately $L=391654\ \mathrm{J\,kg^{-1}}$. We compare latent demand with the no-blowing convection reference using
#
# $$
# q_{\mathrm{extra}}''
# =j_{\mathrm{target}}L-q_0''.
# $$
#
# This residual is the additional net heat needed relative to that particular convection estimate. It does not identify whether the energy comes from the liquid feed, the support or another mechanism.

# %%
plot_heat_budget(report)
plt.show()

# %% [markdown]
# *Ranges span paired property-method calculations. The no-blowing reference is the largest convection estimate among the models tested, not a universal upper bound for the apparatus. The radiation bar assumes all surroundings are at 326 K and unit emissivity.*
#
# Convection supplies **4.74–5.14 kW m⁻²**. The 1984 mass-transfer prediction at the reported temperature needs **5.65–6.04 kW m⁻²**, leaving a paired-method residual of **0.56–1.25 kW m⁻²**. In contrast, the reported-condition flux needs **14.94–15.67 kW m⁻²**, leaving **9.85–10.88 kW m⁻²**. Imposing the reported Prandtl number still leaves **9.60–10.79 kW m⁻²**.
#
# If all surrounding surfaces were at 326 K, the blackbody radiation bound, using the Stefan–Boltzmann constant $\sigma$, would be
#
# $$
# q_{\mathrm{rad,max}}''
# =\sigma\left(T_{\mathrm{wall}}^4-T_s^4\right)
# \approx 0.367\ \mathrm{kW\,m^{-2}}.
# $$
#
# Radiation under that assumption cannot close the larger residual. Actual wall temperature and emissivity were not recovered. Likewise, evaluating a blowing-corrected convection flux at an imposed evaporation rate would require its own thermal balance; the latent-only expression for $B_H$ cannot simply be reused when other heat sources are present.
#
# ## Evidence and scope
#
# The experiment is documented in *Experimental Study of Evaporation of a Volatile Pendant Drop under High Mass Flux Conditions* ([1984 experimental paper](https://doi.org/10.1252/jcej.17.7), methods on pp. 8–9 and selected pentane condition on p. 9). The retained [experimental paper](https://doi.org/10.1252/jcej.17.7) and [companion theory paper](https://doi.org/10.1252/jcej.17.1) provide the source equations and measurement context.
#
# The [computational evidence and reproduction instructions](index.md) record the source checkout, property methods, actual RTX 5060 CUDA execution, CPU comparison, derivative checks, refinement and retained data. This is an algebraic correlation study with CPU property calculations and CUDA evaluations; it does not resolve the fluid field or simulate a time-dependent pendant drop. The registered example refresh runs the full calculation on CUDA; later figure-only replots reuse its verified saved data.
#
# ## What this means for interpreting evaporation measurements
#
# **A drop's measured temperature and evaporation rate constrain different parts of the physics, and a useful model must satisfy both.** In this pentane case, convection-only heat balance and independently reconstructed vapor transport cannot explain the reported pair. The tested blowing corrections and property methods leave that conclusion unchanged within the stated model.
#
# The next step is to recover the original unrounded injection ratio, gas velocity, property convention and paired drop dimensions. Those establish whether the flux conversion is comparable. Feed temperature and rate, support dimensions and temperatures, and wall temperature would then constrain the missing heat paths.
#
# A model with actual pendant geometry and heat conduction through the feed and support could change the thermal prediction. However, **adding heat alone cannot fix the separate mass-transfer discrepancy at the same surface temperature**. The practical lesson is to resolve the flow, geometry and energy accounting before fitting an evaporation coefficient or assigning the residual to an unmeasured heat source.
