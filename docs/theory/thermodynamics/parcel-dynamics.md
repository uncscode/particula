# Water, aerosol, and motion in a parcel

A rising parcel can contain the same number of particles at cloud base as at the surface, even though its particle number concentration has fallen. Its physical volume has increased. Keeping inventories in kilograms and represented particle counts makes it possible to follow the same aerosol through expansion, condensation, and exchange with another parcel.

## Volume and water inventory

Consider a warm, closed parcel with dry-air mass $m_d$ (kg), pressure $p$ (Pa), and temperature $T$ (K). Write $q_v=m_v/m_d$ and $q_\ell=m_\ell/m_d$ for vapor and liquid mixing ratios in kg water per kg dry air. The gas occupies the parcel volume $V$ (m³). Neglecting condensed volume and dry aerosol loading gives

$$
V=\frac{m_d(R_d+q_vR_v)T}{p},\qquad
\rho_d=\frac{m_d}{V},\qquad
e=\rho_d q_v R_v T .
$$

Here $R_d$ and $R_v$ are the specific gas constants for dry air and water vapor, and $e$ is vapor partial pressure. The water vapor concentration is $\rho_d q_v$ (kg/m³). A size class representing $N_j$ particles has number concentration $N_j/V$ (m⁻³). Expansion alone conserves both vapor mass and particle count. The native discrete aerosol snapshot uses a 1 m³ normalization for concentrations. Multiply those concentrations by the **physical** parcel volume to recover its inventories. [Native particle measures](../../guide/data-model/index.md)

## Cooling and droplet growth

Prescribe vertical air velocity $w$ (m/s, positive upward) and let $z$ be height (m). In a closed ascent the total water mixing ratio $q_t=q_v+q_\ell$ is constant. The parcel approximation uses hydrostatic balance including its water loading, a fixed dry-air heat capacity $c_p$, and constant latent heat of vaporization $L_v$. With specific volume $\alpha=V/m_d$, its coupled tendencies are

$$
\frac{dz}{dt}=w,\qquad
\alpha\frac{dp}{dt}=-(1+q_t)gw,\qquad
c_p\frac{dT}{dt}=\alpha\frac{dp}{dt}+L_v\frac{dq_\ell}{dt}.
$$

Pressure falls during ascent. Condensation warms the parcel and removes vapor. These two effects both change supersaturation $S=e/e_s(T)-1$, where $e_s(T)$ is saturation vapor pressure over liquid water. Setting water and phase exchange to zero recovers dry adiabatic cooling.

For a class with dry radius $a$ and wet radius $r$ (m), the continuum growth law advances squared radius. With constant hygroscopicity $\kappa$, water surface tension $\sigma$, and liquid density $\rho_w$, it takes the form

$$
\frac{d r^2}{dt}=\frac{2}{\mathcal R(T,p)}
\left[\frac{e}{e_s(T)}-
\frac{r^3-a^3}{r^3-a^3+\kappa a^3}
\exp\!\left(\frac{2\sigma}{\rho_w R_v T r}\right)\right].
$$

The term subtracted from ambient relative humidity is the equilibrium relative humidity at the curved droplet surface. The positive resistance $\mathcal R$ (s/m²) combines vapor diffusion through air and heat removal from the droplet surface. Summing $\frac{4}{3}\pi\rho_w N_j(r_j^3-a_j^3)$ over classes gives liquid mass, including the initial equilibrium water. This same mass sets the vapor debit and latent heating in the closed parcel. The fixed-step study checks the cloud-base trajectory and its updraft sensitivity. The adaptive study compares RK23 and ROS3 against an independent Radau integration at common observation times. [κ-Köhler formulation](https://acp.copernicus.org/articles/7/1961/2007/) · [Closed ascent](../../studies/examples/thermodynamics/moist_parcel.py) · [Adaptive parcel integration](../../studies/examples/thermodynamics/adaptive_moist_parcel.py)

## Replaying prescribed water vapor

An offline trace supplies $T,p,V,q_v$ at sampled times. Replay interpolates these state fields linearly and evaluates the same droplet growth law with midpoint forcing on each integration substep. Supersaturation at a forcing record follows from that record. Wet radii and liquid water accumulate the effect of the intervening history. Comparing radii at a common solver step while changing the trace cadence measures the effect of interpolating a sparser history.

The prescribed vapor mass $m_v=m_d q_v$ can disagree with the vapor remaining after local condensation. On an interval let $\Delta m_{\rm cond}$ be signed water mass transferred locally into droplets and $\Delta m_{\rm face,v}$ net vapor mass received through connections. The external water reservoir must supply

$$
\Delta m_{\rm restore}
=\Delta m_v+\Delta m_{\rm cond}-\Delta m_{\rm face,v}.
$$

Positive restoration supplies water to the parcel. Negative restoration removes it. In an isolated replay $\Delta m_{\rm face,v}=0$ and $\Delta m_{\rm cond}=\Delta m_\ell$. With particle exchange, incoming liquid already resides on droplets and enters through a separate face budget.

A second, dilute organic vapor can have its own evolving gas inventory. In the worked experiment a prescribed first-order uptake coefficient $k$ (s⁻¹) gives condensed mass $M_{\rm available}(1-e^{-k\Delta t})$ on a step with particles present. A rate source contributes to available gas over that interval. A prescribed pulse enters the gas at the right endpoint and is available for uptake on the next step. Gas plus particulate organic mass accounts for both types of source. The replay study checks that budget before and after restarting at the pulse. [Offline replay and mixed-species budgets](../../studies/examples/thermodynamics/parcel-experiments/parcel_replay.py)

## Turning winds from successive soundings

Each sounding supplies eastward and northward wind, temperature, absolute
pressure and water-vapor mixing ratio per kg dry air. For field $f$, let
$I_k(f,z)$ be linear interpolation between the measured heights of profile
$k$. At a time between records $t_k$ and $t_{k+1}$, prescribe

$$
f(z,t)=(1-w)I_k(f,z)+wI_{k+1}(f,z),\qquad
w=\frac{t-t_k}{t_{k+1}-t_k}.
$$

All profiles must share a height datum and overlap across the modeled
layer. Interpolating the east and north components permits a continuous
turn through geographic north. A parcel samples thermodynamics at its
moving physical height $z(t)$. A stationary cell samples its center. An
air parcel over the prescribed dune follows a streamline with constant
upstream height $\eta$, so a change of physical height can change its
temperature and humidity even within a single sounding. The replay water
reservoir supplies or removes the mass needed to enforce this humidity.

For uniform horizontal wind the displacement over one linear segment is
$\Delta\mathbf x=(\mathbf u_a+\mathbf u_b)\Delta t/2$. With the mapped
dune wind, $J(x)\,dx/dt=U(\eta,t)$, where $J=1-h(x)/H$, $h$ is bed
height and $H$ is lid height. Integrating $J$ over horizontal displacement
and the upstream velocity over time gives the parcel endpoint. At a fixed
grid face the normal volume flux varies linearly between sounding times.
The grid records a new interval at each knot and each reversal. A positive
transfer followed by an equal negative transfer can mix two different
donor concentrations even though the net air volume is zero. [Turning-wind experiment](../../studies/examples/gas-transport/turning_sounding.py)

## Exchange between parcels

Let $Q_{ij}$ (m³/s) be prescribed flow on a directed connection from parcel $i$ to parcel $j$. During a substep, an enabled inventory $X$ moves by the donor fraction

$$
F_{ij}=\frac{Q_{ij}\Delta t}{V_i}X_i,\qquad
X_i^{\rm after}=X_i-\sum_j F_{ij}+\sum_j F_{ji}.
$$

All edges use the pre-exchange donor state. The summed outgoing volume for each enabled inventory must fit its donor volume. Across each internal edge, receiver credit equals donor debit, so internal flows cancel when summed over the cluster. In unequal-volume cells the corresponding concentration changes are $-F_{ij}/V_i$ and $+F_{ij}/V_j$ for that edge, at fixed volumes. The physical volumes themselves come from the parcel traces. A prescribed connection transports the selected tracers within this imposed geometry.

For binned particles, apply the same fraction to represented count and constituent inventories in each compatible size class. Recover the mean water per particle from the transferred mass and count. This conserves mass during mixing while retaining the model's single representative wet radius per occupied class. The two-cell study follows organic gas into a receiver where some of it condenses and checks each cell against the transfer recorded at their face. [Conservative transfer and receiver uptake](../../studies/examples/thermodynamics/parcel-experiments/parcel_cluster.py)

## Settling relative to a moving face

A resolved particle carries its identity, charge, represented count and component masses across a face. For spherical particles at terminal speed in the creeping-flow regime the mover uses slip-corrected Stokes settling. With particle density $\rho_p$, dynamic viscosity $\mu$, radius $r$ and slip factor $C_c$, its downward speed magnitude is $v_s=2\rho_p g r^2 C_c/(9\mu)$, neglecting air buoyancy. The mover checks $2r v_s\rho_{\rm air}/\mu\leq0.1$.

Over a segment with constant air velocity $\mathbf u$ and rigid mesh velocity $\mathbf u_f$, the trajectory in mesh coordinates obeys

$$
\frac{d\mathbf x_{\rm relative}}{dt} =
\mathbf u-\mathbf u_f-v_s\hat{\mathbf z}.
$$

A particle initially a vertical distance $h$ above a horizontal face reaches it after $h/(v_s+u_{f,z}-u_z)$ when the denominator is positive and no other face is reached first. In an updraft, a particle can rise in world coordinates while falling relative to a rising cell. The neighboring cell receives its complete record on an internal crossing. A periodic face wraps its position, a retaining face prevents outward displacement, and open exits and deposition have distinct event records. Summing live inventories together with their terminal exits accounts for the release. The lofting study follows 1,024 particle IDs in a stack, a periodic row and a moving block, and checks world and mesh displacements, periodic wraps, and open-exit times. The two-cell experiment checks the time of an individual internal crossing. [Two-cell crossing control](../../studies/examples/thermodynamics/parcel-experiments/parcel_cluster.py) · [Multi-parcel lofting](../../studies/examples/thermodynamics/parcel-experiments/cluster_lofting.py)

## Sensitivities at a fixed time

With initial state held fixed, a derivative asks how a scalar observation $J(t_f)$ changes for a small parameter perturbation at the same elapsed time $t_f$. The closed ascent varies the updraft. Replay varies a dimensionless amplitude $b$ that multiplies prescribed vapor by $1+b(t-t_0)/(t_{\rm end}-t_0)$, where $t_{\rm end}$ is the last forcing record even when the observation is earlier. The cluster experiment varies one connection flow $Q$ and observes receiver particulate organic mass after both transport and uptake. These derivatives have different units. For example, $\partial M_{\rm receiver}/\partial Q$ has units kg s/m³.

Warp reverse mode differentiates the smooth numerical steps along the recorded schedule. Compare the resulting slope with $[J(\theta+\delta)-J(\theta-\delta)]/(2\delta)$ over several perturbation sizes, then refine the integration timestep to estimate discretization error. This interpretation assumes the same physical branches and connection topology throughout the perturbation. Discrete face crossings, boundary decisions, and topology changes require separate event-time and inventory checks. [Humidity sensitivity](../../studies/examples/thermodynamics/parcel-experiments/parcel_replay.py) · [Flow sensitivity](../../studies/examples/thermodynamics/parcel-experiments/parcel_cluster.py)
