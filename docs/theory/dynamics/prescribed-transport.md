# Transport in prescribed wind fields

How much of the width of a tracer puff comes from atmospheric mixing, and how much from its numerical representation? A prescribed wind supplies a mean velocity and a thermodynamic history. A supplied eddy diffusivity describes the unresolved spread about that mean. Molecular diffusion describes thermal motion at the molecular scale. Both coefficients have units of m²/s, but the eddy diffusivity depends on the flow, observation scale and averaging time. A point release in uniform wind provides a control for measuring spread before interpreting a terrain or CFD calculation. [Compare the same puff on grids and as particles](../../studies/examples/gas-transport/turbulent_puff.py).

## Gaussian puff in uniform air

Let $x_i$ measure distance along one Cartesian axis in meters. Let $u_i$ be the air velocity relative to the fixed grid, $v_i$ a prescribed particle slip, both in m/s, and $K_i$ a constant directional eddy diffusivity. Add downward terminal slip to the vertical velocity. In unbounded space the one-dimensional transport equation for concentration $c$ is

$$
\partial_t c + (u_i+v_i)\partial_{x_i} c = K_i\partial_{x_i}^2 c .
$$

Integrating over space with weights $1$, $x_i$ and $x_i^2$ and taking the concentration and flux to vanish at infinity gives constant total inventory, mean position $\mu_i(t)=\mu_i(0)+(u_i+v_i)t$ and variance $\sigma_i^2(t)=\sigma_i^2(0)+2K_i t$. A point release has zero initial variance. For independent dimensions the Gaussian solutions multiply. A resolved particle takes a displacement over $h$ seconds given by

$$
\Delta x_i = (u_i+v_i)h + \sqrt{2K_i h}\,Z_i, \qquad Z_i \sim \mathcal{N}(0,1).
$$

This samples the constant-coefficient free-space transition exactly. The local velocity is frozen over a substep. The particle keeps its represented number, charge and material on every cell crossing. A singleton axis represents a slice of specified physical thickness and has no stochastic displacement. For equal-weight records the mean has sampling standard error $\sigma_i/\sqrt{N}$, and the unbiased variance estimate has standard error $\sigma_i^2\sqrt{2/(N-1)}$. Unequal record payloads require the weights appropriate to the quantity being compared.

## Numerical spread on a fixed grid

For one dimension, constant positive air speed $u$, cell width $\Delta x$ and explicit substep $h$, first-order upwind advection transfers fraction $a=uh/\Delta x$ to the downwind cell. Add a symmetric fraction $d=Kh/\Delta x^2$ to each neighbor for diffusion. The three possible cell-center displacements $(-\Delta x,0,+\Delta x)$ have weights $(d,1-a-2d,a+d)$. Their mean is $uh$. Subtract the square of that mean from the second raw moment to obtain the variance growth per second,

$$
\frac{\Delta\sigma_{\rm grid}^2}{h}
=2K+|u|\,\Delta x\left(1-\frac{|u|h}{\Delta x}\right).
$$

This expression assumes negligible interaction with boundaries and constant coefficients. The second term persists for pure advection at Courant number below one. Particula bounds the summed outgoing fraction in all directions by 0.8 per substep. Grid refinement at fixed physical time reduces the extra width. Increasing the number of resolved particles reduces sampling uncertainty in the separate particle calculation.

## Bed contact and reflection

On a flat absorbing plane in still air, the diffusion equation has an image solution. Subtract the reflected free-space Gaussian centered at $-z_0$ from the one centered at $z_0>0$. Integrating the surviving density above the bed gives the probability of first contact by time $t$,

$$
P(T_{\rm bed}\le t)=\operatorname{erfc}\!\left(\frac{z_0}{\sqrt{4K_z t}}\right).
$$

Cartesian trajectories cross cell faces along the chord joining the substep endpoints. A closed exterior reflects a diffusive step. A periodic exterior wraps the particle. A depositing or open exterior records the intersection time and retains the payload as deposited or escaped. A random path can cross the bed and return above it during a single substep. Chord monitoring therefore undercounts contacts at finite timestep. The worked example decreases the maximum substep while holding $K_z$, duration, seed and release height fixed. The recorded first-contact times are substep approximations.

## Well-mixed concentration over terrain

For bed elevation $b(x)$ and fixed lid $H$, the reference height $\eta$ maps to physical height $z=b(x)+J(x)\eta$, where $J=(H-b)/H$. The local physical volume is $J\,dx\,dy\,d\eta$. A uniform suspension therefore contains fewer particles in a cell over the crest than in an equal reference cell over the flank. A displacement made solely from the local value of a variable diffusivity can produce a spurious concentration gradient. [Visser (1997)](https://doi.org/10.3354/meps158275).

The grid uses directional diffusion along its coordinate lines and projected face areas. For this closure, write $D_x=K_x$, $D_y=K_y$ and $D_\eta=K_z/J^2$. Ignoring advection, the reference-coordinate concentration obeys

$$
\partial_t c = \frac{1}{J}\sum_i \partial_i\!\left(JD_i\partial_i c\right).
$$

A continuous Itô walk with variance rate $2D_i$ would require drift $J^{-1}\partial_i(JD_i)$ along each reference axis to produce this equation for concentration per physical volume. Particula uses the discrete grid face exchanges directly on terrain, with variable diffusivity, or when recorded face conductances disagree with constant cell-center coefficients. From cell $i$ a record selects at most one shared face per substep, with probability $P_{ij}=M_{ij}/V_i$. Here $M_{ij}$ is the symmetric exchange in m³ for the substep and $V_i$ the physical cell volume. Since $V_i P_{ij}=M_{ij}=V_j P_{ji}$, constant concentration is stationary under mixing. An x jump carries the same cumulative physical-volume fraction inside the destination cell. This matters where $J$ varies across the cell. Directional diffusion along the sloped mesh is a modeling choice. An isotropic world-coordinate tensor generally introduces cross terms.

Mapped open and depositing exteriors use an absorbing face one half-cell from the center, with the boundary-cell diffusivity and projected area. The event is reported at the end of the selected jump substep. Closed exteriors have zero mixing exchange. Deterministic advection and settling traverse faces before the stochastic jump. Grid and timestep refinement govern the spatial position, first-contact timing and splitting errors of this model. The dune example checks the well-mixed physical concentration. A separate refinement test compares a decaying vertical cosine mode with its analytical solution.

## Conservative forcing and independent CFD replay

A fixed cell carries an extensive inventory $A_i$ in kg or represented particles, with concentration $c_i=A_i/V_i$. A solver face record supplies signed swept volume $Q_{ij}=\int\phi_{ij}\,dt$ in m³, positive outward from the owner cell. The shared face has $Q_{ji}=-Q_{ij}$ and symmetric diffusive exchange $M_{ij}$. For internal neighbors, define $Q^+=\max(Q,0)$ and $Q^-=\max(-Q,0)$. The local inventory change from their shared faces is

$$
\Delta A_i =
\sum_j\big[-Q_{ij}^+ c_i + Q_{ij}^- c_j + M_{ij}(c_j-c_i)\big].
$$

Every internal term has an equal and opposite contribution in the neighbor. Boundary inflow uses the specified inlet concentration and joins the import ledger. Outflow joins the export ledger. A host-supplied dry-air mixing ratio in kg/kg dry air must first be converted using the dry-air mass of its cell. For the incompressible cavity study the passive scalar is a concentration in kg/m³ and the conversion is multiplication by physical cell volume.

Supply face orientation, units, accepted times and stable cell identities with independent CFD data. A cell-center velocity does not identify the native face flux. A wind reversal requires separate accepted segments to preserve the transfers from each donor. With Particula-owned transport, update the inventory from these fluxes once. With host-owned transport, install the already transported inventories and the host's boundary totals, then evaluate the requested local processes. The cavity study compares host and replay field errors against timestep and mesh refinement of the host solution. [Independent OpenFOAM cavity replay](../../studies/validation/prescribed-flow/cavity_replay.py).

## Exact restart at an accepted boundary

A stochastic trajectory is addressed by the user's seed, stable record ID, accepted interval, substep and random draw within that step. A checkpoint stores the algorithm version, seed, maximum substep and next interval index, along with the particle positions, cells, fates, event times and inventory. The forcing and configuration signatures bind the grid and the accepted history. On the same device, restoring the same forcing and controls reproduces the subsequent trajectory exactly. Changing the accepted intervals or maximum substep changes the random path. Use ensemble statistics for comparisons across discretizations. [Set up a prescribed flow or replay](../../guide/processes/parcels.md).
