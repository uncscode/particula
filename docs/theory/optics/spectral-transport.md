# Where does sunlight go in an aerosol column?

A particle that scatters sunlight removes energy from the direct solar beam. That energy can still reach the ground as diffuse light or escape through the top of the atmosphere. To calculate a change in surface or atmospheric energy uptake, follow both the beam and the scattered light to the boundaries of a column. Each calculation below is at one vacuum wavelength unless an integral over wavelength is shown.

## From one parcel to a vertical profile

Divide the column into horizontally uniform layers, ordered from top to bottom. In a layer of vertical thickness $h$ in metres, the extinction coefficient $b_e$ and scattering coefficient $b_s$ have units of m⁻¹. A population of independent particles gives $b_e=\sum_i n_i C_{e,i}$, with number density $n_i$ and particle extinction cross section $C_{e,i}$. Add contributions from all particles and gas molecules to obtain

$$
\Delta\tau = b_e h, \qquad \omega=b_s/b_e .
$$

Optical depth $\tau$ increases downwards. A single parcel supplies the population for one slab once its thickness is specified. Stacking layers requires a vertical profile at the time of interest. In particular, the volume used to convert a particle count to number density leaves the vertical thickness undetermined. [Particula population optics](../../guide/optics/index.md)

Let $E_0$ be direct-normal spectral irradiance at the upper face and $\mu_0>0$ the cosine of solar zenith. Integrating extinction along the oblique ray gives the horizontal unscattered beam $F_{\mathrm{dir}}(\tau)=\mu_0 E_0\exp(-\tau/\mu_0)$. For example, let a purely absorbing parcel have $\Delta\tau=0.2$ and $\mu_0=0.5$. An incident 1 W m⁻² nm⁻¹ normal to the beam delivers 0.5 W m⁻² nm⁻¹ to the horizontal top and 0.3352 W m⁻² nm⁻¹ to the bottom. Over a black surface with no diffuse illumination, the parcel absorbs the difference of 0.1648 W m⁻² nm⁻¹.

## Transport of scattered light

Consider the diffuse radiance $I(\tau,\Omega)$ travelling in direction $\Omega$. Choose the direction cosine $\mu$ relative to the downward vertical. For scalar, elastic scattering in a plane-parallel layer without internal emission, the transfer equation is

$$
\mu\frac{\partial I}{\partial \tau}=-I
 +\frac{\omega}{4\pi}\int_{4\pi}P(\Omega\cdot\Omega')I(\tau,\Omega')\,d\Omega'
 +\frac{\omega E_0}{4\pi}P(\Omega\cdot\Omega_0)e^{-\tau/\mu_0}.
$$

The last term gives the first scattering of the direct beam. The integral allows light already travelling in a diffuse direction to scatter again. The dimensionless phase function has spherical mean one and expands as $P(x)=\sum_\ell(2\ell+1)\beta_\ell P_\ell(x)$, with $\beta_0=1$ and $\beta_1=g$. A mixture weights each constituent's $\beta_\ell$ by its scattering optical depth. For example, a purely absorbing constituent increases extinction while leaving the angular distribution of the scattering events unchanged. Scalar transport omits polarization. Comparisons at specific viewing angles can be sensitive to higher phase moments even where integrated flux is well resolved. [DISORT derivation](https://doi.org/10.1364/AO.27.002502)

A discrete-ordinate solver replaces the angular integral by a weighted sum. Decomposing the azimuth dependence into Fourier modes gives a linear differential system within each homogeneous layer. Its solution relates the incoming streams at the layer faces to the outgoing streams. Let $d$ and $u$ denote diffuse stream vectors at the shared face of two layers. The upper layer returns $d=d_0+R_{\rm upper,bottom}u$. The lower layer returns $u=u_0+R_{\rm lower,top}d$. Here $d_0$ and $u_0$ contain contributions from imposed light and direct-beam scattering. Eliminating $u$ gives

$$
 (\mathbf{1}-R_{\rm upper,bottom}R_{\rm lower,top}) d
   =d_0+R_{\rm upper,bottom}u_0 .
$$

This coupling accounts for repeated reflections across the face. Retaining both directions at internal interfaces lets the calculation locate absorption within the column. [Adding and doubling](https://www.giss.nasa.gov/pubs/abs/ha09500o.html) · [libRadtran transport solvers](https://gmd.copernicus.org/articles/9/1647/2016/)

## Energy absorbed in each layer

Integrating radiance over a hemisphere with weight $|\mu|$ gives its diffuse flux. Write net downward flux as $F_{\mathrm{net}}=F_{\mathrm{dir}}+F_{\mathrm{diff,down}}-F_{\mathrm{up}}$. With no emission, layer absorption equals $F_{\mathrm{net,top}}-F_{\mathrm{net,bottom}}$. At an opaque surface the net downward flux is the surface uptake. Comparing two aerosol states at fixed geometry, illumination, atmosphere and surface gives $\Delta F=F_{\mathrm{case}}-F_{\mathrm{reference}}$. A change in reflected flux has the opposite sign to a change in top net flux when incident light is held fixed.

For a wavelength interval $B$, $F_B=\int_B F_\lambda\,d\lambda$. The spectral density in W m⁻² nm⁻¹ is $10^{-9}$ times its numerical value per metre. A constant 2 W m⁻² nm⁻¹ across 450–550 nm therefore integrates to 200 W/m². A representative-wavelength gas calculation instead approximates a band integral with $\sum_j w_j F_{\lambda_j}$, where the positive weights $w_j$ have units of wavelength. Apply these weights to the solved flux. The reduced Earth reference uses positive quadrature in 22 intervals. The interpretable spectral products are integrals or mean flux densities for these intervals. A single quadrature node can represent several other wavelengths with similar gas transmission. A requested wavelength on a custom grid uses the containing full REPTRAN band’s effective absorption at unit air mass. Individual narrow lines require resolved absorption data. Averaging optical depths before transmission would replace the mean of an exponential with the exponential of a mean and lose the contribution from transparent windows in a gas absorption band. [REPTRAN parameterization](https://doi.org/10.1016/j.jqsrt.2014.06.024)

For a cloud or aerosol slab illuminated by a surface instrument record, the top flux describes the light leaving that slab. For comparison to a satellite, include the overlying atmosphere, which both scatters and absorbs outgoing radiation. Near the horizon Earth's curvature changes the direct ray path. Use the solar-angle comparison in the numerical benchmark to judge the plane-parallel approximation for the intended quantity. [Parcel example](../../studies/examples/coupled-populations/radiation/parcel_surfaces.py) · [Column example](../../studies/examples/coupled-populations/radiation/layer_altitude.py) · [Numerical benchmarks](../../studies/validation/spectral-radiation/benchmark.md)
