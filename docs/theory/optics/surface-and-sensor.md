# From a reflecting surface to a satellite band

A satellite looking toward sun glint can receive a much brighter signal than an instrument looking slightly away, even over ocean pixels receiving the same sunlight. Predicting these views requires a directional radiance. Calculating the energy leaving the ocean requires an integral over all upward directions. Both quantities follow from the surface reflection law and the atmosphere above it.

## Flux and directional reflectance

For fixed wavelength, the bidirectional reflectance distribution function $f_r(\Omega_i,\Omega_v)$ has units of sr⁻¹. Use $\mu_i$ and $\mu_v$ for the positive cosines relative to the surface normal in the incident and viewing hemispheres. With incident diffuse radiance $L_{\rm sky}$ and incident direct-normal irradiance $E_{\rm dir}$ evaluated at the ground, the reflected radiance toward $\Omega_v$ is

$$
L_{\uparrow}(\Omega_v) = f_r(\Omega_s,\Omega_v)\mu_s E_{\rm dir}
 +\int_{\rm sky} f_r(\Omega_i,\Omega_v)L_{\rm sky}(\Omega_i)\mu_i\,d\Omega_i .
$$

For a direct beam the directional-hemispherical reflectance is $\alpha(\Omega_i)=\int_{\rm upward} f_r(\Omega_i,\Omega_v)\mu_v\,d\Omega_v$. Energy conservation requires $0\leq\alpha\leq1$ for a passive surface. A Lambertian surface with albedo $a$ has $f_r=a/\pi$ because the cosine-weighted hemisphere integrates to $\pi$ steradians. For example, a dusty soil assigned $a=0.30$ under 500 W/m² downwelling in a band returns 150 W/m² upwards. Under the same Lambertian assumption its band radiance is 47.75 W m⁻² sr⁻¹ in every upward direction. An actual soil can change its apparent reflectance with both illumination and view. Land BRDF retrievals supply angular model coefficients at specified bands, dates and quality levels. [NASA MODIS BRDF products](https://ladsweb.modaps.eosdis.nasa.gov/missions-and-measurements/science-domain/brdf-albedo-and-nbar/)

Wind changes the probability that an ocean surface facet has the slope connecting the solar and viewing directions by specular reflection. A Cox–Munk model weights Fresnel reflection at that facet by its slope probability. Both the peak brightness and its angular extent vary with wind, so an angular quadrature must resolve the peak before its integral can be used as an albedo. An estimate of the reflected surface term alone describes a different physical quantity from ocean color, which also receives water-leaving light and can include whitecaps. State which terms enter a scene. [Cox and Munk sea-surface measurements](https://doi.org/10.1364/JOSA.44.000838)

## What an instrument records

Propagate surface-reflected light and light scattered within the air column to top of atmosphere. For an energy-sensitive channel with relative spectral response $S(\lambda)$, its equivalent band spectral radiance along the view is

$$
\bar L = \frac{\int S(\lambda)L_{\rm TOA}(\lambda)\,d\lambda}
                    {\int S(\lambda)\,d\lambda}.
$$

This normalization preserves the units of spectral radiance. For a reflected-solar product reporting a reflectance factor, the corresponding ratio under the same response is

$$
\rho = \frac{\pi\int S(\lambda)L_{\rm TOA}(\lambda)\,d\lambda}
 {\mu_0\int S(\lambda)E_{\odot}(\lambda)\,d\lambda},
$$

where $E_{\odot}$ is the direct-normal extraterrestrial spectrum at the overpass Earth–Sun distance. Use the mission's published solar irradiance, response convention, and calibration for a product comparison. Some product fields contain the reflectance factor multiplied by $\mu_0$. A photon-counting response and an energy-based optical response also carry different spectral weighting. Apply the supplied detector response and its documented convention once. [MODIS response and calibration tables](https://mcst.gsfc.nasa.gov/calibration/parameters)

Compare directional bands at matched sun and sensor geometry, band response and footprint. For a CERES broadband comparison, integrate outgoing radiance over the hemisphere and the applicable shortwave spectrum. CERES uses scene-dependent angular distribution models to estimate that hemispheric flux from measured radiance. A land–ocean footprint can contain both bright soil and glint. Solve each subcolumn at its own surface properties before aggregating the predicted observable with the instrument's footprint weights. [CERES angular distribution models](https://ceres.larc.nasa.gov/data/angular-distribution-models/) · [Observational comparison](../../studies/validation/spectral-radiation/observations.md) · [Worked surface comparison](../../studies/examples/coupled-populations/radiation/parcel_surfaces.py)
