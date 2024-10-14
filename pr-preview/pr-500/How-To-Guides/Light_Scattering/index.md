# Index: Light Scattering *BETA*

Aerosol optics examines the interaction between light and aerosol particles suspended in the atmosphere. These interactions, encompassing absorption, scattering, and the emission of light, are pivotal in deciphering the physical properties of aerosols and their environmental ramifications.

## Notebooks
- [Mie Scattering Basics](notebooks/mie_basics.ipynb)
- [Humidified Particle Scattering](notebooks/humid_scattering.ipynb)
- [Kappa-HGF Estimation from Light Extinction](notebooks/kappa_vs_extinction.ipynb)
- [Correcting for Scattering Truncation](notebooks/scattering_truncation.ipynb)


### Mie Scattering Theory

Central to aerosol optics is Mie scattering theory, formulated by Gustav Mie in 1908. This foundational theory enables the precise calculation of light scattering and absorption by spherical particles, taking into account their size, material composition, and the incident light's wavelength. In this context, we leverage the capabilities of pyMieScatt, a comprehensive Python library designed to facilitate Mie scattering computations.

#### Fundamental Concepts

- **Scattering Efficiency**: Quantifies the efficacy of particles in deflecting light in various directions.
- **Absorption Efficiency**: Assesses the extent to which particles absorb incident light.
- **Single Scattering Albedo (SSA)**: This ratio of scattering to total light extinction (scattering plus absorption) provides insight into whether particles are more likely to scatter light rather than absorb it.

### Understanding Particle Distributions

Aerosol particles exhibit a vast diversity in terms of size, shape, and chemical composition, making the study of their distributions crucial for accurate optical modeling.

#### Types of Distributions

- **Monodisperse**: A scenario where all particles are of identical size.
- **Polydisperse**: Represents a realistic distribution where particles vary in size, often characterized by statistical distribution models, such as the log-normal distribution.

### Addressing Truncation Errors in Measurements

Measurements of aerosol optical properties can be compromised by truncation errors, stemming from the inability of instruments to capture the complete angular range of scattered light.

#### Consequences and Mitigation Strategies

- **Scattering Coefficient Underestimation**: The restricted detection of scattered light may lead to inaccuracies in determining aerosol optical depth (AOD) and other key optical properties.
- **Correction Techniques**: A variety of correction methods, including analytical adjustments and empirical calibration, are employed to counteract truncation errors and refine the accuracy of aerosol optical measurements.

### Overview

This series offers a detailed exploration of aerosol optical phenomena through the lens of Mie scattering theory, analysis of particle size distributions, and methodologies for correcting truncation errors in aerosol instrumentation. By enhancing our understanding of these areas, we aim to further our knowledge of aerosol behavior and its environmental impact.

