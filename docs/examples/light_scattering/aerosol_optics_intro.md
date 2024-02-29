# Aerosol Optics

Aerosol optics involves the study of how light interacts with small particles suspended in the air. These interactions include absorption, scattering, and emission of light, which are critical for understanding aerosol properties and their effects on the environment.

## Mie Scattering

Mie scattering theory is a powerful tool for understanding light scattering by spherical particles. Developed by Gustav Mie in 1908, this theory provides a way to calculate the scattering and absorption of light by particles based on their size, composition, and the wavelength of light. We are building on pyMieScatt [link](https://pymiescatt.readthedocs.io/en/latest/), a Python package that implements the Mie scattering calculations.

### Key Concepts

- **Scattering Efficiency**: A measure of how effectively a particle scatters light.
- **Absorption Efficiency**: Indicates how much light is absorbed by the particle.
- **Single Scattering Albedo (SSA)**: The ratio of scattering efficiency to the sum of scattering and absorption efficiencies, representing the particle's propensity to scatter light rather than absorb it.

## Particle Distributions

Aerosol particles are not uniform; they vary in size, shape, and composition. Understanding these particle distributions is essential for accurate modeling of aerosol optics.

### Distribution Types

- **Monodisperse**: All particles have the same size.
- **Polydisperse**: Particles have a range of sizes, typically described by a distribution function (e.g., log-normal distribution).

## Truncation Errors in Instrument Measurements

Instrumental measurements of aerosols are often subject to truncation errors due to limitations in detecting the full angular range of scattered light.

### Impact of Truncation Errors

- **Underestimation of Scattering Coefficients**: Limited angular range can lead to underestimation of particle scattering, affecting aerosol optical depth (AOD) calculations.
- **Correction Methods**: Various approaches, including mathematical corrections and calibration techniques, are used to mitigate truncation errors and improve measurement accuracy.

## Here in

These example code study of aerosol optics through Mie scattering, particle distributions, and the understanding of truncation errors in measurements provides essential insights into aerosol behavior. 