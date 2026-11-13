"""CPU-only result types for prepared simulation observations."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SimulationDiagnostics:
    """Detached per-box diagnostics; no particle-slot arrays are downloaded.

    Number and charge concentrations are in m^-3 and elementary charges/m³.
    Particle/gas mass concentrations are kg/m³ in their independent species
    orders. Active slots count representatives, while number concentration
    includes multiplicity. Latent energy, when configured, is the last step's
    signed per-box/full-gas energy [J]. Temperature is K, pressure Pa,
    and volume m³.
    """

    completed_steps: int
    simulated_time: float
    gas_names: tuple[str, ...]
    active_slots: np.ndarray
    number_concentration: np.ndarray
    charge_concentration: np.ndarray
    particle_mass_concentration: np.ndarray
    gas_concentration: np.ndarray
    temperature: np.ndarray
    pressure: np.ndarray
    saturation_ratio: np.ndarray
    volume: np.ndarray
    latent_heat_energy: np.ndarray | None
