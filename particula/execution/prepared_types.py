"""CPU-only result types for prepared simulation observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from particula.execution.checkpoint import (
        CheckpointPayload,
        ResidentCheckpoint,
    )
    from particula.execution.prepared_processes import _ProcessPlan
    from particula.execution.process_extensions import GPUProcessPlan
    from particula.extension_fields import FieldSpec


@dataclass(frozen=True)
class SimulationCheckpoint:
    """In-memory, exact-device continuation of a prepared simulation.

    Canonical resident bytes preserve primary state, acquired sidecars and
    random words. The frozen process plans and timestep identify the compatible
    runnable sequence. Schema v2 adds absolute initial time, field descriptors
    and bytes, declared adapter-state bytes, and the device clock. Python
    implementations are supplied again through the registry at restart.
    The explicit v1 reader accepts built-in-only state with a zero time origin.
    Auxiliary bytes preserve last-step latent energy.
    Resident CPU inspection arrays are detached but are not restart authority.
    This is not a portable disk format or a device-migration interface.
    """

    schema_version: int
    resident: ResidentCheckpoint
    processes: tuple[_ProcessPlan | GPUProcessPlan, ...]
    time_step: float
    auxiliary: tuple[CheckpointPayload, ...] = ()
    initial_time: float = 0.0
    field_specs: tuple[FieldSpec, ...] = ()
    field_payloads: tuple[CheckpointPayload, ...] = ()
    extension_state: tuple[CheckpointPayload, ...] = ()
    clock: CheckpointPayload | None = None
    execution_mode: str = "resident"
    host_rng: bytes | None = None


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
