"""Replay parcel forcing and checkpoint extensive species inventories."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from particula.dynamics.parcel_trace import OrganicForcing, ParcelTrace

__all__ = [
    "OrganicForcing",
    "ParcelTrace",
    "ReplayCheckpoint",
    "ReplayResult",
    "replay",
]


@dataclass(frozen=True)
class ReplayCheckpoint:
    """Portable, immutable continuation at a completed forcing record."""

    trace_id: str
    parcel_id: str
    species: tuple
    index: int
    time: float
    max_step: float
    humidity_amplitude: float
    inventory: tuple

    def save(self, path):
        """Write the completed checkpoint to JSON."""
        Path(path).write_text(json.dumps(asdict(self), allow_nan=False) + "\n")

    @classmethod
    def load(cls, path):
        """Read a checkpoint for validation at continuation."""
        fields = json.loads(Path(path).read_text())
        return cls(
            **{
                **fields,
                "species": tuple(fields["species"]),
                "inventory": tuple(fields["inventory"]),
            }
        )


@dataclass
class ReplayResult:
    """Record-time outputs. Water and organic budgets are extensive kg."""

    time: np.ndarray
    volume: np.ndarray
    wet_radius: np.ndarray
    liquid: np.ndarray
    supersaturation: np.ndarray
    water_exchange: np.ndarray
    water_restored: np.ndarray
    organic_gas: np.ndarray
    organic_particle: np.ndarray
    organic_added: np.ndarray
    number: np.ndarray
    bin_water: np.ndarray
    bin_organic: np.ndarray
    gas_species: tuple[str, ...]
    gas_masses: np.ndarray
    dry_mass: np.ndarray | None
    humidity_gradient: float | None
    checkpoint: ReplayCheckpoint | None


def replay(
    trace,
    *,
    max_step,
    device="cpu",
    humidity_amplitude=0.0,
    gradient=False,
    species=None,
    stop_index=None,
    checkpoint=None,
):
    """Integrate prescribed humid air and a first-order organic uptake tracer.

    The humidity perturbation multiplies vapor by 1 + amplitude × normalized
    elapsed time. A reverse pass gives d(terminal liquid kg)/d(amplitude).
    Sources use interval rates and one impulse at the right endpoint.
    """
    from particula.dynamics._parcel_experiment import integrate

    if gradient and checkpoint is not None and checkpoint.index:
        raise ValueError(
            "a full-trace gradient requires the original initial state"
        )
    names = ("water", trace.organic.name) if species is None else tuple(species)
    if len(names) != 2 or set(names) != {"water", trace.organic.name}:
        raise ValueError("species must name each declared owner exactly once")
    return integrate(
        {trace.parcel_id: trace},
        max_step=max_step,
        device=device,
        amplitude=humidity_amplitude,
        humidity_target=trace.parcel_id if gradient else None,
        checkpoint=checkpoint,
        stop_index=stop_index,
        species=names,
    ).parcels[trace.parcel_id]
