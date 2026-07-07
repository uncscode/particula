"""Shared support helpers for the GPU mass-precision study tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class _MassPrecisionCase:
    """Container for a deterministic mass-precision baseline case."""

    case_name: str
    size_band: str
    radius_unit: str
    density_unit: str
    volume_fraction_unit: str
    target_radius_m: float
    density_kg_m3: np.ndarray
    volume_fractions: np.ndarray
    masses: np.ndarray
    concentration: np.ndarray
    charge: np.ndarray
    volume: np.ndarray


def _build_case(
    *,
    case_name: str,
    size_band: str,
    target_radius_m: float,
    density_kg_m3: list[float],
    volume_fractions: list[float],
    concentration_offset: float,
    charge_offset: float,
    volume_scale: float,
) -> _MassPrecisionCase:
    """Build one deterministic mass-precision case."""
    density = np.asarray(density_kg_m3, dtype=np.float64)
    fractions = np.asarray(volume_fractions, dtype=np.float64)
    total_volume = (4.0 / 3.0) * np.pi * target_radius_m**3
    species_volumes = total_volume * fractions
    per_species_mass = species_volumes * density

    box_scaling = np.array([[1.0], [1.25]], dtype=np.float64)
    particle_scaling = np.array([[1.0, 1.5, 2.0]], dtype=np.float64)
    scaling = box_scaling * particle_scaling
    masses = (
        scaling[..., np.newaxis] * per_species_mass[np.newaxis, np.newaxis, :]
    )

    concentration = (
        np.array(
            [[100.0, 125.0, 150.0], [175.0, 200.0, 225.0]],
            dtype=np.float64,
        )
        + concentration_offset
    )
    charge = (
        np.array([[0.0, 1.0, -1.0], [2.0, -2.0, 0.0]], dtype=np.float64)
        + charge_offset
    )
    volume = np.array([1.0, 1.4], dtype=np.float64) * volume_scale

    return _MassPrecisionCase(
        case_name=case_name,
        size_band=size_band,
        radius_unit="m",
        density_unit="kg/m^3",
        volume_fraction_unit="fraction",
        target_radius_m=target_radius_m,
        density_kg_m3=density,
        volume_fractions=fractions,
        masses=masses.astype(np.float64),
        concentration=concentration,
        charge=charge,
        volume=volume,
    )


def _build_mass_precision_cases() -> list[_MassPrecisionCase]:
    """Build the deterministic study cases in ascending size order."""
    return [
        _build_case(
            case_name="npf_cluster",
            size_band="new-particle-formation cluster",
            target_radius_m=1.5e-9,
            density_kg_m3=[1000.0],
            volume_fractions=[1.0],
            concentration_offset=0.0,
            charge_offset=0.0,
            volume_scale=1.0e-6,
        ),
        _build_case(
            case_name="five_to_ten_nm",
            size_band="5-10 nm particle",
            target_radius_m=7.0e-9,
            density_kg_m3=[1100.0],
            volume_fractions=[1.0],
            concentration_offset=25.0,
            charge_offset=0.0,
            volume_scale=1.5e-6,
        ),
        _build_case(
            case_name="accumulation_mode",
            size_band="accumulation mode",
            target_radius_m=1.5e-7,
            density_kg_m3=[1200.0, 1800.0],
            volume_fractions=[0.65, 0.35],
            concentration_offset=50.0,
            charge_offset=1.0,
            volume_scale=2.0e-6,
        ),
        _build_case(
            case_name="cloud_droplet",
            size_band="cloud droplet",
            target_radius_m=1.0e-5,
            density_kg_m3=[1000.0, 1770.0],
            volume_fractions=[0.92, 0.08],
            concentration_offset=75.0,
            charge_offset=-1.0,
            volume_scale=2.5e-6,
        ),
    ]


def _project_candidate(
    case: _MassPrecisionCase,
    candidate_id: str,
) -> dict[str, Any]:
    """Project one study case into a candidate representation."""
    if candidate_id == "fp32_absolute_mass":
        projected_masses = case.masses.astype(np.float32)
        return {
            "candidate_id": candidate_id,
            "projected": {"masses": projected_masses},
            "reconstructed_masses": projected_masses.astype(np.float64),
        }

    if candidate_id == "mixed_precision_mass_plus_density":
        projected_masses = case.masses.astype(np.float32)
        projected = {
            "masses": projected_masses,
            "concentration": case.concentration.astype(np.float32),
            "charge": case.charge.astype(np.float32),
            "volume": case.volume.astype(np.float32),
            "density": case.density_kg_m3.astype(np.float64),
        }
        return {
            "candidate_id": candidate_id,
            "projected": projected,
            "reconstructed_masses": projected_masses.astype(np.float64),
        }

    if candidate_id == "fp32_total_mass_fp32_mass_fraction":
        total_mass = np.asarray(
            np.sum(case.masses, axis=-1, dtype=np.float64),
            dtype=np.float64,
        )
        projected_total_mass = total_mass.astype(np.float32)
        projected_mass_fractions = np.divide(
            case.masses,
            total_mass[..., np.newaxis],
            where=total_mass[..., np.newaxis] > 0.0,
            out=np.zeros_like(case.masses),
        ).astype(np.float32)
        reconstructed_masses = projected_total_mass.astype(np.float64)[
            ..., np.newaxis
        ] * projected_mass_fractions.astype(np.float64)
        return {
            "candidate_id": candidate_id,
            "projected": {
                "total_mass": projected_total_mass,
                "mass_fractions": projected_mass_fractions,
            },
            "reconstructed_masses": reconstructed_masses,
        }

    raise ValueError(f"Unsupported candidate id: {candidate_id}")
