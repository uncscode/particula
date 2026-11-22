"""Offline warm-parcel forcing with SI units and extensive inventories."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from particula.dynamics.parcel import ParcelInitialState

COLUMNS = (
    "parcel_id",
    "time_s",
    "height_m",
    "temperature_K",
    "pressure_Pa",
    "volume_m3",
    "water_kg_per_kg_dry_air",
)
CONVENTIONS = {
    "schema": 1,
    "columns": list(COLUMNS),
    "humidity": "prescribed water vapor mass per dry-air mass",
    "sampling": "linear state interpolation at recorded times",
    "coordinates": "height in metres, positive upward",
    "interval_forcing": (
        "rates in kg/s throughout interval; totals in kg at its end"
    ),
}


def _vector(value, name):
    array = np.array(value, dtype=np.float64, copy=True)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite vector")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class OrganicForcing:
    """Dilute tracer uptake with a prescribed first-order rate (1/s).

    Gas and particulate inventories are in kg. Rates (kg/s) and totals (kg)
    are keyed to successive trace intervals. Organic uptake is a passive
    tracer experiment and does not change the water-equilibrium curve.
    """

    name: str = "organic"
    initial_gas: float = 0.0
    uptake_rate: float = 0.0
    rates: tuple[float, ...] = ()
    totals: tuple[float, ...] = ()

    def __post_init__(self):
        """Validate the physical inputs and retain owned values."""
        if (
            not isinstance(self.name, str)
            or not self.name
            or self.name == "water"
        ):
            raise ValueError("provide a distinct organic species name")
        for val in (self.initial_gas, self.uptake_rate):
            if not np.isfinite(val) or val < 0:
                raise ValueError(
                    "require finite nonnegative organic gas and uptake"
                )
        for field in ("rates", "totals"):
            object.__setattr__(
                self, field, tuple(_vector(getattr(self, field), field))
            )


@dataclass(frozen=True)
class ParcelTrace:
    """One forcing history with optional wet-radius and liquid references."""

    parcel_id: str
    provenance: str
    time: np.ndarray
    height: np.ndarray
    temperature: np.ndarray
    pressure: np.ndarray
    volume: np.ndarray
    vapor: np.ndarray
    initial: ParcelInitialState
    initial_wet: np.ndarray
    organic: OrganicForcing = OrganicForcing()
    reference_radius: np.ndarray | None = None
    reference_liquid: np.ndarray | None = None
    dry_density: float | None = None  # kg/m³ for transporting dry material

    def __post_init__(self):
        """Validate the physical inputs and retain owned values."""
        if not all(
            isinstance(s, str) and s.strip()
            for s in (self.parcel_id, self.provenance)
        ):
            raise ValueError("parcel ID and trace provenance are required")
        if self.dry_density is not None and (
            not np.isfinite(self.dry_density) or self.dry_density <= 0
        ):
            raise ValueError("dry material density must be positive in kg/m³")
        fields = (
            "time",
            "height",
            "temperature",
            "pressure",
            "volume",
            "vapor",
        )
        for name in fields:
            object.__setattr__(self, name, _vector(getattr(self, name), name))
        if (
            len(self.time) < 2
            or np.any(np.diff(self.time) <= 0)
            or any(getattr(self, k).shape != self.time.shape for k in fields)
        ):
            raise ValueError(
                "require aligned samples and strictly increasing seconds"
            )
        if any(
            np.any(getattr(self, k) <= 0)
            for k in ("temperature", "pressure", "volume")
        ) or np.any(self.vapor < 0):
            raise ValueError("invalid physical forcing values")
        self._validate_initial_state()
        for v in (self.organic.rates, self.organic.totals):
            if len(v) not in (0, len(self.time) - 1):
                raise ValueError(
                    "organic forcing requires one value per interval"
                )
        self._validate_reference_outputs()

    def _validate_reference_outputs(self):
        for name, shape in (
            ("reference_radius", (len(self.time), len(self.initial_wet))),
            ("reference_liquid", self.time.shape),
        ):
            v = getattr(self, name)
            if v is not None:
                a = np.array(v, dtype=float, copy=True)
                if (
                    a.shape != shape
                    or not np.isfinite(a).all()
                    or np.any(a < 0)
                ):
                    raise ValueError(f"invalid {name} comparison outputs")
                a.setflags(write=False)
                object.__setattr__(self, name, a)

    def _validate_initial_state(self):
        initial = replace(
            self.initial,
            dry_radius=_vector(self.initial.dry_radius, "dry radii"),
            number=_vector(self.initial.number, "number"),
        )
        if (
            not np.isfinite([initial.dry_air_mass, initial.kappa]).all()
            or min(initial.dry_air_mass, initial.kappa) <= 0
            or not initial.number.size
            or initial.number.shape != initial.dry_radius.shape
            or np.any(initial.number < 0)
            or np.any(initial.dry_radius <= 0)
        ):
            raise ValueError("invalid initial parcel inventories")
        if not np.allclose(
            [
                initial.height,
                initial.pressure,
                initial.temperature,
                initial.vapor,
            ],
            [
                self.height[0],
                self.pressure[0],
                self.temperature[0],
                self.vapor[0],
            ],
            rtol=1e-12,
            atol=0,
        ):
            raise ValueError(
                "initial state disagrees with first forcing record"
            )
        object.__setattr__(self, "initial", initial)
        wet = _vector(self.initial_wet, "initial wet radii")
        if wet.shape != initial.number.shape or np.any(
            wet < initial.dry_radius
        ):
            raise ValueError("invalid initial wet radii")
        object.__setattr__(self, "initial_wet", wet)

    @classmethod
    def record(
        cls,
        trajectory,
        parcel_id="ascent",
        provenance="Particula-generated ascent",
    ):
        """Record actual output times, including the initial wet spectrum."""
        t = trajectory
        return cls(
            parcel_id,
            provenance,
            t.time,
            t.height,
            t.temperature,
            t.pressure,
            t.volume,
            t.vapor,
            t.initial,
            t.wet_radius[0],
            reference_radius=t.wet_radius,
            reference_liquid=t.liquid * t.initial.dry_air_mass,
        )

    def sample_every(self, stride):
        """Thin forcing records and retain both endpoints."""
        if type(stride) is not int or stride < 1:
            raise ValueError("stride must be a positive integer")
        idx = np.unique(
            np.r_[np.arange(0, len(self.time), stride), len(self.time) - 1]
        )
        if stride != 1 and (self.organic.rates or self.organic.totals):
            raise ValueError(
                "coarsen the state trace before specifying interval sources"
            )
        names = (
            "time",
            "height",
            "pressure",
            "temperature",
            "volume",
            "vapor",
            "reference_radius",
            "reference_liquid",
        )
        return replace(
            self,
            **{
                k: getattr(self, k)[idx]
                for k in names
                if getattr(self, k) is not None
            },
        )

    def _metadata(self):
        initial = asdict(self.initial)
        for k in ("dry_radius", "number"):
            initial[k] = initial[k].tolist()
        return {
            **CONVENTIONS,
            "parcel_id": self.parcel_id,
            "provenance": self.provenance,
            "initial": initial,
            "dry_density_kg_m3": self.dry_density,
            "initial_wet_m": self.initial_wet.tolist(),
            "organic": asdict(self.organic),
            "owners": {"water": "trace", self.organic.name: "Particula"},
        }

    def identity(self):
        """Digest of forcing, owners and initial state for continuation."""
        h = hashlib.sha256(
            json.dumps(self._metadata(), sort_keys=True).encode()
        )
        for v in (
            self.time,
            self.height,
            self.temperature,
            self.pressure,
            self.volume,
            self.vapor,
        ):
            h.update(v.astype("<f8", copy=False).tobytes())
        return h.hexdigest()

    def save(self, path):
        """Write CSV forcing and sibling JSON with metadata and comparisons."""
        path = Path(path)
        if path.suffix.lower() != ".csv":
            raise ValueError("trace table must use a .csv suffix")
        with TemporaryDirectory(dir=path.parent) as temp:
            table = Path(temp) / "forcing.csv"
            sidecar = table.with_suffix(".json")
            with table.open("w", newline="") as f:
                w = csv.writer(f, lineterminator="\n")
                w.writerow(COLUMNS)
                for row in zip(
                    self.time,
                    self.height,
                    self.temperature,
                    self.pressure,
                    self.volume,
                    self.vapor,
                    strict=True,
                ):
                    w.writerow((self.parcel_id, *row))
            metadata = self._metadata()
            metadata["table_sha256"] = hashlib.sha256(
                table.read_bytes()
            ).hexdigest()
            for k in ("reference_radius", "reference_liquid"):
                v = getattr(self, k)
                if v is not None:
                    metadata[k] = v.tolist()
            sidecar.write_text(json.dumps(metadata, allow_nan=False) + "\n")

            table.replace(path)
            sidecar.replace(path.with_suffix(".json"))

    @classmethod
    def load(cls, path):
        """Read one trajectory and verify its declared units and owners."""
        path = Path(path)
        meta = json.loads(path.with_suffix(".json").read_text())
        if any(meta.get(k) != v for k, v in CONVENTIONS.items()):
            raise ValueError(
                "unknown forcing units, sampling or trace metadata"
            )
        table = path.read_bytes()
        digest = meta.get("table_sha256")
        if digest is not None and hashlib.sha256(table).hexdigest() != digest:
            raise ValueError("forcing table and metadata digest disagree")
        from io import StringIO

        with StringIO(table.decode("utf-8"), newline="") as f:
            rows = csv.DictReader(f)
            if tuple(rows.fieldnames or ()) != COLUMNS:
                raise ValueError("unexpected forcing columns or units")
            values = []
            for row in rows:
                if row["parcel_id"] != meta["parcel_id"] or None in row:
                    raise ValueError(
                        "trace rows must match declared parcel ID and columns"
                    )
                values.append([float(row[k]) for k in COLUMNS[1:]])
        organic = OrganicForcing(**meta["organic"])
        if meta.get("owners") != {"water": "trace", organic.name: "Particula"}:
            raise ValueError("invalid species owners")
        if len(values) < 2:
            raise ValueError("require at least two forcing records")
        return cls(
            meta["parcel_id"],
            meta["provenance"],
            *np.array(values).T,
            ParcelInitialState(**meta["initial"]),
            meta["initial_wet_m"],
            organic,
            meta.get("reference_radius"),
            meta.get("reference_liquid"),
            meta.get("dry_density_kg_m3"),
        )
