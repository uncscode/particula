"""Read hourly NASA POWER surface shortwave irradiance from local JSON.

POWER's hourly point API labels the start of each hourly interval. The RE
community reports energy in Wh/m² for each hour. Dividing by its duration
in hours gives the hourly mean in W/m². The derived direct-normal, diffuse
horizontal and global horizontal fields retain their independent estimates.
See https://power.larc.nasa.gov/docs/services/api/temporal/hourly/.
"""

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from particula.optics.solar import SolarInput

POWER_CHANNELS = (
    "ALLSKY_SFC_SW_DWN",
    "ALLSKY_SFC_SW_DNI",
    "ALLSKY_SFC_SW_DIFF",
)


@dataclass(frozen=True)
class PowerHour:
    """Hourly average surface irradiances [W/m²] from a POWER grid cell.

    Coordinates follow the provider response in degrees and metres. The
    timestamp is the UTC start of a one-hour interval. The digest identifies
    the source file, including its original values, units and metadata.
    """

    start_utc: datetime
    longitude_deg: float
    latitude_deg: float
    elevation_m: float
    global_horizontal: float
    direct_normal: float
    diffuse_horizontal: float
    api_version: str
    sha256: str

    def slab_boundary(
        self,
        spectral_shape: SolarInput,
        *,
        direct_fraction: float,
        diffuse_fraction: float,
    ) -> SolarInput:
        """Place the hourly surface fields above an additional modeled slab.

        Supply the cosine of solar zenith, wavelengths and relative spectral
        shapes in `spectral_shape`. State what fraction [0, 1] of each
        broadband component lies in those wavelengths. Each output spectrum
        integrates to its chosen fraction of the corresponding POWER mean.
        Projection onto a horizontal plane happens inside the solver.
        """
        wave = spectral_shape.wavelength_m
        weights = spectral_shape.quadrature_weights_m

        def distribute(shape, fraction, total):
            if not np.isfinite(fraction) or not 0 <= fraction <= 1:
                raise ValueError("Band energy fraction must lie in [0, 1]")
            if total * fraction == 0:
                return np.zeros_like(wave)
            area = (
                np.trapezoid(shape, wave)
                if weights is None
                else np.dot(weights, shape)
            )
            if area <= 0:
                raise ValueError("Positive band energy needs a nonzero shape")
            return total * fraction * shape / area

        direct = distribute(
            spectral_shape.direct_normal, direct_fraction, self.direct_normal
        )
        diffuse = distribute(
            spectral_shape.diffuse_horizontal,
            diffuse_fraction,
            self.diffuse_horizontal,
        )
        if spectral_shape.mu0 == 0 and np.any(direct > 0):
            raise ValueError("Positive direct irradiance needs a daytime Sun")
        return SolarInput(
            wave,
            spectral_shape.mu0,
            direct,
            diffuse,
            weights,
            source=(
                f"NASA POWER Hourly {self.api_version}, "
                f"{self.start_utc.isoformat()} + 1 h, input {self.sha256}, "
                f"surface-level fields imposed above slab, "
                f"shape {spectral_shape.source}, "
                f"direct fraction {direct_fraction}, diffuse {diffuse_fraction}"
            ),
            band_bounds_m=spectral_shape.band_bounds_m,
        )


def read_power_hour(
    path: str | Path, start: datetime, *, sha256: str | None = None
) -> PowerHour:
    """Read one UTC hour of all-sky shortwave data, entirely offline.

    Args:
        path: Unmodified hourly point API JSON response in UTC.
        start: Timezone-aware interval start aligned with a UTC hour.
        sha256: Optional expected SHA-256 digest of the raw bytes.

    Raises:
        ValueError: Missing interval, fill value, non-UTC data, unsupported
            unit, invalid coordinates, or a mismatched expected digest.
    """
    if start.utcoffset() is None:
        raise ValueError("POWER interval start must have a timezone")
    start = start.astimezone(UTC)
    if start.minute or start.second or start.microsecond:
        raise ValueError("POWER interval start must fall on a UTC hour")
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if sha256 is not None and digest != sha256:
        raise ValueError("POWER source SHA-256 mismatch")
    data = json.loads(raw)
    key = start.strftime("%Y%m%d%H")
    try:
        header = data["header"]
        if header["time_standard"] != "UTC":
            raise ValueError("Request time-standard=UTC from NASA POWER")
        if data["geometry"]["type"] != "Point":
            raise ValueError("Expected POWER point geometry")
        lon, lat, elevation = map(float, data["geometry"]["coordinates"])
        if (
            not np.isfinite(elevation)
            or not -180 <= lon <= 180
            or not -90 <= lat <= 90
            or elevation == header["fill_value"]
        ):
            raise ValueError("Invalid POWER point coordinates")
        version = header["api"]["version"]
        global_flux, direct, diffuse = _mean_irradiances(
            data, key, header["fill_value"]
        )
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Missing POWER field or hour {key}: {exc}") from exc
    return PowerHour(
        start,
        lon,
        lat,
        elevation,
        global_flux,
        direct,
        diffuse,
        version,
        digest,
    )


def _mean_irradiances(data, key, fill_value):
    """Extract the three independently reported hourly means."""
    flux = []
    for channel in POWER_CHANNELS:
        units = data["parameters"][channel]["units"]
        if units not in ("Wh/m^2", "W/m^2"):
            raise ValueError(f"Unsupported POWER units for {channel}: {units}")
        value = float(data["properties"]["parameter"][channel][key])
        if not np.isfinite(value) or value < 0 or value == fill_value:
            raise ValueError(f"Missing or invalid POWER {channel} at {key}")
        # Each Wh/m² value spans exactly one hour.
        flux.append(value)
    return flux
