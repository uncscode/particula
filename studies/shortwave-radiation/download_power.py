"""Acquire hourly NASA POWER solar fields and preserve their provenance."""

import argparse
import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlencode
from urllib.request import urlopen

from particula.data.power import POWER_CHANNELS


def download(destination, latitude, longitude, start, end):
    """Fetch one point and write a new directory with source and manifest."""
    if start > end:
        raise ValueError("Start date must precede or equal end date")
    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        raise ValueError("Latitude or longitude out of range")
    query = {
        "parameters": ",".join(POWER_CHANNELS),
        "community": "RE",
        "longitude": longitude,
        "latitude": latitude,
        "start": start.strftime("%Y%m%d"),
        "end": end.strftime("%Y%m%d"),
        "format": "JSON",
        "time-standard": "UTC",
    }
    endpoint = "https://power.larc.nasa.gov/api/temporal/hourly/point?"
    url = endpoint + urlencode(query)
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Choose a new download directory: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent) as work:
        packet = Path(work) / "download"
        packet.mkdir()
        source = packet / "power-hourly.json"
        with urlopen(url, timeout=90) as response:
            source.write_bytes(response.read())
        data = json.loads(source.read_bytes())
        header = data["header"]
        if header["time_standard"] != "UTC":
            raise ValueError("POWER returned an unexpected time standard")
        values = json.dumps(data["properties"]["parameter"], sort_keys=True)
        manifest = {
            "source": "NASA Langley Research Center POWER Hourly API",
            "api_version": header["api"]["version"],
            "source_products": data["header"]["sources"],
            "license": "Public domain. NASA POWER requests acknowledgement.",
            "terms": "https://power.larc.nasa.gov/docs/referencing/",
            "request": url,
            "retrieved_utc": datetime.now(UTC).isoformat(),
            "file": source.name,
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "parameter_sha256": hashlib.sha256(values.encode()).hexdigest(),
            "quality": "Reject missing, fill, nonfinite and negative values at use.",
            "transformation": "Wh/m² over one UTC hour divided by 1 h yields mean W/m².",
        }
        (packet / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        packet.rename(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--latitude", type=float, required=True)
    parser.add_argument("--longitude", type=float, required=True)
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    args = parser.parse_args()
    download(**vars(args))
