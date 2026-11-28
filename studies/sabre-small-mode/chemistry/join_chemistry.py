"""Match final DLH and LIF measurements to the fixed SABRE windows."""

import hashlib
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(STUDY))
from join_flights import read_icartt, windows  # noqa: E402

ARCHIVE = (
    "https://csl.noaa.gov/groups/csl7/measurements/2023sabre/wb57/DataDownload/"
)
PRODUCTS = {
    "DLH": {"H2O_DLH": "h2o_ppmv", "RHw_DLH": "rhw_pct", "RHi_DLH": "rhi_pct"},
    "NOy-LIF": {
        "NO_LIF": "no_pptv",
        "NO2_LIF": "no2_pptv",
        "NOy_LIF": "noy_pptv",
    },
}


def main():
    """Save mean observed quantities and per-variable valid sample counts."""
    original = pd.read_csv(STUDY / "flight-windows.csv")
    keys = ["flight", "start_s", "stop_s"]
    chunks, manifest = [], []
    for date, flight in original[keys].groupby("flight", sort=False):
        part = flight.copy()
        starts, stops = (part[key].to_numpy() for key in keys[1:])
        for product, columns in PRODUCTS.items():
            file = STUDY / "raw" / f"SABRE-{product}_WB57_{date}_R0.ict"
            data, header = read_icartt(file)
            if product == "DLH":
                if data.columns[0] != "Time_Start":
                    raise ValueError("Expected DLH one-second interval start")
                data.iloc[:, 0] += 0.5
            elif data.columns[0] != "Time_Mid":
                raise ValueError("Expected LIF sample midpoint")
            for variable, field in columns.items():
                mean, count = windows(
                    data, starts, stops, [variable], minimum=57
                )
                part[field] = mean[:, 0]
                part[field + "_n"] = count
            blob = file.read_bytes()
            manifest.append(
                dict(
                    flight=date,
                    product=f"SABRE-{product}",
                    revision="R0",
                    filename=file.name,
                    bytes=len(blob),
                    sha256=hashlib.sha256(blob).hexdigest(),
                    archive=ARCHIVE,
                )
            )
        chunks.append(part)
    result = pd.concat(chunks).sort_index()
    pd.testing.assert_frame_equal(result[keys], original[keys])
    result.to_csv(HERE / "flight-chemistry.csv", index=False)
    pd.DataFrame(manifest).to_csv(HERE / "chemistry-sources.csv", index=False)


if __name__ == "__main__":
    main()
