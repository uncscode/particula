"""Attach PALMS composition to the fixed SABRE windows by time overlap."""

import hashlib

import numpy as np
import pandas as pd
from join_chemistry import ARCHIVE, HERE, STUDY, read_icartt

KEYS = ["flight", "start_s", "stop_s"]
FINAL_FRACTIONS = {
    "OrgSulfMF_PALMS": "org_sulf_mf",
    "SulfNeut_PALMS": "nh4_so4_molar",
    "neutralization_capped": "nh4_so4_capped",
    "SulfOrgNitFrac_PALMS": "sulf_org_nit_number_fraction",
    "MeteoricFrac_PALMS": "meteoric_number_fraction",
    "BioBurnFrac_PALMS": "bb_number_fraction",
}


def overlap_average(frame, flight, columns):
    """Treat each reported 180 s composition as constant over its interval."""
    left = frame.Time_Start.to_numpy()
    right = frame.Time_Stop.to_numpy()
    np.testing.assert_array_equal(right - left, 180)
    if np.any(left[1:] < right[:-1]):
        raise ValueError("Unexpected overlap between adjacent PALMS records")
    original = flight.start_s.to_numpy()[:, None] - 0.5
    end = flight.stop_s.to_numpy()[:, None] + 0.5
    overlap = np.maximum(0, np.minimum(right, end) - np.maximum(left, original))
    out = {}
    if "OrgSulfMF_PALMS" in columns:
        source_rows = np.isfinite(frame.OrgSulfMF_PALMS.to_numpy())
        out["palms_org_interval_start_utc_s"] = [
            "|".join(str(int(a)) for a in left[(row > 0) & source_rows])
            for row in overlap
        ]
    for source, name in columns.items():
        values = frame[source].to_numpy()
        valid = np.isfinite(values)
        weights = overlap[:, valid]
        duration = weights.sum(axis=1)
        enough = duration >= 0.75 * (end - original).ravel()
        out[name] = np.divide(
            weights @ values[valid],
            duration,
            out=np.full(len(flight), np.nan),
            where=enough,
        )
        out[name + "_overlap_s"] = duration
    return out


def main():
    """Rebuild the aligned composition and corresponding source manifest."""
    original = pd.read_csv(STUDY / "flight-windows.csv")
    chunks, manifest, contributing = [], [], []
    for date, flight in original[KEYS].groupby("flight", sort=False):
        part = flight.copy()
        for product, revision in (
            ("PALMSng", "R1"),
            ("PALMSng-Chem-Mass", "R0"),
        ):
            file = STUDY / "raw" / f"SABRE-{product}_WB57_{date}_{revision}.ict"
            frame, _ = read_icartt(file)
            blob = file.read_bytes()
            manifest.append(
                dict(
                    flight=date,
                    product=f"SABRE-{product}",
                    revision=revision,
                    filename=file.name,
                    bytes=len(blob),
                    sha256=hashlib.sha256(blob).hexdigest(),
                    archive=ARCHIVE,
                )
            )
            if product == "PALMSng":
                joined = overlap_average(
                    frame, flight, {"OrgSulfMF_PALMS": "fraction"}
                )
                ids = set()
                for starts, frac in zip(
                    joined["palms_org_interval_start_utc_s"],
                    joined["fraction"],
                    strict=True,
                ):
                    if np.isfinite(frac):
                        ids.update(int(t) for t in starts.split("|") if t)
                for record in frame[frame.Time_Start.isin(ids)].itertuples():
                    contributing.append(
                        dict(
                            flight=date,
                            start_s=record.Time_Start,
                            stop_s=record.Time_Stop,
                            org_sulf_mf=record.OrgSulfMF_PALMS,
                            nh4_so4_molar=record.SulfNeut_PALMS,
                        )
                    )
                frame["neutralization_capped"] = frame.SulfNeut_PALMS.clip(
                    upper=2
                )
                columns = FINAL_FRACTIONS
            else:
                # Ratio of means is meaningful only on common valid support.
                columns = {
                    "OrgMass_PALMS": "org_mass_ug_stdm3",
                    "SulfateMass_PALMS": "sulfate_mass_ug_stdm3",
                }
                valid = frame[list(columns)].notna().all(axis=1)
                valid &= (frame[list(columns)] >= 0).all(axis=1)
                frame.loc[~valid, list(columns)] = np.nan
            part = part.assign(**overlap_average(frame, flight, columns))
        total = part.org_mass_ug_stdm3 + part.sulfate_mass_ug_stdm3
        part["bulk_org_sulf_mf"] = part.org_mass_ug_stdm3 / total.where(
            total > 0
        )
        chunks.append(part)
    result = pd.concat(chunks).sort_index()
    pd.testing.assert_frame_equal(result[KEYS], original[KEYS])
    pd.DataFrame(contributing).to_csv(
        HERE / "palms-contributing-intervals.csv", index=False
    )
    result.to_csv(HERE / "flight-composition.csv", index=False)
    pd.DataFrame(manifest).to_csv(HERE / "composition-sources.csv", index=False)


if __name__ == "__main__":
    main()
