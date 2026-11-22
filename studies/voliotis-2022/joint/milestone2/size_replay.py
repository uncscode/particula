"""Replay frozen mass/count fits against Manchester's reserved size records."""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import warp as wp

HERE = Path(__file__).resolve().parent
RESOLVED = HERE.parent / "resolved"
sys.path.insert(0, str(RESOLVED))
from simulate import BANK, Chamber  # noqa: E402, I001 (sets up the study imports)

from bank import Channel  # noqa: E402
from scoring import Targets, spectra  # noqa: E402

# These study settings transfer seed, uptake and nucleation physics from the
# previous frozen bulk-to-particle replay without tuning against size data.
PARTICLE_SETTINGS = [1.5, 0.0, 0.0]  # seed GSD, volume-weighted fraction, Jmax
RESOLUTIONS = [(1024, 30, 201), (2048, 15, 201), (2048, 15, 202)]


def replay(case, n, dt, seed):
    """Compute all 15 trajectories with the study's native stochastic solver."""
    bank = [
        Channel(**{k: channel[k] for k in Channel.__dataclass_fields__})
        for channel in case["channels"]
    ]
    c = Chamber(bank)
    if len(c.ids) != 15:
        raise ValueError("Expected all 15 Manchester campaign experiments")
    p = np.r_[case["x"], case["dose"], PARTICLE_SETTINGS]
    # Targets.evaluate also reports water uptake; dummy fixed κ values below
    # only pass its unchanged scoring interface. No water output is used.
    result = c.run(p[None, :], c.ids, n=n, dt=dt, rng=seed)
    target = Targets(c)
    _, details = target.evaluate(result, [np.r_[p, 0.046, 0]], c.ids, True)
    mode = pd.concat([d for name, d in details if name == "modes"])
    mode = mode[["run", "time_s", "mode_nm", "predicted_nm"]].copy()

    row = c.ids.index(24)
    histogram = result[1][row]
    air = result[4][row]
    # Include all digitized cells and retain the established quality flag.
    spectrum = pd.read_csv(RESOLVED / "size_data/run24_psd.csv")[
        [
            "time_s",
            "low_nm",
            "high_nm",
            "dndlog10d_cm3",
            "reading_band_cm3",
            "valid_fraction",
        ]
    ]
    spectrum["used"] = (
        spectrum.valid_fraction.ge(0.5) & spectrum.dndlog10d_cm3.notna()
    )
    spectrum["predicted"] = spectra(histogram, air, spectrum)
    gmd = pd.read_csv(RESOLVED / "size_data/run24_gmd.csv")[
        ["time_s", "gmd_nm", "reading_band_nm"]
    ]
    centers = 2 * 1.12 ** (np.arange(60) + 0.5)
    visible = (centers >= 22) & (centers <= 540)
    counts = histogram[:, :60][:, visible]
    diameter = np.exp((counts @ np.log(centers[visible])) / counts.sum(axis=1))
    gmd["predicted_nm"] = np.interp(gmd.time_s, target.time, diameter)
    runs = pd.concat([d for name, d in details if name == "runs"])
    return {"modes": mode, "psd": spectrum, "gmd": gmd, "runs": runs}


def summarize(tables):
    """Score observed spectra and modes at their times and diameters."""
    spec = tables["psd"].query("used")
    snap = spec.query("time_s == 7200")
    mode = tables["modes"]
    last = tables["gmd"].query("time_s == 18000").iloc[0]
    error = np.abs(spec.predicted - spec.dndlog10d_cm3)
    all_cells = tables["psd"].dropna(subset=["dndlog10d_cm3"])
    peaks = {}
    for label, column in [
        ("observed", "dndlog10d_cm3"),
        ("predicted", "predicted"),
    ]:
        peak = snap.loc[snap[column].idxmax()]
        peaks[f"{label}_peak_low_nm"] = peak.low_nm
        peaks[f"{label}_peak_high_nm"] = peak.high_nm
    return {
        "mode_mae_nm": np.abs(mode.predicted_nm - mode.mode_nm).mean(),
        "mode_rows": len(mode),
        "mode_runs": mode.run.nunique(),
        "psd_rmse_cm3_decade": np.sqrt(np.mean(error**2)),
        "psd_excess_rmse_cm3_decade": np.sqrt(
            np.mean(np.maximum(error - spec.reading_band_cm3, 0) ** 2)
        ),
        "psd_rmse_including_masked_cell": np.sqrt(
            np.mean((all_cells.predicted - all_cells.dndlog10d_cm3) ** 2)
        ),
        "psd_scored": len(spec),
        "psd_digitized": len(tables["psd"]),
        "gmd_5h_observed_nm": last.gmd_nm,
        "gmd_5h_predicted_nm": last.predicted_nm,
        **peaks,
    }


def check_compact_network(case):
    """Check run 24 with removed archived channels restored at zero rate."""
    reference = Chamber()
    channels = [
        Channel(**{k: c[k] for k in Channel.__dataclass_fields__})
        for c in case["channels"]
    ]
    indices = [BANK.index(c) for c in channels]
    if sorted(indices) != indices:
        raise ValueError("Incumbent channel order differs from the archive")
    full_rates = np.zeros(len(BANK))
    full_rates[indices] = case["x"][16:]
    complete = np.r_[
        case["x"][:16], full_rates, case["dose"], PARTICLE_SETTINGS
    ]
    compact = np.r_[case["x"], case["dose"], PARTICLE_SETTINGS]
    controls = dict(run_ids=[24], n=512, dt=15, rng=201, minutes=30)
    history, dist, _, _, _ = reference.run(complete[None], **controls)
    trial, trial_dist, _, _, _ = Chamber(channels).run(compact, **controls)
    # Histograms include both exact counts and summed composition fractions.
    np.testing.assert_allclose(dist, trial_dist, atol=1e-9, rtol=1e-12)
    np.testing.assert_allclose(history, trial, atol=1e-9, rtol=1e-12)
    return dict(
        run=24,
        minutes=30,
        seed=201,
        n=512,
        step_s=15,
        compact_channels=len(channels),
        padded_channels=len(BANK),
        max_count_bin_difference=float(
            np.abs((dist - trial_dist)[..., :60]).max()
        ),
    )


def main():
    """Save campaign checks and the machine-readable size comparisons."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("--output", type=Path, default=HERE / "size_results")
    parser.add_argument(
        "--smoke", action="store_true", help="Run the coarse realization only"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    content = args.inputs.read_bytes()
    cases = json.loads(content)["cases"]
    checks = []
    records = {}
    device = wp.get_device("cuda:0")
    if not device.is_cuda:
        raise RuntimeError("This study requires CUDA")
    for n, dt, seed in RESOLUTIONS[:1] if args.smoke else RESOLUTIONS:
        prior = {}
        for label, case in cases.items():
            identity = json.dumps(
                [case["x"], case["dose"], case["channels"]], sort_keys=True
            )
            start = perf_counter()
            reused = identity in prior
            if not reused:
                prior[identity] = replay(case, n, dt, seed)
            data = prior[identity]
            params = dict(case=label, particles=n, step_s=dt, seed=seed)
            checks.append(
                dict(
                    **params,
                    **summarize(data),
                    identical_reused=reused,
                    elapsed_s=perf_counter() - start,
                )
            )
            for name, table in data.items():
                records.setdefault(name, []).append(table.assign(**params))
            print(checks[-1], flush=True)
    pd.DataFrame(checks).to_csv(args.output / "summary.csv", index=False)
    for key, tables in records.items():
        merged = pd.concat(tables, ignore_index=True)
        merged.to_csv(
            args.output / f"{key}.csv.gz",
            index=False,
            compression={"method": "gzip", "mtime": 0},
        )
        if key == "modes":
            mode_errors = merged.assign(
                abs_error_nm=(merged.predicted_nm - merged.mode_nm).abs()
            )
            mode_errors.groupby(
                ["case", "particles", "step_s", "seed", "run"]
            ).abs_error_nm.agg(["mean", "count"]).rename(
                columns={"mean": "mae_nm", "count": "observations"}
            ).to_csv(args.output / "mode_by_run.csv")
    metadata = dict(
        inputs_sha256=hashlib.sha256(content).hexdigest(),
        device=device.name,
        compact_network_control=check_compact_network(cases["incumbent"]),
        warp=wp.__version__,
        particle_settings=PARTICLE_SETTINGS,
        resolutions=RESOLUTIONS[:1] if args.smoke else RESOLUTIONS,
    )
    (args.output / "provenance.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
