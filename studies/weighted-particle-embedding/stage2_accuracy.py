"""Score conservation and forecasts against refined references."""

import argparse
import csv
from pathlib import Path

import numpy as np

# BDF uses normalized states. Freeze this physical-unit reporting floor before
# fits.
MASS_ABSOLUTE = 1e-14  # kg/m3
MASS_RELATIVE = 1e-3
NEAR_ZERO = MASS_ABSOLUTE / MASS_RELATIVE


def write_rows(file, rows):
    """Write one header and retain every scored row."""
    file.parent.mkdir(parents=True, exist_ok=True)
    with file.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def phase_score(pred, truth, coarse, time):
    """Bound the pointwise error by the last reference refinement difference."""
    pred, truth, coarse = map(np.asarray, (pred, truth, coarse))
    if not (pred.shape == truth.shape == coarse.shape):
        raise ValueError("Misaligned reference or forecast")
    if not all(np.isfinite(x).all() for x in (pred, truth, coarse)):
        raise ValueError("Nonfinite reference or forecast")
    # The near-zero tolerance is absolute, with all concentrations in kg/m³.
    scale = np.maximum(abs(truth), NEAR_ZERO)
    delta = abs(pred - truth)
    uncertainty = abs(coarse - truth)
    qualified = (delta + uncertainty < MASS_RELATIVE * scale).all(axis=0)
    failed = (delta - uncertainty >= MASS_RELATIVE * scale).any(axis=0)
    bad = np.flatnonzero(~qualified)
    i = int(bad[0]) if len(bad) else len(time)
    resolved = abs(truth) > NEAR_ZERO
    return dict(
        frames=len(time),
        qualified_frames=int(qualified.sum()),
        failed_frames=int(failed.sum()),
        unresolved_frames=int((~(qualified | failed)).sum()),
        initial_qualified=bool(qualified[0]),
        first_unqualified_s=float(time[i]) if i < len(time) else "",
        last_contiguous_s=float(time[i - 1]) if i else "",
        end_s=float(time[-1]),
        worst_resolved_pct=100 * np.max((delta / scale)[resolved], initial=0),
        worst_reference_refinement_pct=100 * np.max(uncertainty / scale),
        worst_upper_tolerance_units=np.max(
            (delta + uncertainty) / (MASS_RELATIVE * scale)
        ),
        near_zero_n=int((~resolved).sum()),
        near_zero_worst_kg_m3=np.max(delta[~resolved], initial=0),
        near_zero_reference_worst_kg_m3=np.max(
            uncertainty[~resolved], initial=0
        ),
        minimum_prediction_kg_m3=pred.min(),
    )


def ledger(wet, gas, t):
    """Relative inventory drift within each leg, including prescribed jumps."""
    total = wet + gas
    initial = total[:, :1]
    # At a reset only gas can be replaced. A particle mass jump is a defect.
    reset = np.r_[False, np.diff(t) == 0]
    source = np.diff(gas, axis=1, prepend=gas[:, :1]) * reset[None, :, None]
    expected = initial + np.cumsum(source, axis=1)
    return abs(total - expected) / np.maximum(abs(expected), NEAR_ZERO)


def summarize(rows, cohort, model, phase, pred, fine, coarse, t, drift):
    """Score each volatile lane independently across boxes."""
    for j in range(fine.shape[-1]):
        rows.append(
            dict(
                cohort=cohort,
                model=model,
                phase=phase,
                species_index=j,
                own_budget_relative_max=drift[:, :, j].max(),
                **phase_score(pred[:, :, j], fine[:, :, j], coarse[:, :, j], t),
            )
        )


def wide(root, out):
    """Align native and learned wide-vapor forecasts to both BDF refinements."""
    rows = []
    for folder in sorted((root / "forecasts").iterdir()):
        key = folder.name
        bdf = root / "bdf" / key / "f8"
        fine, coarse = [
            dict(np.load(next(bdf.glob(f"*-r{x}-coupled.npz"))))
            for x in ("1e-09", "1e-07")
        ]
        t = fine["time"]
        forecasts = {}
        for f in sorted(folder.glob("*.npz")):
            with np.load(f) as d:
                if "particle_kg_m3" in d:
                    np.testing.assert_array_equal(d["ids"], fine["ids"])
                    np.testing.assert_array_equal(d["time_s"], t)
                    forecasts[f.stem] = (d["particle_kg_m3"], d["gas_kg_m3"])
        for n in (32, 64, 128):
            with np.load(
                next((root / "validation" / f"{key}-s{n}-k8").glob("*.npz"))
            ) as d:
                np.testing.assert_array_equal(d["ids"], fine["ids"])
                np.testing.assert_array_equal(d["time"], t)
                forecasts[f"native-{n}"] = tuple(
                    d[k].swapaxes(0, 1) for k in ("bulk", "gas")
                )
        forecasts["full-BDF-coarse"] = coarse["water"], coarse["gas"]
        for model, (wet, gas) in forecasts.items():
            drift = ledger(wet, gas, t)
            for phase, pred, refname in (
                ("particle", wet, "water"),
                ("gas", gas, "gas"),
            ):
                summarize(
                    rows,
                    key,
                    model,
                    phase,
                    pred,
                    fine[refname],
                    coarse[refname],
                    t,
                    drift,
                )
    write_rows(out, rows)


def periodic(root, out):
    """Score the first forcing cycle and all four cycles."""
    rows = []
    for geometry in ("conditioned", "proportional"):
        folder = root / f"{geometry}-c4"
        fine, coarse, compressed = [
            dict(np.load(folder / f"{name}.npz"))
            for name in ("full-tol1e-09", "full-tol1e-07", "ten-tol1e-09")
        ]
        for plateaus in (96, 384):
            n = len(fine["time"]) * plateaus // 384
            t = fine["time"][:n]
            for name, data in (
                ("full-BDF-coarse", coarse),
                ("ten-node-BDF", compressed),
            ):
                drift = ledger(data["wet"][:, :n], data["gas"][:, :n], t)
                for phase, field in (("particle", "wet"), ("gas", "gas")):
                    summarize(
                        rows,
                        f"{geometry}-{plateaus}",
                        name,
                        phase,
                        data[field][:, :n],
                        fine[field][:, :n],
                        coarse[field][:, :n],
                        t,
                        drift,
                    )
    write_rows(out, rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("wide", "periodic"))
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    (wide if args.kind == "wide" else periodic)(args.root, args.output)
