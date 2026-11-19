"""Retrieve pulse capture rates and the distribution across SMPS channels."""

import csv

import numpy as np
from instrument import HERE, SCAN_CENTERS, TARGETS, save
from pulse_campaign import SEEDS, cases, label
from scipy.stats import t
from study import ROOT


def interval(values):
    """95% Student-t half-width across independent seed estimates."""
    return (
        t.ppf(0.975, len(values) - 1)
        * np.std(values, ddof=1)
        / np.sqrt(len(values))
    )


def channel_spectra(seed_data, common, observed):
    """Summarize channel fractions for each loading and output time."""
    spectra = []
    for time in (0, 60, 150, 300):
        for loading in (0, 10000):
            sig = np.array(
                [
                    [
                        float(
                            table[4.0, str(center), loading, time]["signal_sum"]
                        )
                        for center in SCAN_CENTERS
                    ]
                    for table in seed_data
                ]
            )
            norm = sig / sig.sum(axis=1)[:, None]
            for j, center in enumerate(SCAN_CENTERS):
                spectra.append(
                    dict(
                        **common,
                        loading_cm3=loading,
                        time_s=time,
                        channel_nm=center,
                        mean_signal_per_box=sig[:, j].mean() / 2048,
                        signal_fraction=norm[:, j].mean(),
                        mc95_halfwidth=interval(norm[:, j]),
                        observed_pooled_fraction=observed[j],
                    )
                )
    return spectra


def summarize():
    """Retain both observed event rates and integrated tracer hazards."""
    data = ROOT / "docs/data/nacl-scavenging/archived-smps-counts.csv"
    counts = np.genfromtxt(data, names=True, delimiter=",")["pooled_raw_counts"]
    observed = counts / counts.sum()
    all_seeds, results, spectra = [], [], []
    for dt, model, conditioned in cases():
        seed_data = []
        source = "smps_conditioned" if conditioned else "reference"
        common = dict(source=source, model=model, dt_s=dt)
        for seed in SEEDS:
            with (
                HERE / (label(dt, model, seed, conditioned) + "-channels.csv")
            ).open() as stream:
                table = {}
                for row in csv.DictReader(stream):
                    key = (
                        float(row["target_nm"]),
                        row["readout"],
                        float(row["loading_cm3"]),
                        float(row["time_s"]),
                    )
                    table[key] = row
                seed_data.append(table)
        spectra.extend(channel_spectra(seed_data, common, observed))
        for target in (4.0,) if conditioned else TARGETS:
            for readout in (
                ["reference", "archived_scan"]
                if target == 4.0
                else ["reference"]
            ):
                channels = (
                    ["reference"]
                    if readout == "reference"
                    else [str(c) for c in SCAN_CENTERS]
                )
                values = []
                for seed, table in zip(SEEDS, seed_data, strict=True):
                    selected = [table[target, c, 10000, 300] for c in channels]

                    def total(field, selected=selected):
                        return sum(float(r[field]) for r in selected)

                    exposure = total("weighted_exposure")
                    thermal = total("thermal_expectation")
                    mech = total("mechanical_expectation")
                    np.testing.assert_equal(
                        exposure > 0 and thermal > 0 and mech >= 0, True
                    )
                    np.testing.assert_equal(
                        total("fallback_expectation") <= thermal * (1 + 1e-12),
                        True,
                    )
                    base = table[target, "reference", 10000, 300]
                    blank = table[target, "reference", 0, 300]
                    np.testing.assert_equal(
                        float(blank["unweighted_capture"]), 0
                    )
                    values.append(
                        dict(
                            beta_cm3_s=total("weighted_capture") / exposure,
                            beta_hazard_cm3_s=(thermal + mech) / exposure,
                            mechanical_cm3_s=mech / exposure,
                            fallback_fraction=total("fallback_expectation")
                            / thermal,
                            captures=int(float(base["unweighted_capture"])),
                            mechanical_events=int(
                                float(base["mechanical_events"])
                            ),
                            expected_mechanical_events=float(
                                base["unweighted_mechanical_expectation"]
                            ),
                        )
                    )
                    all_seeds.append(
                        dict(
                            **common,
                            target_nm=target,
                            readout=readout,
                            seed=seed,
                            **values[-1],
                        )
                    )
                beta = [v["beta_cm3_s"] for v in values]
                results.append(
                    dict(
                        **common,
                        target_nm=target,
                        readout=readout,
                        beta_cm3_s=np.mean(beta),
                        mc95_halfwidth=interval(beta),
                        **{
                            k: np.mean([v[k] for v in values])
                            for k in (
                                "beta_hazard_cm3_s",
                                "mechanical_cm3_s",
                                "fallback_fraction",
                            )
                        },
                        **{
                            k: sum(v[k] for v in values)
                            for k in (
                                "captures",
                                "mechanical_events",
                                "expected_mechanical_events",
                            )
                        },
                    )
                )
    save("pulse-kernels-summary.csv", results)
    save("pulse-kernels-seeds.csv", all_seeds)
    save("pulse-smps-spectrum.csv", spectra)


if __name__ == "__main__":
    summarize()
