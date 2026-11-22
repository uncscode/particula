"""Reconstruct the inorganic CLOUD rate laws in the units of Dunne SI Table S1."""

from pathlib import Path
import csv
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
DATA = np.loadtxt(ROOT / "raw/cloud_data_table.csv", delimiter=",", skiprows=1)
T, ACID, NH3, Q, RH, J = DATA.T
AIR = 1.005 * 101325 / (1.380649e-23 * T) / 1e6
AMMONIA = NH3 * AIR * 1e-18  # pptv to 10^6 molecules cm^-3
WINDOW = (RH >= 33) & (RH <= 43)
BANDS = np.array([208, 223, 248, 259, 269, 278, 292, 298])
BAND = BANDS[np.argmin(abs(T[:, None] - BANDS), axis=1)]
SPLITS = {
    "248_low": (BAND == 248) & (NH3 >= 0.32) & (NH3 <= 0.45),
    "223_low": (BAND == 223) & (NH3 >= 0.04) & (NH3 <= 0.08),
    "248_temperature": BAND == 248,
    "223_temperature": BAND == 223,
    "278_temperature": BAND == 278,
}
# b,n ; t,n ; b,i ; t,i. Exponents for acid, u, v, w.
S1 = np.array(
    [
        [3.95, 9.70, 12.6, -0.00707],
        [2.89, 182, 1.20, -4.19],
        [3.37, -11.5, 25.5, 0.181],
        [3.14, -23.8, 37, 0.227],
    ]
)
S2 = np.array(
    [
        [3.62, 46.3, 245, 0],
        [2.82, 41.2, 252, 0],
        [2.73, 24.1, 166, 0],
        [2.86, 18.3, 208, 0],
    ]
)
TERN = {
    "S1": ([8.0, 3.07], [1.6e-6, 0.00485]),
    "S2": ([6.76, 5.0], [1.3e-4, 5e-7]),
}

with (ROOT / "rate_sources/coefficients.csv").open() as stream:
    precise = {
        row["symbol"]: float(row["S1_ClimaParams"])
        for row in csv.DictReader(stream)
    }
S1_PRECISE = np.array(
    [
        [precise[f"{letter}_{branch}"] for letter in "puvw"]
        for branch in ("b_n", "t_n", "b_i", "t_i")
    ]
)
TERN["S1_precise"] = (
    [precise["p_A_n"], precise["p_A_i"]],
    [precise["a_n"], precise["a_i"]],
)


def cold_condition_label(csv_line, separator=" · "):
    """Label the neutral holdouts by their reported temperature and acid level."""
    index = csv_line - 2
    acid = f"{ACID[index]:g}"
    qualifier = {366: "lower acid", 332: "higher acid"}.get(csv_line)
    parts = [f"{T[index]:g} K", acid]
    if qualifier:
        parts.append(qualifier)
    return separator.join(parts)


def ion_density(temperature, q, ion_coefficient):
    air = 1.005 * 101325 / (1.380649e-23 * temperature) / 1e6
    alpha = (
        6e-8 * np.sqrt(300 / temperature)
        + 6e-26 * air * (300 / temperature) ** 4
    )
    linear = 1 / 480 + ion_coefficient
    return 2 * q / (linear + np.sqrt(linear**2 + 4 * alpha * q))


def channels(table="S1", temperature=T, acid=ACID, ammonia=AMMONIA, q=Q):
    params = {"S1": S1, "S2": S2, "S1_precise": S1_PRECISE}[table]
    pa, aa = TERN[table]
    terms = []
    for i, (p, u, v, w) in enumerate(params):
        lk = u - (
            np.exp(v * (temperature / 1000 - w))
            if table != "S2"
            else v * temperature / 1000
        )
        lac = p * np.log(acid)
        if i in [1, 3]:
            k = i // 2
            logf = (
                np.log(ammonia)
                + lac
                - np.logaddexp(np.log(aa[k]), lac - pa[k] * np.log(ammonia))
            )
        else:
            logf = lac
        terms.append(np.exp(lk + logf))
    n = ion_density(temperature, q, terms[2] + terms[3])
    return np.array([terms[0], terms[1], n * terms[2], n * terms[3]])


def scores(y, pred):
    residual = np.log10(pred) - np.log10(y)
    return dict(
        n=len(y),
        median_factor=float(10 ** np.median(abs(residual))),
        geometric_bias=float(10 ** np.mean(residual)),
        rmse_log10=float(np.mean(residual**2) ** 0.5),
        log_r2=float(
            1
            - np.sum(residual**2)
            / np.sum((np.log10(y) - np.mean(np.log10(y))) ** 2)
        ),
        within_factor_2p5=float(np.mean(abs(residual) <= np.log10(2.5))),
    )


def dump_csv(path, records):
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    rows = [
        {
            "csv_line": i + 2,
            "temperature_K": T[i],
            "ammonia_pptv": NH3[i],
            "acid_1e6_cm3": ACID[i],
            "q_cm3_s": Q[i],
            "rh_pct": RH[i],
            "observed_J": J[i],
            "primary_RH": bool(WINDOW[i]),
            **{k: bool(v[i]) for k, v in SPLITS.items()},
        }
        for i in range(len(T))
    ]
    dump_csv(OUT / "splits.csv", rows)
    summary = {}
    for tab in ["S1", "S1_precise", "S2"]:
        ch = channels(tab)
        assert np.all(ch[2:].sum(axis=0) <= Q + 1e-10)
        assert np.all(ch[2:, Q == 0] == 0)
        for idx, key in enumerate(
            ["binary_neutral", "ternary_neutral", "binary_ion", "ternary_ion"]
        ):
            for i, row in enumerate(rows):
                row[tab + "_" + key] = ch[idx, i]
        for name, mask in {
            "all": T > 0,
            "rh": WINDOW,
            "neutral": WINDOW & (Q == 0),
            "ionized": WINDOW & (Q > 0),
        }.items():
            summary[tab + "_" + name] = scores(J[mask], ch[:, mask].sum(axis=0))
    dump_csv(OUT / "reference.csv", rows)
    (OUT / "reference_metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=1))
