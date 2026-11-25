"""Run tagged, weighted cloud particles in independent CUDA parcels."""

import argparse
import csv
import ctypes
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.special import ndtr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from equilibrium import state  # noqa: E402
from parcel_study import forcing, parcel  # noqa: E402

# Number/cm³, median dry diameter nm, geometric sigma, kappa, absorbing
# dry mass fraction. A background mode's size support spans 10 to 1000 nm.
REGIMES = {
    "clean": [],
    "benchmark": [(1000, 100, 1.5, 0.3, 0.5)],
    "marine": [(180, 45, 1.5, 0.6, 0.1), (120, 150, 1.5, 0.6, 0.1)],
    "smoke": [(1500, 150, 1.65, 0.12, 0.8)],
    "urban": [(6000, 45, 1.6, 0.20, 0.4), (1000, 150, 1.6, 0.30, 0.4)],
    "soa": [(1160, 44, 1.5, 0.10, 0.8), (830, 154, 1.5, 0.15, 0.8)],
}
RESULTS = HERE / "results"
VOL = 4 * np.pi / 3
V0 = (
    (
        parcel.RD
        + parcel.RV
        * (parcel.RD / parcel.RV)
        * parcel._saturation_pressure(285.0)
        / (90000.0 - parcel._saturation_pressure(285.0))
    )
    * 285.0
    / 90000.0
)


@dataclass(frozen=True)
class Case:
    """Specify aerosol populations, organic uptake and parcel forcing."""

    regime: str = "clean"
    w: float = 0.2
    entr: float = 0.0
    gas: float = 0.0
    scenario: str = "central"
    mix: str = "mixed"
    seed: int = -1
    import_aerosol: bool = True
    rh_ext: float = 0.90
    access: float = 1.0
    cstar: float = 10.0
    nu: float = 1.0
    accommodation: float = 0.5
    gas_ext: float = 1.0
    bg_scale: float = 1.0
    target_total: float = 300.0
    environment: str = "same"
    temp_anomaly: float = 0.0
    target_dg: float = 100.0
    target_gsd: float = 1.5


def build_cuda():
    """Compile and load the device solver for the selected CUDA architecture."""
    src = HERE / "cloud.cu"
    arch = os.environ.get("PARTICULA_CLOUD_CUDA_ARCH", "sm_120")
    digest = hashlib.sha256(
        src.read_bytes() + (HERE / "organic.cuh").read_bytes() + arch.encode()
    ).hexdigest()[:16]
    cache = Path(tempfile.gettempdir()) / f"particula-cloud-{os.getuid()}"
    cache.mkdir(mode=0o700, exist_ok=True)
    lib = cache / f"{digest}.so"
    if not lib.exists():
        # Publish only a complete binary when two studies start together.
        with tempfile.TemporaryDirectory(dir=cache) as directory:
            compiled = Path(directory) / "cloud.so"
            subprocess.run(  # noqa: S603
                [  # noqa: S607
                    "nvcc",
                    "-O3",
                    "-shared",
                    "-Xcompiler",
                    "-fPIC",
                    f"-arch={arch}",
                    str(src),
                    "-o",
                    str(compiled),
                ],
                check=True,
            )
            compiled.replace(lib)
    func = ctypes.CDLL(str(lib)).cloud_run
    func.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    ] * 7 + [ctypes.c_int] * 5
    func.restype = ctypes.c_int
    return func


@dataclass
class Mode:
    """Describe a number-weighted, truncated lognormal population."""

    label: str
    recipe: str
    count: float
    dg: float
    sd: float
    kappa: float
    absorb: float
    limits: tuple
    entrant: bool = False

    def cdf(self, d):
        """Return the fraction of the truncated size distribution below d."""

        def n(x):
            return ndtr(np.log(x / self.dg) / np.log(self.sd))

        a, b = self.limits
        return (n(d) - n(a)) / (n(b) - n(a))


def modes(case):
    """Construct target, resident and entrained dry populations."""
    out = []
    recipes = {
        "mixed": ["octyl", "dodecyl"],
        "acid": ["acid"],
        "octyl": ["octyl"],
        "dodecyl": ["dodecyl"],
        "none": [],
    }[case.mix]
    for r in recipes:
        out.append(
            Mode(
                r,
                r,
                case.target_total / len(recipes),
                case.target_dg,
                case.target_gsd,
                0.0,
                1.0,
                (40, 400),
            )
        )
    for entrant in [False, True]:
        if entrant and (not case.entr or not case.import_aerosol):
            continue
        source = (
            case.environment
            if entrant and case.environment != "same"
            else case.regime
        )
        for j, (count, dg, sd, k, a) in enumerate(REGIMES[source]):
            limits = (40, 400) if source == "benchmark" else (10, 1000)
            out.append(
                Mode(
                    ("incoming" if entrant else "resident") + str(j),
                    "bg",
                    count * case.bg_scale,
                    dg,
                    sd,
                    k,
                    a,
                    limits,
                    entrant,
                )
            )
    return out


def group(mode, case, bins, nodes, brackets):
    """Build thermodynamic tables and weighted sizes for a single mode."""
    edges = np.geomspace(*mode.limits, bins + 1)
    d = np.sqrt(edges[1:] * edges[:-1])
    # Endpoints and local refinement particles sample the coupled history
    # without changing any physical inventory.
    probes = list(mode.limits)
    for lo, hi in brackets:
        probes.extend(np.geomspace(lo, hi, 25))
    diam = np.r_[d, probes]
    num = np.r_[
        np.diff(mode.cdf(edges)) * mode.count, np.zeros(len(diam) - bins)
    ]
    a = diam[:, None] * 0.5e-9
    x = np.linspace(0, np.log(1500.0), nodes)
    wet = a * np.exp(np.maximum(x, 1e-9))
    if mode.recipe == "bg":
        q = np.expm1(3 * x)
        aw = np.broadcast_to(q / (q + mode.kappa), wet.shape)
        sig = np.full_like(wet, parcel.SIGMA)
    else:
        model = "bulk" if case.scenario == "direct" else "finite"
        law = (
            "central"
            if case.scenario in ("direct", "fixed70")
            else case.scenario
        )
        aw, sig, *_ = state(a, wet, mode.recipe, model, law)
        if case.scenario == "fixed70":
            sig = np.full_like(
                wet, {"octyl": 41.8, "dodecyl": 36.5, "acid": 64.4}[mode.recipe]
            )
        sig = sig * 1e-3
    table = np.stack([aw, sig], axis=-1)
    table[:, 0, 0] = 0.0
    eq = table[:, :, 0] * np.exp(2 * sig / (parcel.RV * 1000 * 285 * wet))
    # Entrained aerosol is separately equilibrated at ambient T,p in the kernel.
    above = eq >= 1
    if not above.any(axis=1).all():
        raise ValueError("Missing saturated haze branch")
    idx = above.argmax(axis=1)
    r = np.arange(len(diam))
    below = eq[r, idx - 1]
    init = a[:, 0] * np.exp(
        x[idx - 1] + (x[1] - x[0]) * (1 - below) / (eq[r, idx] - below)
    )
    nabs = (
        VOL
        * a[:, 0] ** 3
        * (1500 if mode.recipe == "bg" else 1665)
        * mode.absorb
        / (0.15 if mode.recipe == "bg" else 0.192124)
    )
    seed = np.column_stack(
        [
            a[:, 0],
            num * 1e6 * (1 if mode.entrant else V0),
            init,
            np.full(len(diam), mode.entrant),
            nabs,
            np.zeros(len(diam)),
        ]
    )
    return seed, table, diam


def prepare(case, bins, nodes, dt, brackets):
    """Initialize populations and the wind history for one parcel."""
    ms = modes(case)
    gr = [group(m, case, bins, nodes, brackets.get(m.label, [])) for m in ms]
    steps = round(100 / case.w / dt)
    times, speeds, _ = forcing(
        case.w, 0.8 if case.seed >= 0 else 0.0, 30, max(case.seed, 0)
    )
    wind = np.interp((np.arange(steps) + 0.5) * dt, times, speeds)
    # Discrete forcing has exactly 100 m net displacement on every resolution.
    wind *= 100 / (wind.sum() * dt)
    params = np.zeros(16)
    params[:11] = [
        dt,
        case.entr,
        case.rh_ext,
        case.gas,
        case.access,
        case.cstar,
        case.nu,
        case.temp_anomaly,
        case.gas_ext,
        0,
        case.accommodation,
    ]
    return ms, gr, params, wind


def call_gpu(func, batch, nodes, steps, nh=100):
    """Advance one batch sharing a step count and carrier dimensions."""

    def join(j):
        return np.ascontiguousarray(
            np.stack([np.concatenate([g[j] for g in r[1]]) for r in batch])
        )

    data, tables = join(0), join(1)
    b, n = data.shape[:2]
    if n > 1024:
        raise ValueError(f"{n} carriers exceed block size")
    if not (steps % nh == 0):
        raise ValueError(
            "Requested output interval does not divide the time grid"
        )
    out = np.zeros((b, 12))
    particles = np.zeros((b, n, 5))
    path = np.zeros((b, nh, 6))
    rc = func(
        np.stack([r[2] for r in batch]),
        data,
        tables,
        np.stack([r[3] for r in batch]),
        out,
        particles,
        path,
        b,
        n,
        nodes,
        steps,
        nh,
    )
    if rc:
        raise RuntimeError(f"CUDA error {rc}")
    if not all(np.all(np.isfinite(v)) for v in [out, particles, path]):
        raise ValueError("Nonfinite CUDA output")
    return out, particles, path


def unpack(case, prep, air, particles, bins, expected_height=100.0):
    """Recover size cutoffs, population numbers and parcel closure."""
    result = dict(
        peak_pct=air[0],
        water_err_kgkg=air[1],
        organic_err_kgkg=air[2],
        organic_gas_ugkg=air[3] * 1e9,
        organic_total_ugkg=air[4] * 1e9,
        event_height_m=air[5],
        height_m=air[6],
        final_p=air[7],
        final_T=air[8],
        final_q=air[9],
        heat_err_Jkg=air[10],
        rh_solver_err=air[11],
    )
    bounds = {}
    rows = []
    offset = 0
    for mode, g in zip(prep[0], prep[1], strict=True):
        diam = g[2]
        part = particles[offset : offset + len(diam)]
        offset += len(diam)
        order = np.argsort(diam)
        d = diam[order]
        flags = part[order, 1] > 0
        idx = np.where(flags[1:] != flags[:-1])[0]
        bounds[mode.label] = [(float(d[j]), float(d[j + 1])) for j in idx]
        # Integrate all successful intervals (scavenged giant haze particles
        # and recently entrained populations need not have a single cutoff).
        cuts = np.r_[d[0], np.sqrt(d[idx] * d[idx + 1]), d[-1]]
        status = np.r_[flags[0], flags[idx + 1]]
        frac = np.sum(np.diff(mode.cdf(cuts))[status])
        width = sum(mode.cdf(d[j + 1]) - mode.cdf(d[j]) for j in idx)
        total = part[:, 0].sum()
        rows.append(
            dict(
                mode=mode.label,
                available_cm3=total,
                activated_cm3=frac * total,
                fraction=frac,
                uncertainty_width_cm3=width * total,
                discrete_cm3=np.sum(part[:, 0] * (part[:, 1] > 0)),
                organic_ugkg=1e9 * np.sum(part[:, 3] * part[:, 4]),
                cutoffs_nm="|".join(f"{v:.8f}" for v in cuts[1:-1]),
            )
        )
    result["activated_cm3"] = sum(r["activated_cm3"] for r in rows)
    comp = {r["mode"]: r for r in rows}
    result["dodecyl_minus_octyl_cm3"] = (
        comp["dodecyl"]["activated_cm3"] - comp["octyl"]["activated_cm3"]
        if case.mix == "mixed"
        else ""
    )
    result["water_ok"] = bool(air[1] < 1e-11)
    recovered = result["organic_gas_ugkg"] + sum(
        r["organic_ugkg"] for r in rows
    )
    if (
        abs(recovered - result["organic_total_ugkg"]) > 1e-6
        or air[1] >= 1e-11
        or abs(air[6] - expected_height) > 1e-7
        or np.min(particles[:, 3]) < 0
    ):
        raise ValueError((case, result, recovered))
    return result, rows, bounds


def write(path, rows):
    """Save records with their field names."""
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0], lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def design(stage):  # noqa: C901 - each branch defines an experiment
    """Construct the scenario matrix for the selected experiment."""
    if stage == "pilot":
        return [Case(), Case(w=1), Case(regime="urban", w=1, gas=1, entr=0.1)]
    if stage == "reference":
        return [
            Case(regime=r, w=w, mix=m)
            for r in ("clean", "benchmark")
            for w in (0.2, 1.0)
            for m in ("mixed", "acid", "octyl", "dodecyl")
        ]
    if stage == "factorial":
        return [
            Case(regime=r, w=w, entr=e, gas=g)
            for r in REGIMES
            if r != "benchmark"
            for w in (0.2, 1.0)
            for e in (0.0, 0.05, 0.2)
            for g in (0.0, 1.0)
        ]
    if stage == "attribution":
        return [
            Case(regime=r, w=w, entr=0.05, gas=g, mix=m)
            for r in ("marine", "urban", "smoke", "soa")
            for w in (0.2, 1.0)
            for g in (0.0, 1.0)
            for m in ("none", "acid", "octyl", "dodecyl")
        ]
    if stage == "forcing":
        return [
            Case(regime=r, w=0.2, entr=0.05, gas=1.0, seed=s, scenario=a)
            for r in ("marine", "urban", "smoke", "soa")
            for s in range(6)
            for a in ("low", "central", "high")
        ]
    if stage == "organic_limits":
        # Include the matched gas-free baseline from the factorial suite.
        # C* = 10 µg/m³ for both organic surrogates.
        return [
            Case(regime=region, w=w, entr=0.05, gas=gas, access=access, nu=nu)
            for region in ("marine", "urban", "smoke", "soa")
            for w in (0.2, 1.0)
            for gas, access, nu in (
                (0.0, 1.0, 1.0),
                (1.0, 1.0, 1.0),
                (1.0, 0.001, 0.1),
            )
        ]
    if stage == "spectrum_refined":
        return [
            Case(
                regime=r,
                w=0.2,
                entr=0.05,
                gas=1.0,
                target_dg=dg,
                target_gsd=width,
                scenario=sc,
            )
            for r, dg in (("marine", 130.0), ("urban", 220.0))
            for width in (1.08, 1.5)
            for sc in ("central", "low", "high")
        ]
    if stage == "spectrum":
        return [
            Case(
                regime=r,
                w=0.2,
                entr=0.05,
                gas=1.0,
                target_dg=dg,
                target_gsd=width,
                scenario=sc,
            )
            for r in ("marine", "urban")
            for dg in (75.0, 100.0, 130.0, 170.0, 220.0)
            for width in (1.08, 1.25, 1.5)
            for sc in ("central", "low", "high")
        ]
    if stage == "tension":
        return [
            Case(regime=r, w=w, entr=0.05, gas=1.0, scenario=sc)
            for r in ("clean", "marine", "smoke", "urban", "soa")
            for w in (0.2, 1.0)
            for sc in (
                "fixed70",
                "direct",
                "low",
                "high",
                "response-low",
                "response-high",
            )
        ]
    if stage == "identity":
        return [
            Case(regime=r, w=0.2, entr=0.05, gas=1.0, scenario=sc)
            for r in ("urban", "marine")
            for sc in ("central", "fixed70", "direct", "low", "high")
        ]
    if stage == "loading":
        return [
            Case(regime=r, w=w, entr=0.05, gas=1.0, target_total=cnt, mix=m)
            for r in ["marine", "smoke", "urban", "soa"]
            for w in (0.2, 1.0)
            for cnt in (3.0,)
            for m in ["mixed", "acid"]
        ]
    raise ValueError(stage)


def main():
    """Run the selected cases and save endpoints, populations and paths."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "stage",
        choices=[
            "pilot",
            "reference",
            "factorial",
            "attribution",
            "forcing",
            "loading",
            "organic_limits",
            "tension",
            "spectrum",
            "identity",
            "spectrum_refined",
        ],
    )
    ap.add_argument("--dt", type=float, default=0.25)
    ap.add_argument("--bins", type=int, default=48)
    ap.add_argument("--nodes", type=int, default=1025)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--out", type=Path, default=RESULTS)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    args.out.mkdir(exist_ok=True, parents=True)
    func = build_cuda()
    all_results = []
    pop = []
    paths = []
    begin = time.monotonic()
    cases = design(args.stage)[args.offset :][: args.limit]
    bounds = [{} for c in cases]
    for it in range(args.rounds):
        # Limit peak memory while batching compatible parcels on the GPU.
        buckets = {}
        for k, c in enumerate(cases):
            ms = modes(c)
            sig = (
                c.w,
                tuple((m.label, len(bounds[k].get(m.label, ()))) for m in ms),
            )
            buckets.setdefault(sig, []).append(k)
        for keys in buckets.values():
            for start in range(0, len(keys), 6):
                kk = keys[start : start + 6]
                batch = [
                    prepare(cases[k], args.bins, args.nodes, args.dt, bounds[k])
                    for k in kk
                ]
                o, par, hs = call_gpu(func, batch, args.nodes, len(batch[0][3]))
                for j, k in enumerate(kk):
                    c = cases[k]
                    res, rows, b = unpack(c, batch[j], o[j], par[j], args.bins)
                    # The new intervals bracket each transition on this pass.
                    bounds[k] = {
                        label: intervals for label, intervals in b.items()
                    }
                    if it == args.rounds - 1:
                        keys0 = dict(
                            case_id=k + args.offset,
                            **asdict(c),
                            bins=args.bins,
                            nodes=args.nodes,
                            dt=args.dt,
                        )
                        all_results.append({**keys0, **res})
                        pop.extend([{**keys0, **row} for row in rows])
                        paths.extend(
                            [
                                dict(
                                    case_id=k + args.offset,
                                    height_m=h[0],
                                    supersaturation_pct=h[1] * 100,
                                    temperature_K=h[2],
                                    water_kgkg=h[3],
                                    organic_gas_kgkg=h[4],
                                    pressure_Pa=h[5],
                                )
                                for h in hs[j]
                            ]
                        )
                print(
                    f"{args.stage} pass {it + 1}, cases {kk[0]}..{kk[-1]}: "
                    f"{time.monotonic() - begin:.1f} s",
                    flush=True,
                )
    prefix = args.out / ("gpu_" + args.stage + args.tag)
    write(
        prefix.with_suffix(".csv"),
        sorted(all_results, key=lambda r: r["case_id"]),
    )
    write(Path(str(prefix) + "_populations.csv"), pop)
    import gzip

    with gzip.open(str(prefix) + "_paths.csv.gz", "wt", newline="") as f:
        w = csv.DictWriter(f, fieldnames=paths[0], lineterminator="\n")
        w.writeheader()
        w.writerows(paths)
    metadata = dict(
        cases=len(cases),
        rounds=args.rounds,
        seconds=time.monotonic() - begin,
        device=subprocess.check_output(
            [  # noqa: S607
                "nvidia-smi",
                "--query-gpu=name,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip(),
        source_hash=hashlib.sha256(
            (HERE / "cloud.cu").read_bytes()
        ).hexdigest(),
    )
    Path(str(prefix) + ".json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
