"""Reconstruct a fed pentane drop using local properties and CUDA correlations.

The model combines the 1984 mass correlation (doi:10.1252/jcej.17.7),
its companion blowing correction (doi:10.1252/jcej.17.1), and a
Ranz-Marshall heat baseline. This is not a resolved pendant-drop solver.
All dimensional inputs are SI. No original absolute-flux records are assumed.
"""

import functools
import itertools
import platform
import time
from importlib.metadata import version

import numpy as np
import warp as wp
from chemicals import lennard_jones as lj
from chemicals.viscosity import Wilke
from particula.util.chemical import get_chemical_vapor_pressure
from particula.util.chemical.thermo_import import Chemical
from scipy.constants import Avogadro, Boltzmann, R, Stefan_Boltzmann
from scipy.optimize import brentq
from thermo import Mixture

wp.set_module_options({"enable_backward": True})
TWO = wp.constant(wp.float64(2.0))
C06 = wp.constant(wp.float64(0.6))
TWO_THIRDS = wp.constant(wp.float64(2.0 / 3.0))
C037 = wp.constant(wp.float64(0.37))
C061 = wp.constant(wp.float64(0.61))
C051 = wp.constant(wp.float64(0.51))
ONE = wp.constant(wp.float64(1.0))
C03 = wp.constant(wp.float64(0.3))
C07 = wp.constant(wp.float64(0.7))
C088 = wp.constant(wp.float64(0.88))
ONE_THIRD = wp.constant(wp.float64(1.0 / 3.0))


NAMES = ["nitrogen", "oxygen", "argon", "pentane"]
CHEMS = [Chemical(s) for s in NAMES]
MW = np.array([c.MW / 1000 for c in CHEMS])
DRY = np.array([0.78084, 0.20946, 0.00934])
DRY /= DRY.sum()
MA = float(DRY @ MW[:3])
P = 101325.0
RE = 60.0
DIAM = 0.00161
PHI = 0.080


def collision_data(method):
    """Load local Lennard-Jones diameters and energy scales with sources."""
    rows = []
    for name, c in zip(NAMES, CHEMS, strict=True):
        sm = (
            lj.molecular_diameter_methods(c.CAS)[0]
            if method == "local_default"
            else "Poling et al. (2001)"
        )
        em = (
            lj.Stockmayer_methods(c.CAS)[0]
            if method == "local_default"
            else "Poling et al. (2001)"
        )
        rows.append(
            dict(
                species=name,
                CAS=c.CAS,
                molar_mass_kg_mol=c.MW / 1000,
                sigma_angstrom=lj.molecular_diameter(c.CAS, method=sm),
                epsilon_over_k_K=lj.Stockmayer(c.CAS, method=em),
                sigma_source=sm,
                epsilon_source=em,
            )
        )
    return rows


DATA = {m: collision_data(m) for m in ["local_default", "all_Poling"]}


def phi_reference(re, p):
    """Return the 1984 dimensionless injection ratio from NumPy."""
    b, rho, sc = np.asarray(p).T
    return (
        (2 + 0.37 * re**0.61 * sc**0.51)
        * rho ** (1 / 3)
        / (0.3 + 0.7 * (1 + b) ** 0.88)
        * b
        / (re * sc)
    )


def calculate(t, table, viscosity, collision):
    """Reconstruct surface mixture transport at temperature t in kelvin."""
    data = DATA[table]
    sig = np.array([r["sigma_angstrom"] for r in data]) * 1e-10
    eps = np.array([r["epsilon_over_k_K"] for r in data])
    omega = (
        lj.collision_integral_Neufeld_Janzen_Aziz
        if collision == "Neufeld"
        else lj.collision_integral_Kim_Monroe
    )
    mole = float(get_chemical_vapor_pressure("pentane", t)) / P
    z = np.append(DRY * (1 - mole), mole)
    rho = P * (z @ MW) / (R * t)
    binary = []
    unit_errors = []
    for j in range(3):
        sigma = (sig[j] + sig[3]) / 2
        ek = np.sqrt(eps[j] * eps[3])
        ts = t / ek
        if not (0.3 < ts < 100):
            raise AssertionError(
                "Collision integral outside qualified temperature range"
            )
        om = omega(ts, 1, 1)
        mass = (MW[j] * MW[3] / (MW[j] + MW[3])) / Avogadro
        d = (
            3
            / 16
            * np.sqrt(2 * Boltzmann * t / (np.pi * mass))
            / (P / (Boltzmann * t) * sigma**2 * om)
        )
        # Algebraically independent arrangement used by INL MOOSE, all SI.
        d_check = (
            3
            / 16
            * np.sqrt(2 * np.pi * Boltzmann * t / mass)
            * Boltzmann
            * t
            / (P * np.pi * sigma**2 * om)
        )
        unit_errors.append(abs(d / d_check - 1))
        binary.append(d)
    diffusivity = 1 / np.sum(DRY / np.array(binary))
    if viscosity == "kinetic_Wilke":
        mus = [
            5
            / 16
            * np.sqrt(m * Boltzmann * t / (Avogadro * np.pi))
            / (s * s * omega(t / e, 2, 2))
            for m, s, e in zip(MW, sig, eps, strict=True)
        ]
        mu = Wilke(z.tolist(), mus, (MW * 1000).tolist())
    else:
        mix = Mixture(NAMES, zs=z.tolist(), T=t, P=P)
        mu = mix.ViscosityGasMixture.calculate(t, P, mix.zs, mix.ws, "WILKE")
    sc = mu / (rho * diffusivity)
    b = mole * MW[3] / ((1 - mole) * MA)
    density_ratio = MA / (z @ MW) * t / 326.0
    phi = float(phi_reference(RE, np.array([[b, density_ratio, sc]]))[0])
    return dict(
        Ts_K=t,
        table=table,
        viscosity=viscosity,
        collision=collision,
        D_pentane_N2_m2_s=binary[0],
        D_pentane_O2_m2_s=binary[1],
        D_pentane_Ar_m2_s=binary[2],
        D_effective_m2_s=diffusivity,
        mu_Pa_s=mu,
        rho_kg_m3=rho,
        Sc=sc,
        B=b,
        density_ratio=density_ratio,
        phi_pred=phi,
        printed_over_predicted=PHI / phi,
        flux_from_printed_phi_sphere_proxy_kg_m2_s=PHI * RE * mu / DIAM,
        flux_pred_sphere_proxy_kg_m2_s=phi * RE * mu / DIAM,
        formula_equivalence_error=max(unit_errors),
    )


T0 = 263.6
TINF = 326.0
D64 = wp.constant(wp.float64(DIAM))
TI64 = wp.constant(wp.float64(TINF))
CONFIGS = list(
    itertools.product(
        DATA,
        ["thermo_Wilke", "kinetic_Wilke"],
        ["Neufeld", "Kim_Monroe"],
        ["LINDSAY_BROMLEY", "LINEAR"],
    )
)


@functools.lru_cache(maxsize=None)
def properties(t, config):
    """Build one SI input row at fixed gas speed for a property variant."""
    table, viscosity, collision, conductivity = config
    r = calculate(t, table, viscosity, collision)
    anchor = calculate(T0, table, viscosity, collision)
    speed = 60 * anchor["mu_Pa_s"] / (anchor["rho_kg_m3"] * DIAM)
    # B = y Mv / ((1-y) Ma); recover y from surface mixture density inputs.
    ma = DRY @ MW[:3]
    y = r["B"] * ma / (MW[3] + r["B"] * ma)
    z = np.append(DRY * (1 - y), y)
    mix = Mixture(NAMES, zs=z.tolist(), T=t, P=P)
    k = mix.ThermalConductivityGasMixture.calculate(
        t, P, mix.zs, mix.ws, conductivity
    )
    cp = mix.Cpg
    latent = Chemical("pentane", T=t, P=P).Hvap
    re = r["rho_kg_m3"] * speed * DIAM / r["mu_Pa_s"]
    return np.array(
        [
            re,
            r["Sc"],
            r["B"],
            r["density_ratio"],
            r["rho_kg_m3"],
            r["D_effective_m2_s"],
            k,
            cp,
            latent,
            r["mu_Pa_s"],
            t,
        ]
    )


def reference(p):
    """Evaluate independent NumPy mass and heat fluxes in kg/m2/s."""
    re, sc, b, ratio, rho, diffusion, k, cp, latent, mu, t = np.asarray(p).T
    base = (2 + 0.37 * re**0.61 * sc**0.51) * ratio ** (1 / 3)
    scale = rho * diffusion * b / DIAM * base
    jm = scale / (0.3 + 0.7 * (1 + b) ** 0.88)
    jf = scale * np.log1p(b) / b
    nu = 2 + 0.6 * np.sqrt(re) * (mu * cp / k) ** (1 / 3)
    jh = k * nu * (TINF - t) / (DIAM * latent)
    bh = cp * (TINF - t) / latent
    jhp = jh / (0.3 + 0.7 * (1 + bh) ** 0.88)
    jhf = jh * np.log1p(bh) / bh
    return np.stack([jm, jf, jh, jhp, jhf, jm - jhp], axis=-1)


@wp.kernel
def balances(
    p: wp.array2d(dtype=wp.float64), out: wp.array2d(dtype=wp.float64)
):
    """Evaluate mass and heat fluxes for independent float64 CUDA rows."""
    i = wp.tid()
    re, sc, b, ratio = p[i, 0], p[i, 1], p[i, 2], p[i, 3]
    rho, diffusion, k, cp = p[i, 4], p[i, 5], p[i, 6], p[i, 7]
    latent, mu, t = p[i, 8], p[i, 9], p[i, 10]
    base = (TWO + C037 * wp.pow(re, C061) * wp.pow(sc, C051)) * wp.pow(
        ratio, ONE_THIRD
    )
    scale = rho * diffusion * b / D64 * base
    jm = scale / (C03 + C07 * wp.pow(ONE + b, C088))
    jf = scale * wp.log(ONE + b) / b
    nu = TWO + C06 * wp.sqrt(re) * wp.pow(mu * cp / k, ONE_THIRD)
    jh = k * nu * (TI64 - t) / (D64 * latent)
    bh = cp * (TI64 - t) / latent
    jhp = jh / (C03 + C07 * wp.pow(ONE + bh, C088))
    jhf = jh * wp.log(ONE + bh) / bh
    out[i, 0] = jm
    out[i, 1] = jf
    out[i, 2] = jh
    out[i, 3] = jhp
    out[i, 4] = jhf
    out[i, 5] = jm - jhp


def evaluate(p, gradient=False):
    """Run CUDA correlations and optionally return residual derivatives."""
    pp = wp.array(p, dtype=wp.float64, device="cuda:0", requires_grad=gradient)
    out = wp.empty(
        (len(p), 6), dtype=wp.float64, device="cuda:0", requires_grad=gradient
    )
    if gradient:
        with wp.Tape() as tape:
            wp.launch(
                balances, len(p), inputs=[pp], outputs=[out], device="cuda:0"
            )
        seed = np.zeros((len(p), 6))
        seed[:, 5] = 1
        tape.backward(
            grads={out: wp.array(seed, dtype=wp.float64, device="cuda:0")}
        )
        return out.numpy(), pp.grad.numpy()
    wp.launch(balances, len(p), inputs=[pp], outputs=[out], device="cuda:0")
    return out.numpy()


def printed_pr(p):
    """Use reported Pr=0.86 in heat transfer, preserving mass transport."""
    result = reference(p).copy()
    re, k, cp, mu = p[0], p[6], p[7], p[9]
    local_nu = 2 + 0.6 * np.sqrt(re) * (mu * cp / k) ** (1 / 3)
    printed_nu = 2 + 0.6 * np.sqrt(re) * 0.86 ** (1 / 3)
    result[2:5] *= printed_nu / local_nu
    result[5] = result[0] - result[3]
    return result


def sensitivity(configs):
    """Solve the reported-Prandtl input sensitivity on CPU."""
    rows = []
    for ci, config in enumerate(configs):
        obs = properties(T0, config)
        v = printed_pr(obs)
        jprinted = 0.08 * 60 * obs[9] / 0.00161
        for mi, mass in enumerate(["paper", "film"]):
            for hi, heat in enumerate(["no_blow", "paper", "film"], start=2):

                def balance(t, config=config, mi=mi, hi=hi):
                    vals = printed_pr(properties(float(t), config))
                    return vals[mi] - vals[hi]

                root = brentq(balance, 250, 290, xtol=1e-10)
                vals = printed_pr(properties(root, config))
                if not (abs(vals[mi] / vals[hi] - 1) < 1e-9):
                    raise AssertionError(
                        "Reported-Pr heat and mass residual exceeded tolerance"
                    )
                rows.append(
                    dict(
                        config=ci,
                        mass=mass,
                        heat=heat,
                        Ts_K=root,
                        flux_kg_m2_s=vals[mi],
                        q0_at_printed_Ts_W_m2=v[2] * obs[8],
                        extra_for_printed_flux_W_m2=(jprinted - v[2]) * obs[8],
                    )
                )

    return rows


def simulate(pilot=False):  # noqa: C901 -- ordered scientific validation campaign
    """Run all 16 property variants (one in pilot) on actual CUDA."""
    properties.cache_clear()
    started = time.perf_counter()
    configs = CONFIGS[:1] if pilot else CONFIGS
    temps = np.linspace(250, 290, 17 if pilot else 321)
    wp.init()
    device = wp.get_device("cuda:0")
    if not (device.is_cuda):
        raise AssertionError("CUDA device required")
    p = np.array([properties(float(t), c) for c in configs for t in temps])
    cpu = reference(p)
    gpu = evaluate(p)
    # Scale residual separately near intersections; other fluxes are positive.
    parity = float(np.max(abs(gpu[:, :5] / cpu[:, :5] - 1)))
    if not (parity < 1e-10):
        raise AssertionError("NumPy/CUDA parity exceeded tolerance")
    if not (np.all(np.isfinite(gpu)) and np.all(gpu[:, :5] > 0)):
        raise AssertionError("Nonfinite or nonpositive model flux")
    curves = gpu.reshape(len(configs), len(temps), 6)
    roots, refinement, observed = [], [], []
    root_inputs = []
    for ci, c in enumerate(configs):
        atobs = properties(T0, c)
        vals = reference(atobs)
        rho, mu, latent = atobs[4], atobs[9], atobs[8]
        speed = 60 * mu / (rho * DIAM)
        jprinted = 0.080 * rho * speed
        qconv = vals[2] * latent
        observed.append(
            dict(
                config=ci,
                table=c[0],
                viscosity=c[1],
                collision=c[2],
                conductivity=c[3],
                speed_m_s=speed,
                Pr=mu * atobs[7] / atobs[6],
                k_W_m_K=atobs[6],
                cp_J_kg_K=atobs[7],
                latent_J_kg=latent,
                mass_paper_kg_m2_s=vals[0],
                mass_film_kg_m2_s=vals[1],
                heat_no_blow_kg_m2_s=vals[2],
                heat_paper_kg_m2_s=vals[3],
                heat_film_kg_m2_s=vals[4],
                printed_proxy_kg_m2_s=jprinted,
                q_convection_no_blow_W_m2=qconv,
                extra_for_mass_paper_W_m2=vals[0] * latent - qconv,
                extra_for_printed_proxy_W_m2=jprinted * latent - qconv,
            )
        )
        for mi, mass in enumerate(["paper", "film"]):
            for hi, heat in enumerate(["no_blow", "paper", "film"], start=2):

                def balance(t, c=c, mi=mi, hi=hi):
                    v = reference(properties(float(t), c))
                    return v[mi] - v[hi]

                root = brentq(balance, 250, 290, xtol=1e-10)
                rp = properties(root, c)
                v = reference(rp)
                roots.append(
                    dict(
                        config=ci,
                        mass=mass,
                        heat=heat,
                        Ts_K=root,
                        flux_kg_m2_s=v[mi],
                        Re=rp[0],
                        relative_balance_residual=abs(v[mi] - v[hi]) / v[mi],
                    )
                )
                if mass == "paper" and heat == "paper":
                    root_inputs.append(rp)
                if not pilot:
                    for step in [4, 2, 1]:
                        vv = curves[ci, ::step, mi] - curves[ci, ::step, hi]
                        tt = temps[::step]
                        ix = np.flatnonzero(vv[:-1] * vv[1:] < 0)
                        if not (len(ix) == 1):
                            raise AssertionError(
                                "Expected exactly one heat-mass intersection"
                            )
                        ix = ix[0]
                        estimate = tt[ix] - vv[ix] * (tt[ix + 1] - tt[ix]) / (
                            vv[ix + 1] - vv[ix]
                        )
                        refinement.append(
                            dict(
                                config=ci,
                                mass=mass,
                                heat=heat,
                                grid_points=len(tt),
                                error_K=abs(estimate - root),
                            )
                        )
    root_inputs = np.array(root_inputs)
    root_values, gradients = evaluate(root_inputs, gradient=True)
    gradient_errors = []
    slopes = []
    for h in [0.01, 0.001, 0.0001]:
        for ci, c in enumerate(configs):
            t = root_inputs[ci, -1]
            plus, minus = properties(t + h, c), properties(t - h, c)
            chain = gradients[ci] @ ((plus - minus) / (2 * h))
            direct = (reference(plus)[5] - reference(minus)[5]) / (2 * h)
            gradient_errors.append(abs(chain / direct - 1))
            slopes.append(
                dict(
                    config=ci,
                    step_K=h,
                    CUDA_chain_kg_m2_s_K=chain,
                    CPU_finite_difference_kg_m2_s_K=direct,
                )
            )
    if not (max(gradient_errors) < 1e-6):
        raise AssertionError(
            "Temperature derivative disagreement exceeded tolerance"
        )
    if not (max(r["relative_balance_residual"] for r in roots) < 1e-9):
        raise AssertionError("Root residual exceeded tolerance")
    if not (np.max(abs(root_values[:, 5]) / root_values[:, 0]) < 1e-9):
        raise AssertionError("CUDA equilibrium residual exceeded tolerance")
    if refinement:
        if not (
            max(r["error_K"] for r in refinement if r["grid_points"] == 321)
            < 0.001
        ):
            raise AssertionError(
                "Temperature-grid refinement exceeded tolerance"
            )
    root_ranges = {}
    for mass, heat in itertools.product(
        ["paper", "film"], ["no_blow", "paper", "film"]
    ):
        rr = [r for r in roots if r["mass"] == mass and r["heat"] == heat]
        root_ranges[mass + "_mass_" + heat + "_heat"] = dict(
            temperature_K=[
                min(r["Ts_K"] for r in rr),
                max(r["Ts_K"] for r in rr),
            ],
            flux_kg_m2_s=[
                min(r["flux_kg_m2_s"] for r in rr),
                max(r["flux_kg_m2_s"] for r in rr),
            ],
        )
    summary = dict(
        device=device.name,
        CUDA_curve_points=len(p),
        CUDA_root_points=len(root_inputs),
        reverse_launches=1,
        cpu_cuda_relative_error=parity,
        gradient_max_relative_error=max(gradient_errors),
        grid_max_temperature_error_K={
            str(n): max(
                r["error_K"] for r in refinement if r["grid_points"] == n
            )
            for n in [81, 161, 321]
        }
        if refinement
        else {},
        root_ranges=root_ranges,
        radiation_blackbody_at_326K_W_m2=Stefan_Boltzmann * (TINF**4 - T0**4),
        observed_ranges={
            k: [min(r[k] for r in observed), max(r[k] for r in observed)]
            for k in observed[0]
            if k
            not in ["config", "table", "viscosity", "collision", "conductivity"]
        },
        configurations=configs,
        versions={
            n: version(n)
            for n in [
                "thermo",
                "chemicals",
                "warp-lang",
                "numpy",
                "scipy",
                "matplotlib",
            ]
        },
        python=platform.python_version(),
        particula_source_version=__import__("particula").__version__,
        property_sources="Local installed thermo and chemicals only",
        property_methods=dict(
            vapor_pressure="HEOS_FIT",
            latent_heat="HEOS_FIT",
            gas_heat_capacity="LINEAR mixture, pure HEOS_FIT",
            conductivity=list({c[3] for c in configs}),
        ),
        fixed=dict(
            Tinf_K=TINF,
            pressure_Pa=P,
            diameter_m=DIAM,
            anchor_Ts_K=T0,
            anchor_Re=60,
            printed_phi=0.080,
        ),
        wall_seconds=time.perf_counter() - started,
    )
    pr_rows = sensitivity(configs)
    summary["total_wall_seconds"] = time.perf_counter() - started
    return dict(
        temperatures_K=temps.tolist(),
        curves=curves.tolist(),
        summary=summary,
        intersections=roots,
        observed=observed,
        printed_pr=pr_rows,
        refinement=refinement,
        gradient_checks=slopes,
        collision_data=DATA,
    )
