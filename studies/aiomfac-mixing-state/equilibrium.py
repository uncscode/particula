"""Independent CPU cloud-point and mineral-saturation controls for the RH ramp."""

import argparse
import json
from hashlib import sha256
from pathlib import Path

import numpy as np
from particula.util.constants import GAS_CONSTANT
from particula_aiomfac import AIOMFACModel, MixtureSpec
from particula_aiomfac.phases import component_molar_masses
from scipy.optimize import least_squares
from scipy.special import softmax

RECIPE = {
    "water": {16: 1},
    "2-propanol": {141: 2, 151: 1, 153: 1},
    "NaCl": {202: 1, 242: 1},
    "KCl": {203: 1, 242: 1},
}
TEMPERATURE = 298.15
SURFACE_TENSION = 0.072
DENSITY = np.array([1000.0, 785.0, 2165.0, 1984.0])
WATER_MW = 0.01801528
LOG10_KSP = np.array([1.57, 0.9])


def _cloud_point(model, dry_fraction, guess=None):
    """Find a nontrivial incipient beta at fixed alpha dry composition.

    Both molecular activities and the two independently transferable
    electrolyte activity products agree. The variable alpha water mass is
    per unit dry mass. Three independent beta mass ratios span its simplex.
    """

    def phases(v):
        return np.stack(
            [np.r_[np.exp(v[0]), dry_fraction], softmax(np.r_[0.0, v[1:]])]
        )

    def residual(v):
        state = model.evaluate(phases(v), TEMPERATURE)
        mus = np.log(
            np.column_stack(
                (state.neutral_activities, state.ion_activity_products)
            )
        )
        return mus[0] - mus[1]

    def run(v):
        root = least_squares(
            residual, v, bounds=(-28, 28), xtol=5e-12, ftol=1e-12, gtol=1e-12
        )
        m = phases(root.x)
        mass_frac = m / m.sum(axis=1, keepdims=True)
        # The formally exact homogeneous root exists for every water amount.
        # Its beta composition cannot certify the onset of a distinct liquid.
        if (
            np.max(abs(root.fun)) < 1e-8
            and mass_frac[0, 0] - mass_frac[1, 0] > 0.01
        ):
            return root.x, mass_frac, float(np.max(abs(root.fun)))
        return None

    if guess is not None:
        candidate = run(guess)
        if candidate is not None:
            return candidate
    seed = model.phase_equilibrium(np.r_[1.0, dry_fraction], TEMPERATURE, 0)
    if seed.beta_masses is None:
        raise ValueError("no two-liquid seed on the tested dry-composition ray")
    beta = seed.beta_masses
    guess = np.r_[np.log(0.68 / 0.32), np.log(beta[1:] / beta[0])]
    candidate = run(guess)
    if candidate is None:
        raise ValueError("no distinct, verified cloud-point solution")
    return candidate


def _crossings(counts, particle_id):
    appearing = (counts[:-1] == 1) & (counts[1:] == 2)
    merging = (counts[:-1] == 2) & (counts[1:] == 1)

    def ids(mask):
        selected = np.asarray(
            np.sort(particle_id[np.any(mask, axis=0)]), dtype="<i8"
        )
        return {
            "count": len(selected),
            "sha256_sorted_little_endian_int64": sha256(
                selected.tobytes()
            ).hexdigest(),
        }

    return {
        "sampled_appearance_events": int(appearing.sum()),
        "unique_particles_with_appearance": ids(appearing),
        "sampled_disappearance_events": int(merging.sum()),
        "unique_particles_with_disappearance": ids(merging),
        "sampled_two_liquid_counts": np.sum(counts == 2, axis=1).tolist(),
    }


def _gradient_residual(model, composition, temperature, relative_step=1e-5):
    """Check the Gibbs differential in neutral molecule and salt formula units."""
    mw = component_molar_masses(model)
    n0 = composition / mw

    def chemical_potentials(n):
        result = model.evaluate(n * mw, temperature)
        return np.log(
            np.r_[result.neutral_activities, result.ion_activity_products]
        )

    grad = np.empty_like(n0)
    for k, step in enumerate(n0 * relative_step):
        shift = np.zeros_like(n0)
        shift[k] = step
        plus, minus = n0 + shift, n0 - shift
        grad[k] = (
            plus @ chemical_potentials(plus)
            - minus @ chemical_potentials(minus)
        ) / (2 * step)
    return grad - chemical_potentials(n0)


def mixed_solvent_screen():
    """Reproduce the activity-gradient check that informed ramp design."""
    recipe = {
        "water": {16: 1},
        "2-propanol": {141: 2, 151: 1, 153: 1},
        "1,4-butanediol": {142: 2, 150: 2, 153: 2},
        "ammonium sulfate": {204: 2, 261: 1},
    }
    model = AIOMFACModel.prepare(MixtureSpec(recipe))
    mass = np.array([0.55, 0.342, 0.009, 0.099])
    temperature = 303.15
    return {
        "material_order": list(recipe),
        "mass_fractions": mass.tolist(),
        "temperature_k": temperature,
        "gibbs_gradient_minus_activity_potential_rt": {
            str(step): _gradient_residual(
                model, mass, temperature, step
            ).tolist()
            for step in [1e-4, 1e-5, 1e-6]
        },
    }


def analyze(archive):
    """Compare the sampled resident population with CPU cloud points."""
    particles = np.asarray(archive["particle_id"])
    dry = np.asarray(archive["dry_mass_kg"])
    water = np.asarray(archive["water_kg"])
    observed = np.asarray(archive["phase_count"])
    rh = np.asarray(archive["rh"])
    if (
        archive["materials"] != list(RECIPE)
        or archive["temperature_K"] != TEMPERATURE
        or archive["surface_tension_N_m"] != SURFACE_TENSION
    ):
        raise ValueError("ramp mixture, temperature or surface tension changed")
    if (
        not np.array_equal(archive["material_density_kg_m3"], DENSITY)
        or archive["kelvin_density_kg_m3"] != DENSITY[0]
    ):
        raise ValueError("ramp volume or curvature density changed")
    if (
        dry.shape != (len(particles), 3)
        or water.shape != observed.shape
        or water.shape != (len(rh), len(particles))
    ):
        raise ValueError(
            "ramp axes must be sample, particle and three dry species"
        )
    if len(set(particles)) != len(particles) or len(
        np.unique(dry, axis=0)
    ) != len(particles):
        raise ValueError("duplicate particle ID or dry inventory")
    model = AIOMFACModel.prepare(MixtureSpec(RECIPE))
    dry_mass = dry.sum(axis=1)
    dry_fraction = dry / dry_mass[:, None]
    salt_fraction = dry_fraction[:, 1:].sum(axis=1)
    anchor = np.argsort(salt_fraction)
    selected = set(anchor[[0, len(anchor) // 2, -1]])
    # Unit statistical weights in a finite volume. Vapor forcing is externally
    # reset at each accepted step. Recover external additions to check closure
    # independently of the accumulated exchange reported by the integrator.
    gas = np.asarray(archive["gas_water_kg"])
    imposed = np.asarray(archive["imposed_rh"])
    total_water = water.sum(axis=1) + gas
    # Infer the pure saturation mass once from the initial reservoir.
    saturation_mass = gas[0] / rh[0]
    forcing_additions = np.r_[
        0.0, np.cumsum(imposed[1:] * saturation_mass - gas[:-1])
    ]
    ledger = total_water - total_water[0] - forcing_additions
    if np.max(abs(ledger)) / total_water[0] > 2e-12:
        raise ValueError(
            "gas, condensed and externally added water fail to close"
        )
    if not np.array_equal(water[0], np.asarray(archive["initial_water_kg"])):
        raise ValueError("initial particle waters differ from trajectory")
    cloud_water = np.full(len(particles), np.nan)
    cloud_rh = np.full(len(particles), np.nan)
    saturation = np.empty((len(particles), 2))
    errors = np.empty(len(particles))
    controls = []
    guess = None
    for i in anchor:
        guess, phases, errors[i] = _cloud_point(model, dry_fraction[i], guess)
        cloud_water[i] = phases[0, 0]
        masses = phases[0] * (dry_mass[i] / (1 - cloud_water[i]))
        radius = (3 * (masses / DENSITY).sum() / (4 * np.pi)) ** (1 / 3)
        kelvin = np.exp(
            2
            * SURFACE_TENSION
            * WATER_MW
            / (1000 * GAS_CONSTANT * TEMPERATURE * radius)
        )
        state = model.evaluate(phases, TEMPERATURE)
        cloud_rh[i] = state.neutral_activities[0, 0] * kelvin
        saturation[i] = np.log10(state.ion_activity_products[0]) - LOG10_KSP
        if i in selected:
            endpoint_indices = sorted(
                {0, len(rh) - 1, int(np.argmin(abs(rh - cloud_rh[i])))}
            )
            endpoint = []
            for j in endpoint_indices:
                mass = np.r_[water[j, i], dry[i]]
                split = model.phase_equilibrium(mass, TEMPERATURE, 0)
                # At both phases a saturated crystal has the same chemical potential.
                si = np.log10(split.alpha.ion_activity_products) - LOG10_KSP
                endpoint.append(
                    {
                        "sample_index": j,
                        "imposed_rh": float(imposed[j]),
                        "sampled_gas_rh": float(rh[j]),
                        "cpu_phase_count": 1 if split.beta is None else 2,
                        "resident_phase_count": int(observed[j, i]),
                        "log10_iap_over_ksp": si.tolist(),
                    }
                )
            controls.append(
                {
                    "particle_id": int(particles[i]),
                    "dry_fractions": dry_fraction[i].tolist(),
                    "dry_mass_kg": float(dry_mass[i]),
                    "equilibrium_onset_rh": float(cloud_rh[i]),
                    "equilibrium_onset_water_mass_fraction": float(
                        cloud_water[i]
                    ),
                    "alpha_mass_fractions": phases[0].tolist(),
                    "incipient_beta_mass_fractions": phases[1].tolist(),
                    "onset_log10_iap_over_ksp": saturation[i].tolist(),
                    "finite_difference_gibbs_gradient_residual_rt": float(
                        np.max(
                            abs(
                                _gradient_residual(
                                    model, phases[0], TEMPERATURE
                                )
                            )
                        )
                    ),
                    "sampled_controls": endpoint,
                }
            )
    water_fraction = water / (dry_mass[None, :] + water)
    independent = np.where(water_fraction < cloud_water[None, :], 2, 1)
    mismatch = independent != observed
    first = np.argmax(observed == 2, axis=0)
    brackets = [[int(j - 1), int(j)] if j else None for j in first]
    if brackets != archive["onset_sample_bracket"]:
        raise ValueError(
            "reported individual transition brackets disagree with the phase history"
        )
    sampled = np.any(observed == 2, axis=0) & (first > 0)
    lag = {"count": int(sampled.sum())}
    if sampled.any():
        lo = cloud_rh[sampled] - np.maximum(
            rh[first[sampled]], rh[first[sampled] - 1]
        )
        hi = cloud_rh[sampled] - np.minimum(
            rh[first[sampled]], rh[first[sampled] - 1]
        )
        lag.update(
            median_lower_bound=float(np.median(lo)),
            median_upper_bound=float(np.median(hi)),
            positive_for_entire_sample_bracket=int(np.sum(lo > 0)),
        )
    return {
        "two_organic_screen": mixed_solvent_screen(),
        "method": "CPU common chemical potentials of infinitesimal beta and water-rich alpha along each fixed dry-composition ray",
        "temperature_k": TEMPERATURE,
        "surface_tension_n_m": SURFACE_TENSION,
        "kelvin_liquid_density_kg_m3": 1000,
        "additive_volume_material_densities_kg_m3": DENSITY.tolist(),
        "material_order": list(RECIPE),
        "salt_order": ["halite", "sylvite"],
        "dissolution_log10_ksp": LOG10_KSP.tolist(),
        "max_relative_closed_step_water_error": float(
            np.max(abs(np.diff(ledger))) / total_water[0]
        ),
        "max_relative_cumulative_water_error": float(
            np.max(abs(ledger)) / total_water[0]
        ),
        "external_water_exchange_kg": float(forcing_additions[-1]),
        "particle_water_change_kg": float(water[-1].sum() - water[0].sum()),
        "gas_water_change_kg": float(gas[-1] - gas[0]),
        "sampled_transition_brackets_verified": len(brackets),
        "equilibrium_rh_minus_sampled_first_appearance_rh": lag,
        "trajectory_samples": len(rh),
        "unique_particle_ids": len(particles),
        "unique_dry_mass_vectors": len(np.unique(dry, axis=0)),
        "max_cloud_point_log_potential_mismatch": float(max(errors)),
        "particle_cloud_points": {
            "particle_id": particles.tolist(),
            "equilibrium_rh": cloud_rh.tolist(),
            "water_mass_fraction": cloud_water.tolist(),
        },
        "equilibrium_onset_rh_range": [
            float(cloud_rh.min()),
            float(cloud_rh.max()),
        ],
        "onset_log10_iap_over_ksp_range": np.column_stack(
            [saturation.min(axis=0), saturation.max(axis=0)]
        ).tolist(),
        "particle_sample_comparisons": int(observed.size),
        "disagreement_count": int(mismatch.sum()),
        "largest_water_fraction_difference_from_cloud_at_disagreement": float(
            np.max(np.where(mismatch, abs(water_fraction - cloud_water), 0))
        ),
        "observed": _crossings(observed, particles),
        "independent": _crossings(independent, particles),
        "representative_controls": controls,
    }


if __name__ == "__main__":
    folder = Path(__file__).resolve().parent
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--ramp", type=Path, default=folder / "ramp.json")
    cli.add_argument("--output", type=Path, default=folder / "equilibrium.json")
    cli.add_argument(
        "--screen-mixture",
        action="store_true",
        help="print the two-organic finite-difference qualification and exit",
    )
    args = cli.parse_args()
    if args.screen_mixture:
        print(json.dumps(mixed_solvent_screen(), indent=2))
        raise SystemExit(0)
    results = analyze(json.loads(args.ramp.read_text()))
    args.output.write_text(
        json.dumps(results, indent=2, allow_nan=False) + "\n"
    )
