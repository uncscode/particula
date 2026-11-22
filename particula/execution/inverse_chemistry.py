"""Differentiate a fixed SiMBA network through deterministic chamber growth."""

from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import warp as wp
from scipy.special import ndtri

from particula.dynamics.chemistry import ReactionMechanism
from particula.dynamics.chemistry.binding import _compile_network
from particula.dynamics.chemistry.observations import ObservationSeries
from particula.execution import _discovery_trajectory as kernels
from particula.execution._discovery_measurements import observe_number_spectrum


@dataclass(frozen=True)
class VaporExchange:
    """Paired phases of one compound and its constant transport properties."""

    gas: str
    particle: str
    diffusivity_m2_s: float
    mean_free_path_m: float


@dataclass(frozen=True)
class SeededChamber:
    """Fixed chemistry, seed composition, instrument and chamber conditions.

    Densities (kg/m³) and initial seed volume fractions are keyed by particle
    species. Gas and particle species require declared molar masses. A flat
    interface, ideal activities and unit accommodation define vapor exchange.
    """

    mechanism: ReactionMechanism
    density: dict[str, float]
    seed_fractions: dict[str, float]
    exchange: tuple[VaporExchange, ...]
    organic: tuple[str, ...]
    measured_gases: tuple[str, ...]
    edges_m: tuple[float, ...]
    log_response_width: float
    cohorts: int


@dataclass
class Trajectory:
    """Predictions, final inventories, loss and optional derivatives."""

    predictions: np.ndarray
    final_state: np.ndarray
    species: tuple[str, ...]
    loss: float | None
    gradients: dict[str, np.ndarray]
    transfer_bytes: dict[str, int]


def _vector(values, length, label):
    array = np.asarray(values, dtype=float)
    if (
        array.shape != (length,)
        or not np.isfinite(array).all()
        or np.any(array < 0)
    ):
        raise ValueError(
            f"Invalid {label}: expected {length} nonnegative finite values"
        )
    return array


def _compile(chamber):
    mechanism = chamber.mechanism
    names = tuple(n for n, _ in mechanism.species)
    gases = {n for n in names if n.startswith("gas:")}
    particles = set(names) - gases
    masses = dict(mechanism.declared_masses)
    if set(masses) != set(names) or set(chamber.density) != particles:
        raise ValueError("Declare all molar masses and all particle densities")
    if (
        not set(chamber.seed_fractions) <= particles
        or not set(chamber.organic) <= particles
    ):
        raise ValueError(
            "Seed fractions and organic inventory require particle species"
        )
    if not set(chamber.measured_gases) <= gases:
        raise ValueError("Invalid measured gas")
    fractions = _vector(
        [chamber.seed_fractions.get(n, 0.0) for n in names],
        len(names),
        "seed fractions",
    )
    if not np.isclose(sum(fractions), 1.0, rtol=0, atol=1e-12) or any(
        not np.isfinite(v) or v <= 0 for v in chamber.density.values()
    ):
        raise ValueError("Invalid seed volume fractions or particle densities")
    vol = [
        masses[n] / chamber.density[n] if n in particles else 0.0 for n in names
    ]
    composition = dict(mechanism.species)
    change, _, _ = _compile_network(
        mechanism, composition, [masses[n] for n in names]
    )
    phase, first, second = [], [], []
    for reaction in mechanism.reactions:
        lhs = [
            names.index(n) for n, order in reaction.orders for _ in range(order)
        ]
        if reaction.law != "constant" or len(lhs) not in (1, 2):
            raise ValueError(
                "Expected constant unimolecular or bimolecular rates"
            )
        phase.append(int(names[lhs[0]] in particles))
        first.append(lhs[0])
        second.append(lhs[1] if len(lhs) == 2 else -1)

    _validate_exchange(chamber, gases, particles, masses, composition)
    return names, masses, vol, fractions, (phase, first, second, change)


def _validate_exchange(chamber, gases, particles, masses, composition):
    if len({(p.gas, p.particle) for p in chamber.exchange}) != len(
        chamber.exchange
    ):
        raise ValueError("Duplicate vapor exchange pair")
    for pair in chamber.exchange:
        if (
            pair.gas not in gases
            or pair.particle not in particles
            or not np.isclose(
                masses[pair.gas], masses[pair.particle], rtol=1e-12, atol=0
            )
        ):
            raise ValueError(
                "Exchange requires matching gas and particle molar masses"
            )
        if (
            composition[pair.gas]
            and composition[pair.particle]
            and composition[pair.gas] != composition[pair.particle]
        ):
            raise ValueError(
                "Exchange species have different elemental compositions"
            )
        if not all(
            np.isfinite(v) and v > 0
            for v in (pair.diffusivity_m2_s, pair.mean_free_path_m)
        ):
            raise ValueError("Invalid vapor transport properties")


def _measurement_grid(chamber, times, max_step, observations, gradient, seed):
    if seed[0] <= 0 or seed[1] <= 0 or seed[2] < 1 or chamber.cohorts < 1:
        raise ValueError("Invalid lognormal seed population")
    times = np.asarray(times, dtype=float)
    if (
        times.ndim != 1
        or not len(times)
        or not np.isfinite(times).all()
        or times[0] < 0
        or np.any(np.diff(times) <= 0)
        or not np.isfinite(max_step)
        or max_step <= 0
    ):
        raise ValueError("Invalid observation times or step")
    edges = _vector(chamber.edges_m, len(chamber.edges_m), "edges")
    if (
        not len(edges)
        or edges[0] <= 0
        or np.any(np.diff(edges) <= 0)
        or not np.isfinite(chamber.log_response_width)
        or chamber.log_response_width <= 0
    ):
        raise ValueError("Invalid size instrument")
    if observations is not None and not np.array_equal(
        observations.times, times
    ):
        raise ValueError("Observation times do not match")
    if gradient and observations is None:
        raise ValueError("A gradient requires observations")
    return times, edges


def _integrate(state, duration, max_step, slope, advance):
    """Take fixed midpoint steps ending exactly at an observation time."""
    steps = int(np.ceil(duration / max_step))
    dt = duration / max(steps, 1)
    for _ in range(steps):
        midpoint = advance(state, slope(state), dt / 2)
        state = advance(state, slope(midpoint), dt)
    return state


def _transfers(device):
    """Count explicit input and output tensor bytes for a CUDA replay."""
    traffic = {"host_to_device": 0, "device_to_host": 0}
    cuda = wp.get_device(device).is_cuda

    def upload(x, dtype, trainable=False):
        array = wp.array(x, dtype=dtype, device=device, requires_grad=trainable)
        if cuda:
            traffic["host_to_device"] += array.size * (
                8 if dtype == wp.float64 else 4
            )
        return array

    def download(array):
        host = array.numpy()
        if cuda:
            traffic["device_to_host"] += host.nbytes
        return host

    return upload, download, traffic


def simulate_chamber(
    chamber: SeededChamber,
    times,
    *,
    dose,
    rates,
    saturation,
    seed,
    max_step,
    device,
    observations: ObservationSeries | None = None,
    gradient=False,
) -> Trajectory:
    """Integrate a fixed candidate with explicit midpoint and optional Warp AD.

    Dose is mol/m³ air in sorted gas-species order. Rates follow native reaction
    order, saturation (mol/m³) follows exchange order. Seed controls are total
    number/m³, median diameter (m), geometric standard deviation, and first-
    order deposition at 100 nm (1/s). Times and max_step are in seconds.
    Predictions consist of measured gases, suspended organic kg/m³, and the
    number spectrum including both tails. A NaN observation masks a channel.
    """
    names, masses, vol, frac, kinetics = _compile(chamber)
    ns = len(names)
    gas_ids = [i for i, n in enumerate(names) if n.startswith("gas:")]
    d = np.zeros(ns)
    d[gas_ids] = _vector(dose, len(gas_ids), "dose")
    inputs = dict(
        dose=d,
        rates=_vector(rates, len(kinetics[0]), "rates"),
        saturation=_vector(saturation, len(chamber.exchange), "saturation"),
        seed=_vector(seed, 4, "seed"),
    )
    times, edges = _measurement_grid(
        chamber, times, max_step, observations, gradient, seed
    )
    opts = dict(dtype=wp.float64, device=device)
    upload, download, traffic = _transfers(device)
    params = {k: upload(v, wp.float64, gradient) for k, v in inputs.items()}

    def floats(x):
        return upload(x, wp.float64)

    def ints(x):
        return upload(np.asarray(x, dtype=np.int32), wp.int32)

    def empty(shape):
        return wp.empty(shape, requires_grad=gradient, **opts)

    def run(kernel, dim, args):
        wp.launch(kernel, dim=dim, inputs=args, device=device)

    n = chamber.cohorts
    shape = (n + 1, 2 * ns + 1)
    mv = floats(vol)
    mass = floats([masses[k] for k in names])
    organic = ints([k in chamber.organic for k in names])
    gases = ints([names.index(k) for k in chamber.measured_gases])
    ix = [ints(col) for col in kinetics]
    gmap = ints([names.index(p.gas) for p in chamber.exchange])
    pmap = ints([names.index(p.particle) for p in chamber.exchange])
    pairs = len(chamber.exchange)
    diffusion = floats([p.diffusivity_m2_s for p in chamber.exchange])
    path = floats([p.mean_free_path_m for p in chamber.exchange])
    cutoffs = floats(edges)
    bad = wp.zeros(1, dtype=int, device=device)

    def geometry(state):
        bulk = empty((n, 4))
        inventory = empty((n, 4))
        run(kernels.inventory, n, [state, mv, inventory])
        run(kernels.geometry, n, [inventory, bulk])
        return bulk

    def slope(state):
        bulk = geometry(state)
        flux = empty((n, pairs))
        if pairs:
            run(
                kernels.exchange,
                (n, pairs),
                [
                    state,
                    bulk,
                    gmap,
                    pmap,
                    diffusion,
                    path,
                    params["saturation"],
                    flux,
                ],
            )
        rhs = empty(shape)
        run(
            kernels.derivative,
            shape,
            [
                state,
                bulk,
                flux,
                gmap,
                pmap,
                params["rates"],
                *ix,
                params["seed"],
                rhs,
            ],
        )
        return rhs

    def advance(state, rhs, dt):
        result = empty(shape)
        run(kernels.advance, shape, [state, rhs, dt, result, bad])
        return result

    tape = wp.Tape() if gradient else None
    outputs = []
    with tape if tape is not None else nullcontext():
        state = empty(shape)
        run(
            kernels.initialize,
            shape,
            [
                params["dose"],
                params["seed"],
                floats(ndtri((np.arange(n) + 0.5) / n)),
                floats(frac),
                mv,
                state,
            ],
        )
        time = 0.0
        for target in times:
            state = _integrate(state, target - time, max_step, slope, advance)
            time = target
            diam, count = empty((1, n)), empty((1, n))
            run(kernels.cohort_observables, n, [geometry(state), diam, count])
            bins = empty((1, len(edges) + 1))
            run(
                observe_number_spectrum,
                bins.shape,
                [diam, count, cutoffs, chamber.log_response_width, bins],
            )
            out = empty(len(chamber.measured_gases) + len(edges) + 2)
            run(
                kernels.observe,
                out.shape,
                [state, bins, mass, organic, gases, out],
            )
            outputs.append(out)
    return _finish(
        bad,
        outputs,
        observations,
        tape,
        floats,
        params,
        gas_ids,
        state,
        names,
        download,
        traffic,
    )


def _finish(
    bad,
    outputs,
    observations,
    tape,
    floats,
    params,
    gas_ids,
    state,
    names,
    download,
    traffic,
):
    if download(bad)[0]:
        raise ValueError("Invalid trajectory stage; reduce max_step")
    predictions = np.stack([download(o) for o in outputs])
    if not np.isfinite(predictions).all():
        raise ValueError("Nonfinite observation")
    loss, adj = (
        observations.loss(predictions)
        if observations is not None
        else (None, None)
    )
    grads = {}
    if tape is not None:
        tape.backward(
            grads={o: floats(a) for o, a in zip(outputs, adj, strict=True)}
        )
        grads = {k: download(p.grad) for k, p in params.items()}
        grads["dose"] = grads["dose"][gas_ids]
    return Trajectory(predictions, download(state), names, loss, grads, traffic)
