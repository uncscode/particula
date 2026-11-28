# Warp DSL buffer annotations are interpreted by its compiler.
# mypy: disable-error-code="valid-type,operator,index,misc"
"""Shortwave observations of live particles on the simulation stream."""

from dataclasses import replace
from math import pi

import numpy as np
import warp as wp

from particula.execution.prepared import PreparedSimulation
from particula.execution.prepared_recording import _ready
from particula.optics._spectral import require_same_grid
from particula.optics.column import SpectralColumn
from particula.optics.gpu._shortwave import LayerData
from particula.optics.gpu.shortwave import prepare_shortwave
from particula.optics.gpu.state import ParticleOpticalCases
from particula.optics.gpu.workspace import prepare_mie
from particula.optics.state import OpticalConfig

F = wp.float64
_PI = wp.constant(F(pi))


@wp.kernel(enable_backward=False)
def _particle_depth_inputs(
    radius: wp.array2d(dtype=F),
    weight: wp.array2d(dtype=F),
    efficiencies: wp.array2d(dtype=F),
    status: wp.array(dtype=int),
    bulk: wp.array(dtype=wp.vec3d),
):
    """Reduce extinction, scattering and its first moment per box/wavelength."""
    group = wp.tid()
    total = wp.vec3d()
    slots = radius.shape[1]
    for j in range(slots):
        i = group * slots + j
        r = radius[group, j]
        number = weight[group, j]
        if status[i] != 0 or not wp.isfinite(number) or number < F(0.0):
            bulk[group] = wp.vec3d(F(-1.0), F(0.0), F(0.0))
            return
        area = _PI * r * r * number
        ext = area * efficiencies[i, 0]
        sca = area * efficiencies[i, 1]
        total += wp.vec3d(ext, sca, sca * efficiencies[i, 3])
    bulk[group] = total


@wp.kernel(enable_backward=False)
def _combine(
    base: LayerData,
    active: LayerData,
    mapping: wp.array2d(dtype=int),
    thickness: wp.array2d(dtype=F),
    bulk: wp.array(dtype=wp.vec3d),
    order: int,
):
    """Add live aerosol to a fixed column using scattering-depth weights."""
    c, k, w = wp.tid()
    box = mapping[c, k]
    tau = base.depth[c, k, w]
    old_scattering = tau * base.albedo[c, k, w]
    added = F(0.0)
    g = F(0.0)
    if box >= 0:
        optical = bulk[box * active.depth.shape[2] + w]
        if optical[0] < F(0.0) or not wp.isfinite(optical[0]):
            active.depth[c, k, w] = F(-1.0)
            return
        tau += optical[0] * thickness[c, k]
        added = optical[1] * thickness[c, k]
        if optical[1] > F(0.0):
            g = optical[2] / optical[1]
    scattering = old_scattering + added
    active.depth[c, k, w] = tau
    active.albedo[c, k, w] = F(0.0)
    if tau > F(0.0):
        active.albedo[c, k, w] = scattering / tau
    hg = F(1.0)
    for j in range(active.moments.shape[3]):
        moment = F(0.0)
        if scattering > F(0.0):
            moment = old_scattering * base.moments[c, k, w, j]
            if j <= order:
                moment += added * hg
            moment /= scattering
        if j == 0:
            moment = F(1.0)
        active.moments[c, k, w, j] = moment
        hg *= g


class ResidentShortwave:
    """Observe spectral radiation between steps of a prepared Warp runnable.

    Attributes:
        workspace (ShortwaveWorkspace): Flux and broadband arrays on the
            simulation stream, available to subsequent device calculations.
    """

    def __init__(
        self,
        simulation,
        config,
        columns,
        sun,
        surface,
        *,
        box_indices,
        streams=8,
        hg_order=32,
        coefficient_capacity=512,
        particle_tile=256,
        **transport_options,
    ):
        """Bind live particle buffers and freeze atmospheric boundary inputs.

        Args:
            simulation (PreparedSimulation): Active Warp PreparedSimulation
                with discrete or particle-resolved counts and a positive fixed
                timestep.
            config (OpticalConfig): OpticalConfig for native particle Mie cross
                sections.
            columns (SpectralColumn | Sequence[SpectralColumn]): Background
                SpectralColumn(s) excluding the live aerosol.
            sun (SolarInput | Sequence[SolarInput]): Shared SolarInput or
                one per column.
            surface (Surface | Sequence[Surface]): Shared surface boundary
                or one per column.
            box_indices (ArrayLike): Integer (case, layer) map from top-first
                layers to simulation boxes. A value of -1 leaves that layer
                unchanged.
            streams (int): Angular ordinates per hemisphere.
            hg_order (int): Highest aerosol Henyey–Greenstein Legendre degree.
            coefficient_capacity (int): Maximum Mie series order in device
                scratch.
            particle_tile (int): Maximum particle/wavelength cases in Mie
                scratch.
            **transport_options (dict): Additional prepare_shortwave options
                such as view_zenith_deg, case_tile and sensor_response.

        Raises:
            ValueError: If particle layout, indices or optical inputs conflict.
            RuntimeError: If the owner is closed or inside CUDA capture.
        """
        if not isinstance(simulation, PreparedSimulation):
            raise TypeError("Expected a PreparedSimulation")
        if not isinstance(config, OpticalConfig):
            raise TypeError("Expected a native OpticalConfig")
        cols = (
            (columns,)
            if isinstance(columns, SpectralColumn)
            else tuple(columns)
        )
        if not cols or not all(isinstance(c, SpectralColumn) for c in cols):
            raise ValueError("Require at least one background column")
        if (
            not isinstance(hg_order, (int, np.integer))
            or not 1 <= hg_order <= 255
        ):
            raise ValueError("HG order must be an integer in [1,255]")
        if not isinstance(streams, (int, np.integer)) or not 2 <= streams <= 16:
            raise ValueError("Warp requires 2–16 streams per hemisphere")
        mapping = np.asarray(box_indices)
        nl, nw = cols[0].optical_depth.shape
        backgrounds = []
        for c in cols:
            require_same_grid(
                c.wavelength_m, config.wavelength, "Particle wavelengths"
            )
            if c.optical_depth.shape != (nl, nw):
                raise ValueError("Background columns must share layer counts")
            degrees = min(hg_order + 1, 2 * streams + 1)
            extra = max(0, degrees - c.phase_moments.shape[-1])
            moments = np.pad(c.phase_moments, ((0, 0), (0, 0), (0, extra)))
            backgrounds.append(
                replace(
                    c,
                    phase_moments=moments,
                    sources=(
                        *c.sources,
                        f"live native aerosol, HG order {hg_order}",
                    ),
                )
            )
        self._owner = simulation
        with simulation._lock:
            _ready(simulation)
            particles = simulation._session.particles
            if (
                mapping.dtype.kind not in "iu"
                or mapping.shape != (len(cols), nl)
                or np.any(mapping < -1)
                or np.any(mapping >= particles.masses.shape[0])
            ):
                raise ValueError(
                    "Require a valid simulation box or -1 per case and layer"
                )
            dev = simulation.device.native
            stream = simulation._stream
            with simulation._scope():
                self.workspace = prepare_shortwave(
                    backgrounds,
                    sun,
                    surface,
                    device=dev,
                    stream=stream,
                    streams=streams,
                    **transport_options,
                )
                cases = ParticleOpticalCases(
                    particles, config, device=dev, stream=stream
                )
                mie = prepare_mie(
                    cases.index.flatten(),
                    cases.size.flatten(),
                    coefficient_capacity=coefficient_capacity,
                    case_tile=particle_tile,
                    stream=stream,
                )
                bulk = wp.empty(cases.size.shape[0], dtype=wp.vec3d, device=dev)
                indices = wp.array(mapping, dtype=wp.int32, device=dev)
                dz = wp.array(
                    np.stack([-np.diff(c.interface_height_m) for c in cols]),
                    dtype=F,
                    device=dev,
                )
                base = LayerData()
                active = self.workspace.layers
                for name in ("depth", "albedo", "moments"):
                    setattr(base, name, wp.clone(getattr(active, name)))
                self._reduce = wp.launch(
                    _particle_depth_inputs,
                    bulk.size,
                    inputs=[
                        cases.radius,
                        cases.weight,
                        mie.efficiencies,
                        mie.status,
                        bulk,
                    ],
                    device=dev,
                    stream=stream,
                    record_cmd=True,
                )
                self._combine = wp.launch(
                    _combine,
                    active.depth.shape,
                    inputs=[base, active, indices, dz, bulk, hg_order],
                    device=dev,
                    stream=stream,
                    record_cmd=True,
                )
                self._cases = cases
                self._mie = mie
                self._resources = (base, indices, dz, bulk)
                self._particles = particles

    def enqueue(self):
        """Sample current particle state without advancing or downloading it.

        Raises:
            RuntimeError: The simulation is closed, replaced or capturing.
        """
        owner = self._owner
        with owner._lock:
            _ready(owner)
            if owner._session.particles is not self._particles:
                raise RuntimeError(
                    "Particle storage changed; rebind the diagnostic"
                )
            self._cases.enqueue()
            self._mie.enqueue()
            self._reduce.launch(stream=owner._stream)
            self._combine.launch(stream=owner._stream)
            self.workspace.enqueue()

    def check(self):
        """Synchronize the simulation and inspect optical status on its stream.

        Raises:
            RuntimeError: If no observation exists or physics or optics failed.
        """
        owner = self._owner
        with owner._lock:
            _ready(owner)
            owner.synchronize()
            self.workspace.check()

    def download(self):
        """Synchronize physics and optics and return detached ShortwaveFluxes.

        Returns:
            (tuple[ShortwaveFlux, ...]): One result per column for the latest
                enqueue call.

        Raises:
            RuntimeError: An owner, Mie or radiation calculation failed.
        """
        owner = self._owner
        with owner._lock:
            _ready(owner)
            owner.synchronize()
            return self.workspace.download()
