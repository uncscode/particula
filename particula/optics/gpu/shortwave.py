"""Batched scalar shortwave transfer on a caller-selected Warp device.

The azimuthal quadrature and static BRDFs are prepared once. Device kernels
recompute layer scattering, adding/doubling, and interface fluxes as optical
properties change. See Hansen & Travis (1974), doi:10.1007/BF00168069.
"""

from dataclasses import dataclass

import numpy as np
import warp as wp

from particula.optics._spectral import require_same_grid
from particula.optics.column import SpectralColumn
from particula.optics.gpu._shortwave import (
    BoundaryData,
    LayerData,
    begin_solve,
    difference_flux,
    integrate_flux,
    integrate_radiance,
    transport_kernel,
)
from particula.optics.radiative_transfer import (
    ShortwaveFlux,
    _phase_basis,
    _surface_modes,
)
from particula.optics.solar import SolarInput
from particula.optics.surface import Surface


def _sequence(values, cls, count):
    options = (values,) * count if isinstance(values, cls) else tuple(values)
    if len(options) != count or not all(isinstance(v, cls) for v in options):
        raise ValueError(f"Require a {cls.__name__} for each column")
    return options


def _quadrature(sun):
    if sun.quadrature_weights_m is not None:
        return sun.quadrature_weights_m
    wave = sun.wavelength_m
    if len(wave) == 1:
        return None
    return 0.5 * (np.r_[0.0, np.diff(wave)] + np.r_[np.diff(wave), 0.0])


def _integration(suns, sensor_response, has_view):
    weights = [_quadrature(s) for s in suns]
    available = all(x is not None for x in weights)
    if not available and any(x is not None for x in weights):
        raise ValueError(
            "Use consistent integration widths for single-node cases"
        )
    response = None
    if sensor_response is not None:
        rsp = np.asarray(sensor_response, dtype=float)
        if (
            not has_view
            or not available
            or rsp.shape != suns[0].wavelength_m.shape
            or np.any(~np.isfinite(rsp))
            or np.any(rsp < 0)
            or not np.any(rsp > 0)
        ):
            raise ValueError(
                "Require a view, widths and a positive response integral"
            )
        response = np.asarray(weights) * (rsp / rsp.max())
        response /= response.sum(axis=1, keepdims=True)
    return weights if available else None, response


def _reference_indices(columns, suns, indices):
    if indices is None:
        return None
    refs = np.asarray(indices)
    if (
        refs.shape != (len(columns),)
        or refs.dtype.kind not in "iu"
        or np.any(refs < 0)
        or np.any(refs >= len(columns))
    ):
        raise ValueError("Require one in-range reference case per column")
    for i, j in enumerate(refs):
        require_same_grid(
            columns[i].interface_height_m,
            columns[j].interface_height_m,
            "Interface heights",
        )
        for name in ("quadrature_weights_m", "band_bounds_m"):
            require_same_grid(
                getattr(suns[i], name), getattr(suns[j], name), name
            )
    return refs


def _upload_boundary(suns, surfaces, mu, weight, modes, limit, device):
    identities = {}
    distinct = []
    indices = []
    for sun, surface in zip(suns, surfaces, strict=True):
        key = id(sun), id(surface)
        if key not in identities:
            identities[key] = len(distinct)
            distinct.append((sun, surface))
        indices.append(identities[key])
    count, nw, n = len(distinct), len(suns[0].wavelength_m), len(mu)
    basis = np.zeros((count, modes + 1, limit, 2 * n + 1))
    reflection = np.empty((count, modes + 1, nw, n * n))
    direct = np.empty((count, modes + 1, nw, n))
    for b, (sun, surface) in enumerate(distinct):
        mu0 = sun.mu0 or 1.0
        for m in range(min(modes + 1, limit)):
            basis[b, m, m:] = _phase_basis(np.r_[mu, -mu, mu0], m, limit)
        for start in range(0, nw, 16):
            block = slice(start, min(start + 16, nw))
            for m, (r, s) in enumerate(
                _surface_modes(surface, mu, weight, mu0, modes, block)
            ):
                reflection[b, m, block] = r.reshape(-1, n * n)
                direct[b, m, block] = s[..., 0]
    bc = BoundaryData()
    bc.identity = wp.array(indices, dtype=wp.int32, device=device)
    for name, array in (
        ("basis", basis),
        ("reflection", reflection),
        ("direct_surface", direct),
        ("mu", mu),
        ("weight", weight),
        ("mu0", [s.mu0 or 1.0 for s, _ in distinct]),
        (
            "direct",
            [s.direct_normal if s.mu0 else np.zeros(nw) for s, _ in distinct],
        ),
        ("diffuse", [s.diffuse_horizontal for s, _ in distinct]),
        (
            "leaving",
            [
                g.leaving_radiance
                if g.leaving_radiance is not None
                else np.zeros(nw)
                for _, g in distinct
            ],
        ),
    ):
        setattr(
            bc,
            name,
            wp.array(np.asarray(array), dtype=wp.float64, device=device),
        )
    return bc


@dataclass
class ShortwaveWorkspace:
    """Retained inputs, launch commands and latest results on one stream.

    Attributes:
        layers: Writable dimensionless depth, albedo (case, layer, wavelength)
            and normalized moments with an additional Legendre degree axis.
        flux: (case, interface, wavelength) vec3d of physical direct down,
            diffuse down and upwelling densities in W m⁻² m⁻¹.
        broadband: (case, interface) vec3d of those fluxes in W m⁻².
            None for a single wavelength without quadrature widths.
        broadband_delta: Optional (case, interface) vec3d of broadband
            case minus reference fluxes in W m⁻², in broadband field order.
        radiance: (case, wavelength) emergent W m⁻² sr⁻¹ m⁻¹ if a view
            was requested. Defined after enqueue and a successful check.
        band_radiance: Optional (case,) response mean in W m⁻² sr⁻¹ m⁻¹.
        status: (case, wavelength) integers. Zero means success, one
            invalid layer data, two a nonfinite result, negative an incomplete
            angular solve. Check before consuming device outputs.
    """

    layers: LayerData
    flux: wp.array
    broadband: wp.array | None
    broadband_delta: wp.array | None
    radiance: wp.array | None
    band_radiance: wp.array | None
    status: wp.array
    _commands: tuple
    _resources: tuple
    _columns: tuple
    _suns: tuple
    _surfaces: tuple
    _view: tuple | None
    _stream: wp.Stream | None
    _ready: bool = False

    def enqueue(self):
        """Queue transport on the prepared stream, overwriting prior results."""
        for command in self._commands:
            command.launch(stream=self._stream)
        self._ready = True

    def check(self):
        """Synchronize and raise RuntimeError on missing or failed results."""
        if not self._ready:
            raise RuntimeError("Enqueue shortwave transport before observation")
        if self._stream is not None:
            wp.synchronize_stream(self._stream)
        with wp.ScopedStream(self._stream, sync_enter=False):
            codes = self.status.numpy()
        if np.any(codes):
            raise RuntimeError(
                f"Shortwave transport failed: {np.unique(codes[codes != 0])}"
            )

    def download(self):
        """Download spectral results in case order.

        Returns:
            tuple[ShortwaveFlux, ...]: Detached CPU results for each case.

        Raises:
            RuntimeError: If no solve has run or a device result failed.
        """
        self.check()
        with wp.ScopedStream(self._stream, sync_enter=False):
            flux = self.flux.numpy().copy()
            view = (
                None if self.radiance is None else self.radiance.numpy().copy()
            )
        results = []
        for i, (col, sun, surface) in enumerate(
            zip(self._columns, self._suns, self._surfaces, strict=True)
        ):
            results.append(
                ShortwaveFlux(
                    col.wavelength_m,
                    col.interface_height_m,
                    flux[i, ..., 0],
                    flux[i, ..., 1],
                    flux[i, ..., 2],
                    None if view is None else view[i],
                    sun.quadrature_weights_m,
                    (sun.source, surface.source, *col.sources),
                    self._view,
                    sun.band_bounds_m,
                )
            )
        return tuple(results)


def _column_layout(columns, sun, surface):
    cols = (columns,) if isinstance(columns, SpectralColumn) else tuple(columns)
    if not cols or not all(isinstance(col, SpectralColumn) for col in cols):
        raise ValueError("Require at least one SpectralColumn")
    suns = _sequence(sun, SolarInput, len(cols))
    surfaces = _sequence(surface, Surface, len(cols))
    wave = cols[0].wavelength_m
    nl, nw = cols[0].optical_depth.shape
    for col, src, ground in zip(cols, suns, surfaces, strict=True):
        if col.optical_depth.shape != (nl, nw):
            raise ValueError("Columns must share layer and wavelength counts")
        for item in (col, src, ground):
            require_same_grid(wave, item.wavelength_m, "Wavelengths")
    return cols, suns, surfaces, nl, nw


def _angles(streams, fourier_order, relative_azimuth_deg, view_zenith_deg):
    if not isinstance(streams, (int, np.integer)) or not 2 <= streams <= 16:
        raise ValueError("Warp requires 2–16 streams per hemisphere")
    if fourier_order is not None and (
        not isinstance(fourier_order, (int, np.integer))
        or not 0 <= fourier_order <= 64
    ):
        raise ValueError("Fourier order must be an integer from 0 to 64")
    if not np.isfinite(relative_azimuth_deg) or (
        view_zenith_deg is not None and not 0 <= view_zenith_deg < 90
    ):
        raise ValueError("Invalid viewing geometry")
    has_view = view_zenith_deg is not None
    view = (view_zenith_deg, relative_azimuth_deg) if has_view else None
    modes = (
        (2 * streams - 1 if fourier_order is None else fourier_order)
        if has_view
        else 0
    )
    mu, wt = np.polynomial.legendre.leggauss(streams)
    mu, wt = (mu + 1) / 2, wt / 2
    if has_view:
        mu, wt = np.r_[mu, np.cos(np.deg2rad(view_zenith_deg))], np.r_[wt, 0.0]
    return mu, wt, view, modes


def _tile_size(case_tile, lanes, n, nl):
    if case_tile is None:
        bytes_per_lane = 8 * (nl * (2 * n * n + 2 * n) + 2 * (nl + 1))
        case_tile = min(4096, max(1, (32 * 1024**2) // bytes_per_lane))
    if not isinstance(case_tile, (int, np.integer)) or case_tile < 1:
        raise ValueError("Require positive integer case_tile")
    return min(case_tile, lanes)


def _moments(cols, nl, nw, streams, delta_m):
    limit = min(2 * streams, max(c.phase_moments.shape[-1] for c in cols))
    delta = delta_m and any(c.phase_moments.shape[-1] > limit for c in cols)
    degrees = limit + int(delta)
    moments = np.zeros((len(cols), nl, nw, degrees))
    for i, c in enumerate(cols):
        take = min(degrees, c.phase_moments.shape[-1])
        moments[i, :, :, :take] = c.phase_moments[:, :, :take]
    return limit, delta, moments


def _device_stream(device, stream):
    dev = wp.get_device(device)
    if stream is not None and (not dev.is_cuda or stream.device != dev):
        raise ValueError("Stream must belong to the selected CUDA device")
    if stream is None and dev.is_cuda:
        stream = wp.get_stream(dev)
    return dev, stream


def prepare_shortwave(
    columns,
    sun,
    surface,
    *,
    device,
    stream=None,
    streams=8,
    view_zenith_deg=None,
    relative_azimuth_deg=0.0,
    fourier_order=None,
    delta_m=True,
    case_tile=None,
    reference_cases=None,
    sensor_response=None,
):
    """Prepare independent column/wavelength solves with reusable scratch.

    Args:
        columns (SpectralColumn | Sequence[SpectralColumn]): One
            SpectralColumn or a sequence with equal wavelength grids and layer
            counts. Interface heights may vary between columns.
        sun (SolarInput | Sequence[SolarInput]): SolarInput shared by all
            columns or one per column.
        surface (Surface | Sequence[Surface]): Shared surface or a sequence
            of spectral surface boundaries.
        device (str | warp.Device): Explicit Warp CPU or CUDA device. No
            automatic fallback.
        stream (warp.Stream | None): Optional CUDA stream on device. Input
            updates and consumers must use this stream or explicit inter-stream
            ordering.
        streams (int): Gauss–Legendre ordinates per hemisphere, 2 through 16.
        view_zenith_deg (float | None): Optional outgoing zenith angle [0, 90)
            in degrees.
        relative_azimuth_deg (float): Surface Sun–observer azimuth difference
            [deg].
        fourier_order (int | None): Maximum azimuthal harmonic, default 2 *
            streams - 1.
        delta_m (bool): Apply delta-M if a next phase moment is available.
        case_tile (int | None): Maximum concurrent column/wavelength lanes.
            The default targets 32 MiB of layer scratch and up to 4096 lanes.
        reference_cases (ArrayLike | None): Optional integer reference index
            for each case. Paired cases need matching heights and integration
            coordinates.
        sensor_response (ArrayLike | None): Optional nonnegative energy
            response on the grid. Requires a view and wavelength integration
            widths.

    Returns:
        ShortwaveWorkspace: Retained layer and boundary data. Use enqueue()
            after updating inputs. Use download() for detached CPU results.

    Raises:
        ValueError: If shapes, geometry, stream, or quadrature are invalid.
    """
    cols, suns, surfaces, nl, nw = _column_layout(columns, sun, surface)
    mu, wt, view, modes = _angles(
        streams, fourier_order, relative_azimuth_deg, view_zenith_deg
    )
    has_view = view is not None
    weights, response = _integration(suns, sensor_response, has_view)
    references = _reference_indices(
        columns=cols, suns=suns, indices=reference_cases
    )
    available = weights is not None
    count = len(cols)
    if references is not None and not available:
        raise ValueError("Broadband differences require integration widths")
    tile = _tile_size(case_tile, count * nw, len(mu), nl)
    limit, delta, moments = _moments(cols, nl, nw, streams, delta_m)
    dev, stream = _device_stream(device, stream)
    transport, matrix, vector = transport_kernel(len(mu))

    def empty(shape, dtype):
        return wp.empty(shape, dtype=dtype, device=dev)

    with wp.ScopedStream(stream, sync_enter=False):
        layers = LayerData()
        layers.depth = wp.array(
            np.stack([c.optical_depth for c in cols]),
            dtype=wp.float64,
            device=dev,
        )
        layers.albedo = wp.array(
            np.stack([c.single_scattering_albedo for c in cols]),
            dtype=wp.float64,
            device=dev,
        )
        layers.moments = wp.array(moments, dtype=wp.float64, device=dev)
        bc = _upload_boundary(suns, surfaces, mu, wt, modes, limit, dev)
        scratch = [
            empty((tile, nl), dtype)
            for dtype in (matrix, matrix, vector, vector)
        ]
        beams = empty((tile, nl + 1), wp.vec2d)
        flux = empty((count, nl + 1, nw), wp.vec3d)
        radiance = empty((count, nw), wp.float64)
        status = empty((count, nw), wp.int32)
        commands = []
        resources = [bc, *scratch, beams, radiance]

        def record(kernel, dim, args):
            commands.append(
                wp.launch(
                    kernel,
                    dim,
                    inputs=args,
                    device=dev,
                    stream=stream,
                    record_cmd=True,
                )
            )

        record(begin_solve, (count, nw), [status])
        for start in range(0, count * nw, tile):
            for m in range(modes + 1):
                factor = (
                    1
                    if m == 0
                    else 2
                    * np.cos(m * (np.pi - np.deg2rad(relative_azimuth_deg)))
                )
                record(
                    transport,
                    min(tile, count * nw - start),
                    [
                        layers,
                        bc,
                        start,
                        m,
                        modes,
                        limit,
                        int(delta),
                        int(has_view),
                        wp.float64(factor),
                        *scratch,
                        beams,
                        flux,
                        radiance,
                        status,
                    ],
                )
        broadband = None
        broadband_delta = None
        if available:
            broadband = empty((count, nl + 1), wp.vec3d)
            widths = wp.array(np.stack(weights), dtype=wp.float64, device=dev)
            resources.append(widths)
            record(
                integrate_flux,
                (count, nl + 1),
                [flux, widths, broadband, status],
            )
            if references is not None:
                refs = wp.array(references, dtype=wp.int32, device=dev)
                resources.append(refs)
                broadband_delta = empty((count, nl + 1), wp.vec3d)
                record(
                    difference_flux,
                    (count, nl + 1),
                    [broadband, refs, broadband_delta, status],
                )
        band = None
        if response is not None:
            band = empty(count, wp.float64)
            sensor = wp.array(response, dtype=wp.float64, device=dev)
            resources.append(sensor)
            record(integrate_radiance, count, [radiance, sensor, band, status])
    return ShortwaveWorkspace(
        layers,
        flux,
        broadband,
        broadband_delta,
        radiance if has_view else None,
        band,
        status,
        tuple(commands),
        tuple(resources),
        cols,
        suns,
        surfaces,
        view,
        stream,
    )
