# Warp interprets the device array and specialized matrix annotations.
# mypy: disable-error-code="valid-type,operator,index,call-overload,misc"
"""Device discrete ordinates and layer adding for scalar shortwave transport."""

from functools import lru_cache
from math import pi

import warp as wp

F = wp.float64
_PI = wp.constant(F(pi))


@wp.struct
class LayerData:
    depth: wp.array3d(dtype=F)
    albedo: wp.array3d(dtype=F)
    moments: wp.array4d(dtype=F)


@wp.struct
class BoundaryData:
    identity: wp.array(dtype=int)
    # Static angular responses indexed by boundary, Fourier mode and wavelength.
    basis: wp.array4d(dtype=F)
    reflection: wp.array4d(dtype=F)
    direct_surface: wp.array4d(dtype=F)
    mu: wp.array(dtype=F)
    weight: wp.array(dtype=F)
    mu0: wp.array(dtype=F)
    direct: wp.array2d(dtype=F)
    diffuse: wp.array2d(dtype=F)
    leaving: wp.array2d(dtype=F)


@wp.func
def _scaled_depth(tau: F, omega: F, forward: F):
    return tau * (F(1.0) - omega * forward)


@lru_cache(maxsize=None)
def transport_kernel(n):  # noqa: C901 - specialize one Warp module per size
    """Specialize local dense matrices to the number of angular ordinates."""
    size = 2 * n + 1
    matrix_type = wp.types.matrix((n, n), F)
    vector_type = wp.types.vector(n, F)
    generator_type = wp.types.matrix((size, size), F)
    module = wp.get_module(f"{__name__}.ordinates_{n}")
    module.options.update(max_unroll=1, enable_backward=False)

    @wp.func(module=module)
    def inverse(matrix: matrix_type, eye: matrix_type):
        # Partial-pivot Gauss–Jordan elimination for the small adding systems.
        work = matrix
        result = eye
        for k in range(n):
            pivot = int(k)
            for i in range(k + 1, n):
                if wp.abs(work[i, k]) > wp.abs(work[pivot, k]):
                    pivot = i
            for j in range(n):
                a = work[k, j]
                b = result[k, j]
                work[k, j] = work[pivot, j]
                result[k, j] = result[pivot, j]
                work[pivot, j] = a
                result[pivot, j] = b
            diagonal = work[k, k]
            for j in range(n):
                work[k, j] = work[k, j] / diagonal
                result[k, j] = result[k, j] / diagonal
            for i in range(n):
                if i != k:
                    factor = work[i, k]
                    for j in range(n):
                        work[i, j] = work[i, j] - factor * work[k, j]
                        result[i, j] = result[i, j] - factor * result[k, j]
        return result

    @wp.func(module=module)
    def slab(  # noqa: C901 - thin layer operator and repeated doubling
        c: int,
        k: int,
        w: int,
        mode: int,
        boundary: int,
        limit: int,
        tau: F,
        omega: F,
        forward: F,
        layers: LayerData,
        bc: BoundaryData,
        eye: matrix_type,
    ):
        generator = generator_type()
        mu0 = bc.mu0[boundary]
        den = F(1.0) - omega * forward
        omega_star = F(0.0)
        if den > F(0.0):
            omega_star = omega * (F(1.0) - forward) / den
        depth = _scaled_depth(tau, omega, forward)
        # The last coordinate is the attenuating direct-normal beam.
        for degree in range(mode, limit):
            beta = F(0.0)
            if forward < F(1.0):
                beta = (layers.moments[c, k, w, degree] - forward) / (
                    F(1.0) - forward
                )
            for i in range(2 * n):
                factor = beta * bc.basis[boundary, mode, degree, i]
                for j in range(size):
                    generator[i, j] += (
                        factor * bc.basis[boundary, mode, degree, j]
                    )
        bound = depth / mu0
        for i in range(2 * n):
            signed_mu = bc.mu[i % n]
            if i >= n:
                signed_mu = -signed_mu
            row_norm = F(0.0)
            for j in range(size):
                scale = F(1.0) / (F(4.0) * _PI)
                if j < 2 * n:
                    scale = F(0.5) * bc.weight[j % n]
                entry = omega_star * generator[i, j] * scale
                if i == j:
                    entry -= F(1.0)
                generator[i, j] = depth * entry / signed_mu
                row_norm += wp.abs(generator[i, j])
            bound = wp.max(bound, row_norm)
        generator[size - 1, size - 1] = -depth / mu0
        scale = F(1.0)
        doubles = int(0)
        while bound > F(0.5) and doubles < 1023:
            bound *= F(0.5)
            scale *= F(0.5)
            doubles += 1
        exponential = generator_type()
        for i in range(size):
            exponential[i, i] = F(1.0)
        term = exponential
        scaled = scale * generator
        for order in range(1, 17):
            term = (term * scaled) * (F(1.0) / F(order))
            exponential += term

        uu = matrix_type()
        ud = matrix_type()
        du = matrix_type()
        dd = matrix_type()
        up_source = vector_type()
        down_source = vector_type()
        for i in range(n):
            up_source[i] = -exponential[n + i, size - 1]
            down_source[i] = exponential[i, size - 1]
            for j in range(n):
                uu[i, j] = exponential[n + i, n + j]
                ud[i, j] = -exponential[n + i, j]
                du[i, j] = exponential[i, n + j]
                dd[i, j] = exponential[i, j]
        eliminate = inverse(uu, eye)
        r = eliminate * ud
        su = eliminate * up_source
        t = dd + du * r
        sd = down_source + du * su
        beam = wp.exp(-depth * scale / mu0)
        for _step in range(doubles):
            combine = inverse(eye - r * r, eye)
            dt = combine * t
            ds = combine * (sd + r * (beam * su))
            su = su + t * (r * ds + beam * su)
            sd = t * ds + beam * sd
            r = r + t * (r * dt)
            t = t * dt
            beam *= beam
        return r, t, su, sd

    @wp.kernel(module=module, enable_backward=False)
    def transport(  # noqa: C901 - one thread solves one wavelength and column
        layers: LayerData,
        bc: BoundaryData,
        start: int,
        mode: int,
        last_mode: int,
        limit: int,
        delta: int,
        view: int,
        fourier_factor: F,
        reflect: wp.array2d(dtype=matrix_type),
        transmit: wp.array2d(dtype=matrix_type),
        sup: wp.array2d(dtype=vector_type),
        sdown: wp.array2d(dtype=vector_type),
        beams: wp.array2d(dtype=wp.vec2d),
        flux: wp.array3d(dtype=wp.vec3d),
        radiance: wp.array2d(dtype=F),
        status: wp.array2d(dtype=int),
    ):
        lane = wp.tid()
        nw = layers.depth.shape[2]
        c = (start + lane) // nw
        w = (start + lane) % nw
        b = bc.identity[c]
        nl = layers.depth.shape[1]
        mu0 = bc.mu0[b]
        physical = bc.direct[b, w]
        scaled = physical
        if status[c, w] != -1 - mode:
            return
        if mode == 0:
            radiance[c, w] = F(0.0)
        for k in range(nl):
            beams[lane, k] = wp.vec2d(physical, scaled)
            tau = layers.depth[c, k, w]
            omega = layers.albedo[c, k, w]
            if (
                not wp.isfinite(tau)
                or tau < F(0.0)
                or not wp.isfinite(omega)
                or omega < F(0.0)
                or omega > F(1.0)
            ):
                status[c, w] = 1
                return
            if mode == 0:
                for j in range(layers.moments.shape[3]):
                    beta = layers.moments[c, k, w, j]
                    if (
                        not wp.isfinite(beta)
                        or wp.abs(beta) > F(1.0)
                        or (j == 0 and tau * omega > F(0.0) and beta != F(1.0))
                    ):
                        status[c, w] = 1
                        return
            forward = F(0.0)
            if delta:
                forward = wp.clamp(
                    layers.moments[c, k, w, limit], F(0.0), F(1.0)
                )
            physical *= wp.exp(-tau / mu0)
            scaled *= wp.exp(-_scaled_depth(tau, omega, forward) / mu0)
        beams[lane, nl] = wp.vec2d(physical, scaled)

        rg = matrix_type()
        sg = vector_type()
        eye = matrix_type()
        for i in range(n):
            eye[i, i] = F(1.0)
            sg[i] = scaled * bc.direct_surface[b, mode, w, i]
            if mode == 0:
                sg[i] += bc.leaving[b, w]
            for j in range(n):
                rg[i, j] = bc.reflection[b, mode, w, i * n + j]
        # Eliminate the unknown upward diffuse field from the ground upward.
        for rev in range(nl):
            k = nl - 1 - rev
            forward = F(0.0)
            if delta:
                forward = wp.clamp(
                    layers.moments[c, k, w, limit], F(0.0), F(1.0)
                )
            r, t, su, sd = slab(
                c,
                k,
                w,
                mode,
                b,
                limit,
                layers.depth[c, k, w],
                layers.albedo[c, k, w],
                forward,
                layers,
                bc,
                eye,
            )
            incident = beams[lane, k][1]
            combine = inverse(eye - r * rg, eye)
            dt = combine * t
            ds = combine * (incident * sd + r * sg)
            reflect[lane, k] = rg
            transmit[lane, k] = dt
            sup[lane, k] = sg
            sdown[lane, k] = ds
            sg = incident * su + t * (rg * ds + sg)
            rg = r + t * (rg * dt)

        down = vector_type()
        if mode == 0:
            for i in range(n):
                down[i] = bc.diffuse[b, w] / _PI
        for k in range(nl + 1):
            up = rg * down + sg
            if mode == 0:
                pair = beams[lane, k]
                direct_flux = mu0 * pair[0]
                down_flux = mu0 * (pair[1] - pair[0])
                up_flux = F(0.0)
                for i in range(n):
                    angular_weight = F(2.0) * _PI * bc.mu[i] * bc.weight[i]
                    down_flux += angular_weight * down[i]
                    up_flux += angular_weight * up[i]
                flux[c, k, w] = wp.vec3d(direct_flux, down_flux, up_flux)
                if not wp.isfinite(down_flux) or not wp.isfinite(up_flux):
                    status[c, w] = 2
            if view and k == 0:
                radiance[c, w] += fourier_factor * up[n - 1]
                if not wp.isfinite(radiance[c, w]):
                    status[c, w] = 2
            if k < nl:
                down = transmit[lane, k] * down + sdown[lane, k]
                rg = reflect[lane, k]
                sg = sup[lane, k]
        if status[c, w] < 0:
            status[c, w] = 0 if mode == last_mode else -2 - mode

    return transport, matrix_type, vector_type


@wp.kernel(enable_backward=False)
def begin_solve(status: wp.array2d(dtype=int)):
    """Mark every lane pending until all its angular launches have completed."""
    c, w = wp.tid()
    status[c, w] = -1


@wp.func
def _finite_flux(value: wp.vec3d):
    """Check all three directional flux components for NaN or infinity."""
    return (
        wp.isfinite(value[0])
        and wp.isfinite(value[1])
        and wp.isfinite(value[2])
    )


@wp.kernel(enable_backward=False)
def integrate_flux(
    flux: wp.array3d(dtype=wp.vec3d),
    weights: wp.array2d(dtype=F),
    total: wp.array2d(dtype=wp.vec3d),
    status: wp.array2d(dtype=int),
):
    """Integrate the three spectral fluxes at each column interface."""
    c, k = wp.tid()
    value = wp.vec3d()
    for w in range(flux.shape[2]):
        value += weights[c, w] * flux[c, k, w]
    total[c, k] = value
    if not _finite_flux(value):
        wp.atomic_max(status, c, 0, 2)


@wp.kernel(enable_backward=False)
def integrate_radiance(
    radiance: wp.array2d(dtype=F),
    response: wp.array2d(dtype=F),
    output: wp.array(dtype=F),
    status: wp.array2d(dtype=int),
):
    """Apply normalized wavelength weights to each directional spectrum."""
    c = wp.tid()
    value = F(0.0)
    for w in range(radiance.shape[1]):
        value += response[c, w] * radiance[c, w]
    output[c] = value
    if not wp.isfinite(value):
        wp.atomic_max(status, c, 0, 2)  # type: ignore[type-var]


@wp.kernel(enable_backward=False)
def difference_flux(
    total: wp.array2d(dtype=wp.vec3d),
    refs: wp.array(dtype=int),
    delta: wp.array2d(dtype=wp.vec3d),
    status: wp.array2d(dtype=int),
):
    """Subtract reference broadband flux at the same interface [W m⁻²]."""
    c, k = wp.tid()
    value = total[c, k] - total[refs[c], k]
    delta[c, k] = value
    if not _finite_flux(value):
        wp.atomic_max(status, c, 0, 2)
