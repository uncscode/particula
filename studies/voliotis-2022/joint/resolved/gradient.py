"""AD steps for chemical tendencies and the smoothed stochastic objective."""

import numpy as np
import warp as wp
from scipy.optimize import minimize
from simulate import BANK, DEVICE, F64, upload


@wp.kernel
def likelihood_gradient(
    center: wp.array(dtype=F64),
    samples: wp.array2d(dtype=F64),
    loss: wp.array(dtype=F64),
    objective: wp.array(dtype=F64),
    sigma: F64,
):
    """Accumulate a Gaussian search log-density weighted by simulation loss."""
    i, j = wp.tid()
    delta = (samples[i, j] - center[j]) / sigma
    wp.atomic_add(
        objective,
        0,
        -F64(0.5) * loss[i] * delta * delta / F64(samples.shape[0]),
    )


def gaussian_ad(samples, losses, sigma):
    """Differentiate the search density at fixed particle event outcomes."""
    center = wp.zeros(
        samples.shape[1], dtype=F64, device=DEVICE, requires_grad=True
    )
    target = wp.zeros(1, dtype=F64, device=DEVICE, requires_grad=True)
    with wp.Tape() as tape:
        wp.launch(
            likelihood_gradient,
            dim=samples.shape,
            inputs=[
                center,
                upload(samples),
                upload(losses - losses.mean()),
                target,
                sigma,
            ],
            device=DEVICE,
        )
    tape.backward(target)
    gradient = center.grad.numpy()
    # Independent analytic score-function identity on the very same batches.
    np.testing.assert_allclose(
        gradient,
        (samples.T @ (losses - losses.mean())) / (len(losses) * sigma**2),
        atol=1e-10,
    )
    return gradient


@wp.kernel
def tendency_objective(
    design: wp.array2d(dtype=F64),
    residual: wp.array(dtype=F64),
    scales: wp.array(dtype=F64),
    change: wp.array(dtype=F64),
    loss: wp.array(dtype=F64),
):
    """Integrate residuals under a local kinetic parameter change."""
    t = wp.tid()
    value = residual[t]
    for h in range(change.shape[0]):
        value += design[t, h] * scales[h] * change[h]
    wp.atomic_add(loss, 0, value * value / F64(design.shape[0]))


def conditional_chemistry(chamber, h, gas, v, upper):
    """Fit the derivative of net particulate mass along frozen chemical states.

    The linear design is the integrated mass-action tendency at each snapshot.
    State changes following a perturbation are evaluated by a nonlinear replay.
    """
    features = np.zeros((len(chamber.ids), 61, 70))
    mw = np.array([184.0, 140.0, 118.0])
    for j, channel in enumerate(BANK):
        a, b, dest = channel.a, channel.b, channel.dest

        def solid(q):
            return 6 <= q <= 9 or 11 <= q <= 25

        net = float(solid(dest)) - float(solid(a))
        if channel.split >= 0:
            net = 0.5 * (solid(dest) + solid(channel.split)) - float(solid(a))
        if not net:
            continue
        rate = h[:, :, 11 + a] if solid(a) else gas[:, :, a]
        if b >= 0:
            if solid(a):
                raise ValueError("Unexpected bimolecular particle mass sink")
            rate = rate * gas[:, :, b] * (1 / mw[a % 3] + 1 / mw[b % 3])
        if channel.area_scaled:
            rate = (
                rate * h[:, :, 10] * 1e-9
            )  # nm2/cm3 to m2/m3, divided by 0.001
        features[:, :, j] = rate * net
    integrated = np.zeros_like(features)
    integrated[:, 1:] = np.cumsum(
        (features[:, 1:] + features[:, :-1]) / 24, axis=1
    )
    designs = []
    residuals = []
    for i, (_, _, t, y, _) in enumerate(chamber.runs):
        # Match irregular AMS observations on the same five-minute grid.
        time = np.arange(61) * 300
        scale = max(y.max(), 1) * np.sqrt(len(t))
        designs.append(
            np.column_stack([np.interp(t, time, f) for f in integrated[i].T])
            / scale
        )
        residuals.append((np.interp(t, time, h[i, :, 0]) - y) / scale)
    design = upload(np.vstack(designs))
    residual = upload(np.concatenate(residuals))
    rate = v[16:86]
    scale = np.minimum(0.20 * upper, 0.35 * (rate + 0.03))
    scales = upload(scale)

    def fun(z):
        arg = wp.array(z, dtype=F64, device=DEVICE, requires_grad=True)
        target = wp.zeros(1, dtype=F64, device=DEVICE, requires_grad=True)
        with wp.Tape() as tape:
            wp.launch(
                tendency_objective,
                dim=design.shape[0],
                inputs=[design, residual, scales, arg, target],
                device=DEVICE,
            )
        tape.backward(target)
        return float(target.numpy()[0]) * 1000 + 0.001 * np.mean(
            z * z
        ), arg.grad.numpy() * 1000 + 0.002 * z / len(z)

    origin = np.zeros(len(rate))
    # Check actual AD against a symmetric directional derivative of this
    # objective.
    probe = np.random.default_rng(41).normal(size=len(rate))
    probe /= np.linalg.norm(probe)
    derivative = (fun(1e-4 * probe)[0] - fun(-1e-4 * probe)[0]) / 2e-4
    np.testing.assert_allclose(fun(origin)[1] @ probe, derivative, atol=1e-8)
    low = np.maximum(-1, -rate / scale)
    high = np.minimum(1, (upper - rate) / scale)
    opt = minimize(
        fun,
        origin,
        jac=True,
        method="L-BFGS-B",
        bounds=list(zip(low, high, strict=True)),
        options={"maxiter": 80},
    )
    candidate = v.copy()
    candidate[16:86] = np.clip(rate + opt.x * scale, 0, upper)
    return candidate, dict(
        initial=fun(origin)[0],
        final=float(opt.fun),
        iterations=opt.nit,
        verified_directional_derivative=float(derivative),
    )
