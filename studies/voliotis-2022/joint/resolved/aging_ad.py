"""Differentiate the finite-time product-aging reactions in five families."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize


def fit(archive, inputs, destination):
    """Propagate the five product families for a conditional 30-minute fit."""
    torch.set_num_threads(1)
    data = np.load(inputs)
    network = json.loads(Path(archive).read_text())["reactions"][20:]
    # For independent first-order particle reactions, propagating summed
    # inventories is algebraically identical to propagating every particle.
    basis = np.zeros((50, 5, 4, 4))
    for i, r in enumerate(network):
        a = r["a"]
        family = (a - 11) % 5
        col = (a - 11) // 5
        basis[i, family, col, col] -= 1
        for to, fraction in (
            (r["dest"], 1.0 if r["split"] < 0 else 0.5),
            (r["split"], 0.5),
        ):
            if to >= 11:
                if (to - 11) % 5 != family:
                    raise ValueError(
                        "Reaction connects unexpected product families"
                    )
                basis[i, family, (to - 11) // 5, col] += fraction
    basis = torch.tensor(basis, dtype=torch.float64)
    # Secondary-vapor background loss occurs in every family.
    decay = torch.zeros((5, 4, 4), dtype=torch.float64)
    decay[:, 3, 3] = -1
    init = torch.tensor(data["pools"], dtype=torch.float64)
    k0 = torch.tensor(data["rates"], dtype=torch.float64)
    residual = torch.tensor(data["residual"], dtype=torch.float64)
    normalization = torch.tensor(data["normalization"], dtype=torch.float64)

    def retained(k):
        matrix = decay + torch.einsum("k,kfij->fij", k, basis)
        transfer = torch.linalg.matrix_exp(matrix * 0.5)
        return torch.einsum("fij,nfj->n", transfer[:, :3, :], init)

    reference = retained(k0).detach()
    scale = 0.35 * (k0 + 0.05)

    def objective(z):
        x = torch.tensor(z, requires_grad=True, dtype=torch.float64)
        error = (
            residual + retained(k0 + x * scale) - reference
        ) / normalization
        loss = (error**2).mean() + 0.0001 * (x**2).mean()
        loss.backward()
        return loss.item(), x.grad.detach().numpy()

    origin = np.zeros(len(k0))
    probe = np.random.default_rng(905).normal(size=len(k0))
    probe /= np.linalg.norm(probe)
    fd = (objective(1e-4 * probe)[0] - objective(-1e-4 * probe)[0]) / 2e-4
    np.testing.assert_allclose(
        objective(origin)[1] @ probe, fd, atol=1e-8, rtol=1e-4
    )
    lo = np.maximum(-1, -k0.numpy() / scale.numpy())
    hi = np.minimum(1, (5 - k0.numpy()) / scale.numpy())
    result = minimize(
        objective,
        origin,
        jac=True,
        bounds=list(zip(lo, hi, strict=True)),
        method="L-BFGS-B",
        options={"ftol": 1e-13, "gtol": 1e-9, "maxiter": 150},
    )
    rates = (k0.numpy() + result.x * scale.numpy()).clip(0, 5)
    Path(destination).write_text(
        json.dumps(
            dict(
                rates=rates.tolist(),
                initial=objective(origin)[0],
                final=result.fun,
                steps=result.nit,
                success=bool(result.success),
                directional_derivative=fd,
                torch=torch.__version__,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("archive")
    p.add_argument("inputs")
    p.add_argument("destination")
    args = p.parse_args()
    fit(args.archive, args.inputs, args.destination)
