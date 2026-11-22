"""Fit an isolated synthetic dimer aging sequence with PyTorch CUDA AD."""

import numpy as np
import torch
from scipy.optimize import minimize


def inventories(rates, hours):
    """Propagate particle dimer, oxidized particle and gas daughter carbon."""
    oxidation, cleavage = rates.unbind()
    zero = rates.new_zeros(())
    generator = torch.stack(
        (
            torch.stack((-oxidation, zero, zero)),
            torch.stack((oxidation, -cleavage, zero)),
            torch.stack((zero, cleavage, zero)),
        )
    )
    # Column zero propagates one unit of initial particle dimer carbon.
    return torch.linalg.matrix_exp(hours[:, None, None] * generator)[:, :, 0]


def main():
    """Compare fits to species observations and ambiguity in suspended mass."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "This example requires a CUDA-enabled PyTorch build."
        )
    options = {"device": "cuda", "dtype": torch.float64}
    hours = torch.tensor([0.0, 0.25, 0.75, 1.5, 3.0, 5.0], **options)
    truth = torch.tensor([0.8, 0.3], **options)  # per hour
    observed = inventories(truth, hours)
    torch.testing.assert_close(observed.sum(1), torch.ones_like(hours))
    torch.testing.assert_close(
        observed[:, :2].sum(1), inventories(truth.flip(0), hours)[:, :2].sum(1)
    )

    def objective(values):
        rates = torch.tensor(values, requires_grad=True, **options)
        residual = inventories(rates, hours)[:, :2] - observed[:, :2]
        loss = residual.square().mean()
        (gradient,) = torch.autograd.grad(loss, rates)
        return loss.item(), gradient.detach().cpu().numpy()

    start = np.array([0.45, 0.55])
    direction = np.array([0.6, -0.8])
    _, grad = objective(start)
    for step in (1e-4, 1e-5):
        plus, _ = objective(start + step * direction)
        minus, _ = objective(start - step * direction)
        np.testing.assert_allclose(
            grad @ direction, (plus - minus) / (2 * step), rtol=1e-5, atol=1e-9
        )

    for name, upper in (("oxidation only", 0.0), ("oxidation + cleavage", 3.0)):
        fit = minimize(
            objective,
            [start[0], min(start[1], upper)],
            method="L-BFGS-B",
            jac=True,
            bounds=[(0.0, 3.0), (0.0, upper)],
            options={"gtol": 1e-10, "ftol": 1e-14},
        )
        if not fit.success:
            raise RuntimeError(f"{name}: {fit.message}")
        print(f"{name}: rates {fit.x} per hour, species MSE {fit.fun:.6g}")
        if upper:
            np.testing.assert_allclose(fit.x, truth.cpu().numpy(), atol=1e-5)
    print("Swapping rates preserves suspended carbon at every sample.")


if __name__ == "__main__":
    main()
