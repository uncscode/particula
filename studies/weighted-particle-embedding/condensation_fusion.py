"""Two GPU passes for the 32-vapor implicit mass-transfer update."""

import torch
import triton
import triton.language as tl


@triton.jit
def _exchange(
    masses,
    radii,
    totals,
    path,
    linear,
    quad,
    coeff,
    sat,
    gas,
    partial,
    n,
    dt: tl.constexpr,
    apply: tl.constexpr,
    block: tl.constexpr,
):
    group = tl.program_id(0)
    i = group * block + tl.arange(0, block)
    j = tl.arange(0, 32)
    mask = i[:, None] < n
    offset = i[:, None] * 33 + j[None, :] + 1
    r = tl.load(radii + i, i < n, other=1.0)[:, None]
    total = tl.load(totals + i, i < n, other=1.0)[:, None]
    m = tl.load(masses + offset, mask, other=0.0)
    # Exact rational coefficient with constant species factors precomputed.
    kdt = (
        dt
        * tl.load(coeff + j)
        * (r * r)
        * (r + tl.load(path + j))
        / (r * (r + tl.load(linear + j)) + tl.load(quad + j))
    )
    c = tl.load(sat + j)
    inv = 1.0 / (1.0 + kdt * c / total)
    w = kdt * inv
    if apply:
        tl.store(masses + offset, m * inv + w * tl.load(gas + j), mask)
    else:
        tl.store(
            partial + group * 64 + j, tl.sum(tl.where(mask, w, 0.0), axis=0)
        )
        tl.store(
            partial + group * 64 + 32 + j,
            tl.sum(tl.where(mask, w * c * (m / total), 0.0), axis=0),
        )


def fused_step(box, dt):
    """Reduce vapor budgets, then commit the particle masses."""
    radii, totals = box.geometry()
    n = len(radii)
    groups = triton.cdiv(n, 64)
    partial = torch.empty((groups, 64), device=box.device, dtype=box.dtype)
    args = (
        box.mass,
        radii,
        totals,
        box.path,
        box.linear,
        box.quad,
        box.coeff,
        box.cstar,
        box.gas,
        partial,
        n,
        dt,
    )
    _exchange[(groups,)](*args, False, 64, enable_fp_fusion=False)
    weights, released = partial.sum(0).chunk(2)
    box.gas.copy_(
        (box.gas + released / box.volume) / (1 + weights / box.volume)
    )
    _exchange[(groups,)](*args, True, 64, enable_fp_fusion=False)
    box.vsteps += 1
