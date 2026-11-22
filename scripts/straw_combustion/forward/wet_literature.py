"""Hydrolysis free-potassium balance at the given gas concentrations."""

import warp as wp

from .common import F


@wp.func
def _hydrolyzed(
    kh: F, binding: F, total_k: F, water: F, hcl: F, oxidative_loss: F
) -> F:
    # x = KOH + KHSO3. R10 releases an additional x HCl. binding is
    # KHSO3/KOH; the oxidative sink is replenished by hydrolysis.
    scaled = kh * (F(1) + binding)
    a = F(1) - scaled
    b = hcl + binding * oxidative_loss + scaled * (total_k + water)
    c = scaled * total_k * water
    return F(2) * c / (b + wp.sqrt(b * b + F(4) * a * c))
