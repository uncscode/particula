"""Derived ion-pair rates shared by the ordinary majorant and sampler paths."""

from typing import no_type_check

import warp as wp


@wp.struct
class IonPairData:
    """Derived ion data; disabled instances need no device allocation."""

    enabled: wp.int32
    kinds: wp.array2d(dtype=wp.int32)  # type: ignore[valid-type]
    attachment: wp.array3d(dtype=wp.float64)  # type: ignore[valid-type]
    recombination: wp.float64


@no_type_check
@wp.func
def replace_ion_pair_rate(
    data: IonPairData,
    box: wp.int32,
    first: wp.int32,
    second: wp.int32,
    aerosol_rate: wp.float64,
) -> wp.float64:
    """Select one physical law after setup validates rates."""
    if data.enabled == 0:
        return aerosol_rate
    first_kind = data.kinds[box, first]
    second_kind = data.kinds[box, second]
    if first_kind == 0 and second_kind == 0:
        return aerosol_rate
    if first_kind == 2 or second_kind == 2:
        return wp.float64(0.0)
    if first_kind == 0:
        sign = wp.int32(0)
        if second_kind > 0:
            sign = wp.int32(1)
        return data.attachment[box, first, sign]
    if second_kind == 0:
        sign = wp.int32(0)
        if first_kind > 0:
            sign = wp.int32(1)
        return data.attachment[box, second, sign]
    if first_kind == second_kind:
        return wp.float64(0.0)
    return data.recombination
