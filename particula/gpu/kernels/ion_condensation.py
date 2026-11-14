"""Gate ion-only vapor proposals before condensation inventory reduction."""

from typing import no_type_check

import warp as wp


@no_type_check
@wp.kernel
def exclude_ion_candidates(
    candidate: wp.array3d(dtype=wp.float64),
    masses: wp.array3d(dtype=wp.float64),
    positive_species: wp.int32,
    negative_species: wp.int32,
):
    """Zero uptake and evaporation proposals for ion-only slots."""
    box, slot = wp.tid()
    aerosol = wp.int32(0)
    for species in range(masses.shape[2]):
        if species != positive_species and species != negative_species:
            if masses[box, slot, species] > wp.float64(0.0):
                aerosol = wp.int32(1)
    if aerosol == 0:
        for species in range(candidate.shape[2]):
            candidate[box, slot, species] = wp.float64(0.0)


def gate_ion_condensation(prepared):
    """Apply the optional prepared filter without touching native state."""
    if prepared.ion_material_indices is None:
        return
    wp.launch(
        exclude_ion_candidates,
        dim=prepared.dimensions[:2],
        inputs=[
            prepared.candidate,
            prepared.particle_masses,
            *prepared.ion_material_indices,
        ],
        device=prepared.device,
    )
