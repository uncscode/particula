"""Check the two-pass kernel on an incomplete GPU block and after growth."""

import pytest
import torch
from condensation_options import SPECIES, TrialBox
from process_scale import DT

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Tests the study's CUDA implementation",
)


@torch.inference_mode()
def test_fused_exchange_preserves_signed_particle_and_gas_inventories():
    """Match all per-particle and gas masses after births and half steps."""

    def make(method):
        return TrialBox(
            "physical", 105, None, None, 1031, SPECIES, method=method, heads={}
        )

    reference, optimized = make("physical"), make("fused")
    for box in (reference, optimized):
        for _ in range(3):
            box.vapor(DT / 2, 0)
            box.inject(DT)
            box.collide(DT)
            box.vapor(DT / 2, 0)
            box.observe()
    assert reference.births == optimized.births
    assert reference.collisions == optimized.collisions
    torch.testing.assert_close(
        optimized.mass, reference.mass, rtol=1e-12, atol=0
    )
    torch.testing.assert_close(optimized.gas, reference.gas, rtol=1e-12, atol=0)


@torch.inference_mode()
def test_residual_step_reverts_entire_species_on_negative_proposal():
    """Retain a closed, nonnegative solution even for a poor fitted weight."""

    def make():
        return TrialBox(
            "physical",
            13,
            None,
            None,
            127,
            SPECIES,
            method="learned-step",
            heads={"step": 1e4},
        )

    box, fine = make(), make()
    coarse_m, coarse_g = fine.proposals(DT / 2)
    m = fine.mass[:, 1:]
    bad = ((m + 1e4 * (m - coarse_m[:, 1:])).amin(0) < 0) | (
        fine.gas + 1e4 * (fine.gas - coarse_g) < 0
    )
    assert bool(bad.any())
    box.vapor(DT / 2, 0)
    torch.testing.assert_close(
        box.mass[:, 1:][:, bad], m[:, bad], rtol=0, atol=0
    )
    torch.testing.assert_close(box.gas[bad], fine.gas[bad], rtol=0, atol=0)
    assert bool((box.mass >= 0).all() & (box.gas >= 0).all())
    box.observe()
