"""Regression tests for the Manchester network admission rules."""

import pytest
from proposals import balanced_accretion
from selection import choose, native_threshold


def test_accretion_preserves_precursor_equivalents():
    """Check source labels independently against the archived pool map."""
    products = {11: [0, 0], 12: [0, 1], 14: [1, 1], 15: [2, 2]}
    proposals = list(balanced_accretion())
    assert proposals
    for reaction in proposals:
        assert (
            sorted([reaction.a % 3, reaction.b % 3]) == products[reaction.dest]
        )
        assert (reaction.a < 6) == (reaction.b < 6)


@pytest.mark.parametrize(
    ("child_scores", "winner"),
    [
        ([0.147, 0.150], "parent"),  # Forward-only gain.
        ([0.1495, 0.1495], "parent"),  # Smaller than restart uncertainty.
        ([0.148, 0.148], "child"),  # Both orders clear the 0.001 gate.
    ],
)
def test_paired_child_competes_with_best_simpler_parent(child_scores, winner):
    """A weak paired refit cannot grant admission over a stronger parent."""

    def trial(name, scores, ids, gate=0):
        return dict(
            name=name,
            qualification_scores=scores,
            ids=ids,
            eligible=True,
            threshold=gate,
        )

    choices = [
        trial("weaker parent", [0.20, 0.20], [0]),
        trial("parent", [0.15, 0.15], [0]),
        trial("child", child_scores, [0, 1], 0.001),
    ]
    best_parent, selected = choose(choices, [0])
    assert best_parent["name"] == "parent"
    assert selected["name"] == winner


def test_child_restart_variation_can_dominate_numerical_shift():
    """Both sides of a nested fit contribute to the admission threshold."""
    pair = dict(
        numerical=0.002,
        parent={"starts": [0.2, 0.2]},
        child={"starts": [0.19, 0.17]},
    )
    assert native_threshold(pair) == pytest.approx(0.02)
