"""Reachability, hidden-label symmetry and explicit search limits."""

from particula.dynamics.chemistry.discovery import enumerate_networks


def test_hidden_chain_symmetry_and_reachability():
    """A direct conversion and two equivalent hidden routes give two models."""
    steps = [
        (-1, 1, 0, 0),
        (-1, 0, 1, 0),
        (-1, 0, 0, 1),
        (0, 1, -1, 0),
        (0, 1, 0, -1),
        (0, 0, -1, 1),
        (0, 0, 1, -1),
    ]
    kwargs = dict(
        max_steps=3,
        max_trials=100,
        max_candidates=10,
        net=(-1, 1, 0, 0),
        interchangeable=(2, 3),
    )
    result = enumerate_networks(
        steps, [True, False, False, False], [1], **kwargs
    )
    assert [len(n) for n in result.networks] == [1, 2, 3]
    assert result.limit is None
    for field, limit in [
        ("max_trials", "trials"),
        ("max_candidates", "candidates"),
    ]:
        bounded = enumerate_networks(
            steps, [True, False, False, False], [1], **{**kwargs, field: 1}
        )
        assert bounded.limit == limit


def test_balanced_dimer_search():
    """Two gas monomers can form their dimer but cannot form particle mass."""
    from particula.dynamics.chemistry.discovery import balanced_steps

    assert balanced_steps(["gas", "gas", "particle"], [(1,), (2,), (2,)]) == (
        (-2, 1, 0),
        (2, -1, 0),
    )
