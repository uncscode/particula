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


def test_discovered_bimolecular_reaction_executes():
    """Execute the generated A+B→C candidate and check its analytic decay."""
    import numpy as np
    from particula.dynamics.chemistry.discovery import (
        balanced_steps,
        compile_network,
    )
    from particula.dynamics.chemistry.process import (
        ChemistrySolver,
        GasChemistry,
    )
    from particula.dynamics.chemistry.tests.process_test import (
        state_and_process,
    )

    state, _ = state_and_process("arrhenius", "rk23")
    formulas = [(1, 0), (0, 1), (1, 1)]
    steps = balanced_steps(["gas"] * 3, formulas)
    found = enumerate_networks(
        steps,
        [True, True, False],
        [2],
        net=(-1, -1, 1),
        max_steps=1,
        max_trials=10,
        max_candidates=2,
    )
    (candidate,) = found.networks
    mechanism = compile_network(
        candidate, {f"gas:{s}": {} for s in "ABC"}, [2.0]
    )
    model = GasChemistry(mechanism.bind(state.gas), ChemistrySolver(rtol=1e-8))
    model.execute(state, time_step=0.5)
    # Equimolar reactants, c(t) = c0 / (1 + k c0 t).
    np.testing.assert_allclose(
        state.gas.concentration[:, 0] / 0.01, [0.5, 2 / 3], rtol=2e-6
    )


def test_condensation_makes_particle_intermediate_reachable():
    """Vapor produced by A supplies particle aging via a reversible pair."""
    options = dict(
        max_steps=2, max_trials=10, max_candidates=5, exchange_pairs=[(1, 2)]
    )
    search = enumerate_networks(
        [(-1, 1, 0, 0), (0, 0, -1, 1)], [1, 0, 0, 0], [3], **options
    )
    assert len(search.networks) == 1 and len(search.networks[0]) == 2
    assert not enumerate_networks(
        [(0, 0, -1, 1)], [1, 0, 0, 0], [3], **options
    ).networks


def test_intermediate_budget_keeps_direct_parent():
    """A zero hidden-species budget retains a nested direct control."""
    result = enumerate_networks(
        [(-1, 0, 1), (-1, 1, 0), (0, -1, 1)],
        [1, 0, 0],
        [2],
        net=(-1, 0, 1),
        max_steps=2,
        max_trials=10,
        max_candidates=4,
        max_intermediates=0,
    )
    assert result.networks == (((-1, 0, 1),),)
