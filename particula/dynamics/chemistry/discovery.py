"""Bounded stoichiometric searches for hidden kinetic intermediates.

Rows encode net molecular coefficients of irreversible elementary reactions.
The overall-stoichiometry mode follows de Carvalho Servia et al.
(doi:10.1039/D5SC01473E): each selected step contributes once to the net
reaction. Competing branches can instead be searched without a net target.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import combinations, permutations
from operator import index

from particula.dynamics.chemistry.mechanism import ReactionMechanism

Step = tuple[int, ...]
Network = tuple[Step, ...]


@dataclass(frozen=True)
class SearchResult:
    """Unique networks and whether a work or result limit stopped search."""

    networks: tuple[Network, ...]
    examined: int
    limit: str | None


def enumerate_networks(
    steps: Iterable[Step],
    initial: Sequence[bool],
    products: Sequence[int],
    *,
    max_steps: int,
    max_trials: int,
    max_candidates: int,
    net: Step | None = None,
    max_intermediates: int | None = None,
    interchangeable: Sequence[int] = (),
    exchange_pairs: Sequence[tuple[int, int]] = (),
) -> SearchResult:
    """Search combinations of allowed, prebalanced elementary steps.

    Columns share a caller-defined species order. Initial flags indicate
    substrates available at the start. Every required product must be made
    and every selected reaction must be reachable. Supply interchangeable
    indices only for hidden species with identical physical constraints.
    max_intermediates bounds used columns other than initial species and
    specified products. Reversible exchange pairs link phases once either
    phase has a source. They add no fitted reaction or net coefficient.
    Limits count examined combinations and unique returned networks.
    """
    rows, required, hidden, target = _validate_search(
        steps,
        initial,
        products,
        interchangeable,
        net,
        (max_steps, max_trials, max_candidates),
    )
    links, hidden_columns, hidden_limit = _search_constraints(
        initial, required, hidden, exchange_pairs, max_intermediates
    )
    found: dict[Network, Network] = {}
    examined = 0
    for count in range(1, min(max_steps, len(rows)) + 1):
        for network in combinations(rows, count):
            if examined == max_trials:
                return SearchResult(tuple(found.values()), examined, "trials")
            examined += 1
            if (
                target is not None
                and tuple(map(sum, zip(*network, strict=True))) != target
            ):
                continue
            if (
                sum(any(row[i] for row in network) for i in hidden_columns)
                > hidden_limit
            ):
                continue
            if not _reachable(network, initial, required, links):
                continue
            key = _canonical(network, hidden)
            if key not in found:
                if len(found) == max_candidates:
                    return SearchResult(
                        tuple(found.values()), examined, "candidates"
                    )
                found[key] = network
    return SearchResult(tuple(found.values()), examined, None)


def _search_constraints(
    initial, required, hidden, exchange_pairs, max_intermediates
):
    links = tuple((index(a), index(b)) for a, b in exchange_pairs)
    if any(
        min(a, b) < 0 or max(a, b) >= len(initial) or a == b for a, b in links
    ):
        raise ValueError("Invalid exchange species indices")
    if any(a in hidden or b in hidden for a, b in links):
        raise ValueError("Exchange-linked species require fixed labels")
    hidden_columns = tuple(
        i
        for i, present in enumerate(initial)
        if not present and i not in required
    )
    hidden_limit = (
        len(hidden_columns)
        if max_intermediates is None
        else index(max_intermediates)
    )
    if hidden_limit < 0:
        raise ValueError("Intermediate limit must be nonnegative")
    return links, hidden_columns, hidden_limit


def _validate_search(steps, initial, products, interchangeable, net, bounds):
    """Normalize integer coefficients and validate the search domains."""
    width = len(initial)
    bounds = tuple(map(index, bounds))
    if not width or min(bounds) < 1:
        raise ValueError("Species and positive search limits are required")
    rows = sorted({tuple(map(index, row)) for row in steps})
    if any(len(row) != width or min(row) >= 0 or max(row) <= 0 for row in rows):
        raise ValueError("Each step needs reactants and products")
    hidden = tuple(map(index, interchangeable))
    required = tuple(map(index, products))
    if any(i < 0 or i >= width for i in (*hidden, *required)):
        raise ValueError("Invalid species index")
    if len(set(hidden)) != len(hidden) or any(
        initial[i] or i in required for i in hidden
    ):
        raise ValueError("Interchangeable species must be distinct and hidden")
    target = None if net is None else tuple(map(index, net))
    if target is not None and (
        len(target) != width or any(target[i] for i in hidden)
    ):
        raise ValueError("Invalid overall stoichiometry")
    return rows, required, hidden, target


def _reachable(network, initial, required, links):
    """Check that the initial substrates can supply all selected steps."""
    available = {i for i, present in enumerate(initial) if present}
    made = {i for row in network for i, v in enumerate(row) if v > 0}
    _exchange_closure(made, links)
    if not made.issuperset(required):
        return False
    remaining = list(network)
    while remaining:
        _exchange_closure(available, links)
        reachable = [
            row
            for row in remaining
            if all(i in available for i, v in enumerate(row) if v < 0)
        ]
        if not reachable:
            break
        for row in reachable:
            available.update(i for i, v in enumerate(row) if v > 0)
            remaining.remove(row)
    return not remaining


def _exchange_closure(available, links):
    """Follow reversible links including chains sharing a species."""
    size = -1
    while size != len(available):
        size = len(available)
        for a, b in links:
            if a in available or b in available:
                available.update((a, b))


def _canonical(network: Network, hidden: tuple[int, ...]) -> Network:
    """Canonicalize labels of participating equivalent intermediates."""
    active = [i for i in hidden if any(row[i] for row in network)]
    fixed = [i for i in range(len(network[0])) if i not in hidden]
    return min(
        tuple(
            sorted(tuple(row[i] for i in (*fixed, *order)) for row in network)
        )
        for order in permutations(active)
    )


def balanced_steps(
    phases: Sequence[str],
    composition: Sequence[Sequence[int]],
) -> tuple[Step, ...]:
    """Enumerate unimolecular and bimolecular steps within each phase.

    Composition rows contain nonnegative atom counts (or conserved pool
    units) per species. Columns share an explicitly declared basis. Each
    reaction has one or two molecules on each side and conserves this basis.
    Gas-particle exchange is handled separately by the condensation process.
    """
    from itertools import combinations_with_replacement

    atoms = [tuple(map(index, counts)) for counts in composition]
    if not atoms or len(atoms) != len(phases) or not atoms[0]:
        raise ValueError("One composition vector per species is required")
    if any(
        len(row) != len(atoms[0]) or min(row) < 0 or not any(row)
        for row in atoms
    ):
        raise ValueError("Inconsistent or empty species composition")
    groups: dict[tuple[str, tuple[int, ...]], list[tuple[int, ...]]] = {}
    for phase in set(phases):
        if phase not in {"gas", "particle"}:
            raise ValueError("Unsupported species phase")
        indices = [i for i, p in enumerate(phases) if p == phase]
        for size in (1, 2):
            for side in combinations_with_replacement(indices, size):
                budget = tuple(
                    map(sum, zip(*(atoms[i] for i in side), strict=True))
                )
                groups.setdefault((phase, budget), []).append(side)
    steps = set()
    for sides in groups.values():
        for left, right in permutations(sides, 2):
            if set(left).isdisjoint(right):
                steps.add(
                    tuple(
                        right.count(i) - left.count(i)
                        for i in range(len(phases))
                    )
                )
    return tuple(sorted(steps))


def compile_network(
    network: Network,
    species: dict[str, dict],
    rates: Sequence[float],
) -> ReactionMechanism:
    """Build a native mechanism from elementary steps and rate coefficients.

    The insertion order of the phase-qualified species catalog is the matrix
    column order. Each rate is in inverse seconds for a unimolecular step or
    cubic metres per mole per second for a bimolecular step. Gas and particle
    rates use their respective phase volumes. Binding the returned mechanism
    to an aerosol state performs the native mass and elemental balance checks.
    """
    names = tuple(species)
    resolved = any(name.startswith("particle:") for name in names)
    reactions = []
    for number, (row, rate) in enumerate(zip(network, rates, strict=True)):
        terms = dict(zip(names, map(index, row), strict=True))
        left = {name: -n for name, n in terms.items() if n < 0}
        right = {name: n for name, n in terms.items() if n > 0}
        order = sum(left.values())
        if order not in (1, 2) or sum(right.values()) not in (1, 2):
            raise ValueError(
                "Discovery requires elementary uni/bimolecular steps"
            )
        reactions.append(
            {
                "id": f"r{number}",
                "reactants": left,
                "products": right,
                "rate": {
                    "type": "constant",
                    "k": float(rate),
                    "units": "1/s" if order == 1 else "m^3/mol/s",
                },
            }
        )
    return ReactionMechanism.from_dict(
        {
            "version": 2 if resolved else 1,
            "basis": "mol/m^3-phase" if resolved else "mol/m^3-air",
            "species": species,
            "reactions": reactions,
        }
    )
