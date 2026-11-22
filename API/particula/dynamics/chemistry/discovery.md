# particula.dynamics.chemistry.discovery

*module*

Bounded stoichiometric searches for hidden kinetic intermediates.

Rows encode net molecular coefficients of irreversible elementary reactions.
The overall-stoichiometry mode follows de Carvalho Servia et al.
(doi:10.1039/D5SC01473E): each selected step contributes once to the net
reaction. Competing branches can instead be searched without a net target.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/discovery.py#L1-L1)

## particula.dynamics.chemistry.discovery.Network

*attribute* · *module attribute*

```python
Network = tuple[Step, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/discovery.py#L17-L17)

## particula.dynamics.chemistry.discovery.Step

*attribute* · *module attribute*

```python
Step = tuple[int, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/discovery.py#L16-L16)

## particula.dynamics.chemistry.discovery.SearchResult

*class* · *dataclass*

```python
class SearchResult
```

Unique networks and whether a work or result limit stopped search.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/discovery.py#L20-L26)

### particula.dynamics.chemistry.discovery.SearchResult.examined

*attribute* · *instance attribute*

```python
examined: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/discovery.py#L25-L25)

### particula.dynamics.chemistry.discovery.SearchResult.limit

*attribute* · *instance attribute*

```python
limit: str | None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/discovery.py#L26-L26)

### particula.dynamics.chemistry.discovery.SearchResult.networks

*attribute* · *instance attribute*

```python
networks: tuple[Network, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/discovery.py#L24-L24)

### particula.dynamics.chemistry.discovery.SearchResult.__init__

*method*

```python
def __init__(networks: tuple[Network, ...], examined: int, limit: str | None) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/discovery.py#L0-L0)

## particula.dynamics.chemistry.discovery._canonical

*function*

```python
def _canonical(network: Network, hidden: tuple[int, ...]) -> Network
```

Canonicalize labels of participating equivalent intermediates.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/discovery.py#L176-L185)

## particula.dynamics.chemistry.discovery._exchange_closure

*function*

```python
def _exchange_closure(available, links)
```

Follow reversible links including chains sharing a species.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/discovery.py#L166-L173)

## particula.dynamics.chemistry.discovery._reachable

*function*

```python
def _reachable(network, initial, required, links)
```

Check that the initial substrates can supply all selected steps.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/discovery.py#L143-L163)

## particula.dynamics.chemistry.discovery._search_constraints

*function*

```python
def _search_constraints(initial, required, hidden, exchange_pairs, max_intermediates)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/discovery.py#L93-L115)

## particula.dynamics.chemistry.discovery._validate_search

*function*

```python
def _validate_search(steps, initial, products, interchangeable, net, bounds)
```

Normalize integer coefficients and validate the search domains.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/discovery.py#L118-L140)

## particula.dynamics.chemistry.discovery.balanced_steps

*function*

```python
def balanced_steps(phases: Sequence[str], composition: Sequence[Sequence[int]]) -> tuple[Step, ...]
```

Enumerate unimolecular and bimolecular steps within each phase.

Composition rows contain nonnegative atom counts (or conserved pool
units) per species. Columns share an explicitly declared basis. Each
reaction has one or two molecules on each side and conserves this basis.
Gas-particle exchange is handled separately by the condensation process.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/discovery.py#L188-L230)

## particula.dynamics.chemistry.discovery.compile_network

*function*

```python
def compile_network(network: Network, species: dict[str, dict], rates: Sequence[float]) -> ReactionMechanism
```

Build a native mechanism from elementary steps and rate coefficients.

The insertion order of the phase-qualified species catalog is the matrix
column order. Each rate is in inverse seconds for a unimolecular step or
cubic metres per mole per second for a bimolecular step. Gas and particle
rates use their respective phase volumes. Binding the returned mechanism
to an aerosol state performs the native mass and elemental balance checks.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/discovery.py#L233-L277)

## particula.dynamics.chemistry.discovery.enumerate_networks

*function*

```python
def enumerate_networks(steps: Iterable[Step], initial: Sequence[bool], products: Sequence[int], *, max_steps: int, max_trials: int, max_candidates: int, net: Step | None = None, max_intermediates: int | None = None, interchangeable: Sequence[int] = (), exchange_pairs: Sequence[tuple[int, int]] = ()) -> SearchResult
```

Search combinations of allowed, prebalanced elementary steps.

Columns share a caller-defined species order. Initial flags indicate
substrates available at the start. Every required product must be made
and every selected reaction must be reachable. Supply interchangeable
indices only for hidden species with identical physical constraints.
max_intermediates bounds used columns other than initial species and
specified products. Reversible exchange pairs link phases once either
phase has a source. They add no fitted reaction or net coefficient.
Limits count examined combinations and unique returned networks.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/discovery.py#L29-L90)
