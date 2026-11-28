"""Recover CAMx/PMCAMx stoichiometry and kinetic orders from CMC tables."""

import re
from collections import Counter, defaultdict

from particula.dynamics.chemistry.catalog.readers import NUMBER, number


def _gain_loss(source):
    stoich: defaultdict[str, tuple[dict[str, float], dict[str, float]]] = (
        defaultdict(lambda: ({}, {}))
    )
    source = re.sub(r"\n     &", "", source)
    assignment = r"^\s*(Gain|Loss)\(l(\w+)\s*\)\s*=\s*(.*)$"
    pattern = rf"\s*\+\s*(?:\(\s*({NUMBER})\s*\)\s*\*\s*)?r\(\s*(\d+)\s*\)"
    for op, name, rhs in re.findall(assignment, source, re.M | re.I):
        rhs = re.sub(rf"^{op}\(l{name}\s*\)\s*", "", rhs, flags=re.I)
        if re.fullmatch(r"\s*0(?:\.0*)?(?:d0)?\s*", rhs, re.I):
            continue
        if re.sub(pattern, "", rhs, flags=re.I).strip():
            raise ValueError(f"unsupported CMC {op} expression for {name}")
        for n, label in re.findall(pattern, rhs, re.I):
            terms = stoich[label][int(op.lower() == "gain")]
            terms[name] = terms.get(name, 0) + number(n or "1")
    return stoich


def _orders(source):
    orders = {}
    for label, rhs in re.findall(
        r"^\s*r\(\s*(\d+)\)\s*=\s*(.*)$", source, re.M | re.I
    ):
        factors = rhs.replace(" ", "").split("*")
        if factors[0].lower() != f"dbrk({label})" or label in orders:
            raise ValueError(f"unsupported CMC flux for {label}")
        names = []
        for term in factors[1:]:
            found = re.fullmatch(r"y\(l(\w+)\)", term, re.I)
            if found:
                names.append(found[1])
            elif re.fullmatch(r"M|H2O|O2|CH4|H2|N2", term):
                names.append(term)
            else:
                raise ValueError(f"unsupported CMC factor {term}")
        orders[label] = dict(Counter(names))
    return orders


def read_cmc(companions, expected=None):
    """Read literal CMC assignments from a matched gain/loss and flux pair."""
    tables = {}
    numbers = set()
    for source in companions:
        m = re.search(
            r"^\s*subroutine\s+(lsrate|lsrxn)(\d+)\(", source, re.I | re.M
        )
        if m:
            name = m[1].lower()
            if name in tables:
                raise ValueError(f"multiple CMC {name} tables")
            tables[name] = source
            numbers.add(m[2])
    if not tables:
        return {}
    if (
        len(numbers) != 1
        or len(tables) != 2
        or (expected and expected not in numbers)
    ):
        raise ValueError("supply matching CMC gain/loss and flux files")
    stoich = _gain_loss(tables["lsrate"])
    orders = _orders(tables["lsrxn"])
    if not stoich or stoich.keys() - orders.keys():
        raise ValueError("CMC tables reference different reaction IDs")
    return {k: (*stoich[k], v) for k, v in orders.items()}
