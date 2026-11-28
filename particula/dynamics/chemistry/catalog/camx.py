"""Read CAMx numbered rate decks and PMCAMx-SR precursor yield tables."""

import re

from particula.dynamics.chemistry.catalog.readers import (
    NUMBER,
    Entry,
    channel,
    number,
)
from particula.dynamics.chemistry.catalog.sources import recipes

# Reaction 3 in these exact molecular-unit decks is ozone titration.
# See CAMx v7.20 Guide, CB6r5 listing, and the Zenodo 4071362 CB05
# ierxn7/lsrate7 flux and production routines. Never infer an equation
# from a row number in a different version or user-modified mechanism.
TITRATION_DECKS = {
    "camx": {
        "8d267b966a4cf3d9be236f4bbed5fdd52bd3cfe3e463f97c454aa7a9807bfe25"
    },
    "pmcamx": {
        "d0a18f9beff7538e20238220e7f57dbc3a49e7ceb134001db6d49b6b4cd1fec9"
    },
}


def equations(text):
    """Read a CAMx mech.* stoichiometry listing and continuation lines."""
    if not re.match(r"\s*1\s+\w+[^\n]*=", text):
        return {}
    rows = {}
    current = ""
    for line in text.splitlines():
        if not line.strip():
            continue
        match = re.match(r"\s*(\d+)\s+(.*?)\s*=\s*(.*)$", line)
        if match:
            current, lhs, rhs = match.groups()
            if current in rows:
                raise ValueError(f"duplicate CAMx equation {current}")
            rows[current] = (lhs.split(), rhs.split())
        elif current and "=" not in line:
            rows[current][1].extend(line.split())
        else:
            raise ValueError("invalid CAMx equation continuation")
    result = {}
    for label, (reactants, products) in rows.items():
        if len(products) % 2:
            raise ValueError(f"incomplete CAMx product list for {label}")
        rhs = [
            f"{c}*{s}"
            for c, s in zip(products[::2], products[1::2], strict=True)
        ]
        result[label] = " + ".join(reactants) + " -> " + " + ".join(rhs)
    return result


def _deck_header(lines):
    header = {}
    photo = set()
    count_secondary = 0
    for i, line in enumerate(lines):
        if line.strip().lower() == "reaction records":
            if i + 1 >= len(lines):
                raise ValueError("missing CAMx rate columns")
            break
        if "|" not in line:
            continue
        key, val = (t.strip().lower() for t in line.split("|", 1))
        header[key] = val
        if "prim photo" in key:
            count, *ids = val.split()
            if len(ids) != int(count):
                raise ValueError("invalid primary photolysis list")
            photo.update(ids)
        elif "sec photo" in key:
            count_secondary = int(val)
        elif count_secondary and key in {"id, prim id, scale", ""}:
            photo.add(val.split()[0])
            count_secondary -= 1
    else:
        raise ValueError("missing CAMx Reaction Records")
    if count_secondary or not lines[i + 1].lower().split()[:3] == [
        "rxn",
        "typ",
        "order",
    ]:
        raise ValueError("incomplete CAMx header or unsupported rate columns")
    return i + 2, header, photo


def _deck_rate(kind, values):
    if kind == "1" and len(values) == 1:
        kinetics = (number(values[0]), 0, 0, 300)
    elif kind == "3" and len(values) == 4:
        a, e, b, tref = map(number, values)
        kinetics = (a, -e, b, tref)
    else:
        kinetics = None
    return kinetics


def _deck_units(pinned, header):
    # See the deck-level verification in plans/chemical-channel-catalog.md.
    if pinned:
        return "molecule/cm^3", "s", "verified source deck SHA-256"
    clauses = re.split(r"[:;,]", header.get("description", ""))
    declarations = {clause.strip().rstrip(".") for clause in clauses}
    declared = declarations & {"molecular units", "ppm units"}
    if declared == {"molecular units"}:
        return "molecule/cm^3", "s", "source Description: molecular units"
    if declared == {"ppm units"}:
        return "ppm", "min", "source Description: ppm units"
    return None, None, "no unambiguous source unit declaration"


def _deck_entry(raw, line_num, pinned, units, photo, listings, tables):
    label, kind, order, *values = raw.split()
    eq = listings.get(label, "")
    if pinned and label == "3" and not listings:
        eq = "O3 + NO -> NO2"
    kinetics = _deck_rate(kind, values)
    concentration, time, evidence = units
    reason = {
        "molecule/cm^3": "",
        "ppm": "requires CAMx ppm/min rate conversion before native export",
        None: "source concentration and time units require verification",
    }[concentration]
    if kinetics is None:
        reason = "requires CAMx compound or host-defined rate"
    if label in photo:
        reason = "requires a source photolysis frequency"
    if label in tables:
        lhs, rhs, orders = tables[label]
        sides = [
            " + ".join(f"{v:g}*{s}" for s, v in terms.items())
            for terms in (lhs, rhs)
        ]
        eq = " -> ".join(sides)
        if orders != lhs:
            reason = "requires host kinetic orders or fixed reservoirs"
    if not eq:
        reason = "requires matching CAMx stoichiometry listing"
    entry = channel(
        label,
        line_num,
        eq or " -> ",
        raw.strip(),
        kinetics,
        reason,
        kinetic_order=int(order),
        rate_type=kind,
        source_rate_units=dict(
            concentration_unit=concentration, time_unit=time, evidence=evidence
        ),
    )
    if label in tables:
        entry.metadata["orders"] = tables[label][2]
    return entry


def rate_deck(text, source, digest, listings, tables):
    """Join numbered rate rows to an optional matching mechanism listing."""
    lines = text.splitlines()
    start, header, photo = _deck_header(lines)
    count = int(header.get("num reactions", header.get("no of reactions", "0")))
    labels = {str(k) for k in range(1, count + 1)}
    if tables.keys() - labels or (listings and listings.keys() != labels):
        raise ValueError("CAMx equation and rate record IDs differ")
    pinned = digest in TITRATION_DECKS[source]
    units = _deck_units(pinned, header)
    seen = set()
    for line_num, raw in enumerate(lines[start:], start + 1):
        if not raw.strip():
            continue
        label = raw.split()[0]
        if label in seen or label not in labels:
            raise ValueError(f"invalid CAMx rate ID {label}")
        seen.add(label)
        yield _deck_entry(raw, line_num, pinned, units, photo, listings, tables)
    if seen != labels or not seen:
        raise ValueError("incomplete CAMx reaction deck")


def _soap_tables(text):
    tables = {}
    for name, width in (
        ("rkpar", 2),
        ("lnoxval", 4),
        ("hnoxval", 4),
        ("idCG", 4),
    ):
        match = re.search(rf"^\s*data\s+{name}\s*/", text, re.I | re.M)
        if not match:
            raise ValueError(f"missing PMCAMx SOAP data array {name}")
        start = text.count("\n", 0, match.end()) + 1
        rows = []
        for line_num, line in enumerate(
            text[match.end() :].splitlines(), start
        ):
            values, _, label = line.partition("!")
            end = "/" in values
            vals = [
                number(v.strip())
                for v in values.strip().strip("&/,").strip().split(",")
                if v.strip()
            ]
            if len(vals) != width:
                raise ValueError(
                    f"unexpected {name} array width at line {line_num}"
                )
            rows.append((line_num, vals, label.strip()))
            if end:
                break
        tables[name] = rows
    return tables


def soap(text, digest=None):
    """Inventory PMCAMx-SR SOAP precursor rates and paired NOx yield tables."""
    published = recipes()
    verified = digest in {
        published[key]["files"][0]["sha256"]
        for key in ("pmcamx-soap", "pmcamx-soap-alt")
    }
    tables = _soap_tables(text)
    for idx, (rate, low, high, ids) in enumerate(
        zip(*tables.values(), strict=True), 1
    ):
        marker = re.fullmatch(
            r"rxn\s+(\d+):\s*(\w+)\s*\+\s*(\w+)", rate[2], re.I
        )
        if not marker or int(marker[1]) != idx:
            raise ValueError("missing SOAP precursor/oxidant labels")
        bin_ids = [int(i) for i in ids[1]]
        if any(
            i != orig or not 1 <= i <= 10
            for i, orig in zip(bin_ids, ids[1], strict=True)
        ):
            raise ValueError("unknown SOAP product bin")
        bins = [f"CAS{i}" if i <= 5 else f"CBS{i - 5}" for i in bin_ids]
        products = list(
            dict.fromkeys(
                name
                for name, lo, hi in zip(bins, low[1], high[1], strict=True)
                if lo != 0 or hi != 0
            )
        )
        entry = Entry(
            f"SOAP{idx}",
            rate[0],
            f"{marker[2]} + {marker[3]} -> " + ", ".join(products),
            f"{rate[1][0]} * exp(-({rate[1][1]})/T)",
            {marker[2]: 1, marker[3]: 1},
            {},
            None,
            "requires prescribed oxidants and NOx-dependent SOAP yields",
            dict(
                product_bins=products,
                yield_bins=bins,
                low_nox_yields=low[1],
                high_nox_yields=high[1],
                yield_basis="source lnoxval/hnoxval before density scaling",
            ),
        )
        if verified:
            entry.metadata.update(
                prefactor_unit="cm^3/molecule/s",
                activation_temperature_unit="K",
                yield_unit="ppm/ppm",
                unit_evidence="archived SOAP/soachem.f",
            )
        yield entry
    for entry in _soap_aging(text):
        if verified:
            entry.metadata.update(
                host_rate_unit="1/ppm/h",
                oxidant_unit="ppm",
                yield_unit="ppm/ppm",
                unit_evidence="archived SOAP/soachem.f",
            )
        yield entry


def _soap_aging(text):
    """Read the literal bin gain terms in the published SOAP aging loops."""
    start = text.find("c-----Calculate and Apply Gain and Loss")
    if start < 0:
        return
    disabled = {
        int(i)
        for i in re.findall(
            r"^\s*dcCG\((\d+)\)\s*=\s*0\s*(?:!.*)?$", text[:start], re.M | re.I
        )
    }
    loops = re.finditer(
        r"^\s*do iorg\s*=\s*(\d+),(\d+)\s*\n(.*?)^\s*enddo",
        text[start:],
        re.M | re.S | re.I,
    )
    for loop in loops:
        lo, hi, body = loop.groups()
        rhs = re.sub(r"!.*", "", body)
        rhs = re.sub(r"\s+|&", "", rhs)
        base = "cCG(iorg)=cCG(iorg)-dcCG(iorg)"
        if not rhs.startswith(base):
            raise ValueError("unrecognized SOAP aging assignment")
        gain = rhs[len(base) :]
        term = rf"\+\(({NUMBER})\)\*dcCG\(iorg\+1(-8)?\)"
        if re.sub(term, "", gain):
            raise ValueError("unrecognized SOAP bin gain")
        for factor, offset in re.findall(term, gain):
            for target in range(int(lo), int(hi) + 1):
                origin = target + 1 + int(offset or 0)
                if origin in disabled:
                    continue
                source_bin, target_bin = f"CG{origin}", f"CG{target}"
                rate = (
                    "rkasoa"
                    if origin <= 5
                    else "rkbsoa"
                    if origin <= 10
                    else "rkpoa"
                )
                yield Entry(
                    f"aging-{origin}-{target}",
                    text.count("\n", 0, start + loop.start(3)) + 1,
                    f"{source_bin} -> {factor}*{target_bin}",
                    f"{rate} * avgox(2)",
                    {source_bin: 1},
                    {target_bin: number(factor)},
                    None,
                    "requires SOAP bin inventory and prescribed OH exposure",
                    dict(
                        kind="aging",
                        yield_basis="source gas-bin mixing ratio",
                        oxidant="OH",
                        host_rate=rate,
                    ),
                )
