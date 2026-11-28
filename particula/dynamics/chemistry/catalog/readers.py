"""Read source mechanisms as data; preserve source expressions and locations."""

import hashlib
import re
from collections import Counter
from dataclasses import asdict, dataclass, field

NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?"
SOURCES = ("waccm", "e3sm", "mcm", "wrf-chem", "cmaq", "camx", "pmcamx")


@dataclass
class Entry:
    """One source channel; kinetics are (A, C, b, Tref) for exp(C/T)."""

    label: str
    line: int
    equation: str
    expression: str
    reactants: dict
    products: dict
    kinetics: tuple | None
    reason: str = ""
    metadata: dict = field(default_factory=dict)

    def record(self):
        """Return a JSON-compatible source record."""
        record = asdict(self)
        if self.kinetics:
            units = self.metadata.get("source_rate_units", {})
            order = self.metadata.get(
                "kinetic_order", sum(self.reactants.values())
            )
            record["rate_model"] = {
                "formula": "A * (T/Tref)**b * exp(C/T)",
                "temperature_unit": "K",
                "concentration_unit": units.get(
                    "concentration_unit", "molecule/cm^3"
                ),
                "time_unit": units.get("time_unit", "s"),
                "order": order,
                **dict(
                    zip(("A", "C", "b", "Tref"), self.kinetics, strict=True)
                ),
            }
        return record


def number(value):
    """Convert atmospheric-model Fortran numeric literals."""
    return float(value.lower().replace("d", "e"))


def _terms(side):
    terms: dict[str, float] = {}
    # Keep negative diagnostic coefficients, such as CRACMM's DELTA_C.
    side = re.sub(r"(?<=[\d.])([eEdD])\+(?=\d)", r"\1", side)
    for term in re.split(r"\s*\+\s*", re.sub(r"\s+-\s+", " + -", side)):
        if not term.strip():
            continue
        match = re.fullmatch(
            rf"(?:({NUMBER})(?:\s*\*\s*|\s+))?([A-Za-z][\w]*)",
            term.strip(),
        )
        if not match:
            raise ValueError(f"unrecognized species term {term!r}")
        coefficient, species = match.groups()
        terms[species] = terms.get(species, 0) + number(coefficient or "1")
    return terms


def channel(label, line, equation, rate, kinetics=None, reason="", **metadata):
    """Keep the original equation even when its terms need a new dialect."""
    try:
        plain = re.sub(r"\{[^{}]*\}", "", equation)
        if plain != equation:
            metadata["annotated_equation"] = True
        left, right = re.split(r"->|=", plain)
    except ValueError as exc:
        return Entry(
            label, line, equation, rate, {}, {}, kinetics, str(exc), metadata
        )
    sides = []
    for side in (left, right):
        try:
            sides.append(_terms(side))
        except ValueError as exc:
            sides.append({})
            reason = str(exc)
    lhs, rhs = sides
    return Entry(
        label, line, equation, rate, lhs, rhs, kinetics, reason, metadata
    )


def _mozart_rate(section, rate):
    kinetics = None
    issue = ""
    if section == "photolysis":
        issue = "requires a source photolysis frequency"
    elif re.fullmatch(rf"\s*{NUMBER}(?:\s*,\s*{NUMBER})?\s*", rate):
        vals = [number(x) for x in rate.split(",")]
        kinetics = (vals[0], vals[1] if len(vals) == 2 else 0, 0, 300)
    elif rate.strip():
        issue = "requires source falloff or compound rate law"
    else:
        issue = "requires a host-defined rate routine"
    return kinetics, issue


def mozart(text):
    """Read MOZART preprocessor photolysis, kinetic and fixed-species blocks."""
    section = ""
    fixed = set()
    for line, raw in enumerate(text.splitlines(), 1):
        row = raw.strip()
        if not row or row.startswith("*"):
            continue
        key = row.lower()
        if key in {"fixed", "reactions", "photolysis"}:
            section = key
        elif key.startswith("end "):
            section = ""
        elif section == "fixed":
            fixed.update(re.findall(r"[\w]+", row))
        elif section in {"reactions", "photolysis"}:
            match = re.fullmatch(r"\[([^\]]+)\]\s*(.*)", row)
            if not match:
                raise ValueError(
                    f"line {line}: expected a labeled MOZART channel"
                )
            tag, rest = match.groups()
            eq, _, rate = rest.partition(";")
            label = re.split(r"->|,|=", tag)[0].strip()
            kinetics, issue = _mozart_rate(section, rate)
            yield channel(
                label,
                line,
                eq.strip(),
                rate.strip(),
                kinetics,
                issue,
                source_tag=tag,
                fixed_species=sorted(fixed),
            )

    if section:
        raise ValueError(f"unterminated MOZART {section} block")


def _statements(text, start, stop=None):
    """Yield labeled semicolon-terminated statements, and continuations."""
    active = False
    found = False
    pending = ""
    first = 0
    for line, raw in enumerate(text.splitlines(), 1):
        row = raw.split("//")[0].split("!")[0].strip()
        if row == start:
            active = True
            found = True
            continue
        if not active or not row:
            continue
        if row == stop:
            break
        if start == "#EQUATIONS" and row.startswith("#"):
            if pending:
                raise ValueError(f"unfinished reaction near line {first}")
            active = False
            continue
        if not pending:
            first = line
        pending = (pending + " " + row).strip()
        while ";" in pending:
            statement, pending = pending.split(";", 1)
            yield first, statement.strip()
            pending = pending.strip()
            first = line
    if not found or pending:
        raise ValueError(
            f"missing {start!r} or unfinished reaction near line {first}"
        )


def _expression(rate, source):
    compact = re.sub(r"\s+|_dp", "", rate, flags=re.I)
    if re.fullmatch(NUMBER, compact):
        return (number(compact), 0, 0, 300)
    if source == "mcm":
        product = re.fullmatch(rf"({NUMBER})\*({NUMBER})", compact)
        if product:
            return (number(product[1]) * number(product[2]), 0, 0, 300)
        m = re.fullmatch(
            rf"({NUMBER})\*EXP\(({NUMBER})/TEMP\)(?:\*({NUMBER}))?",
            compact,
            re.I,
        )
        if m:
            a, c, branch = m.groups()
            return (number(a) * number(branch or "1"), number(c), 0, 300)
    if source == "wrf-chem":
        m = re.fullmatch(rf"ARR2\(({NUMBER}),({NUMBER}),TEMP\)", compact, re.I)
        if m:
            return (number(m[1]), -number(m[2]), 0, 300)
    if source == "cmaq":
        m = re.fullmatch(rf"({NUMBER})@({NUMBER})", compact)
        if m:
            return (number(m[1]), -number(m[2]), 0, 300)
    return None


def _mcm_rates(companions):
    # These two York distributions share the same unconditional rate routine.
    digest = "d98b98ebfc168de798ecdcd2da947438a440d9e0edf95d568436387fc68aa31f"
    for data in companions:
        if hashlib.sha256(data).hexdigest() != digest:
            continue
        return dict(
            re.findall(r"^    (K\w+) = ([^\r\n]+)", data.decode(), re.M)
        )
    return {}


def _resolve(rate, source, definitions):
    match = re.fullmatch(rf"(K\w+)(?:\s*\*\s*({NUMBER}))?", rate)
    if match and match[1] in definitions:
        expression = definitions[match[1]]
        kinetics = _expression(expression, "mcm")
        if kinetics:
            a, c, b, tref = kinetics
            kinetics = (a * number(match[2] or "1"), c, b, tref)
        return kinetics, {match[1]: expression}
    return _expression(rate, source), {}


def ro2_members(text):
    """Read the explicit KPP radical pool and its continuation lines."""
    preamble = text.partition("#EQUATIONS")[0]
    logical_lines = re.sub(r"&\s*\n\s*&?", "", preamble)
    pool = re.search(r"^\s*RO2\s*=([^\n]*)", logical_lines, re.M | re.I)
    return sorted(set(re.findall(r"C\(ind_(\w+)\)", pool[1]))) if pool else []


def _mcm_elements(text):
    catalog = {}
    for section in re.finditer(
        r"^#DEFVAR\s*$(.*?)(?=^#|\Z)", text, re.M | re.S
    ):
        for name, formula in re.findall(
            r"^(\w+)\s*=\s*(.*?)\s*;", section[1], re.M
        ):
            counts: Counter[str] = Counter()
            for term in formula.split("+"):
                atom = re.fullmatch(r"(\d*)([A-Z][a-z]?)", term.strip())
                if atom is None:
                    break
                counts[atom[2]] += int(atom[1] or "1")
            else:
                catalog[name] = dict(counts)
    return catalog


def _kpp_fixed(text):
    # Drop comments before recognizing directives and species assignments.
    text = re.sub(r"\{[^}]*\}|//[^\n]*|![^\n]*", "", text)
    for block in re.finditer(
        r"^[ \t]*#DEFFIX\b(.*?)(?=^[ \t]*#|\Z)", text, re.I | re.M | re.S
    ):
        yield from re.findall(r"\b([A-Za-z]\w*)\s*=", block[1])


def _fixed_species(source, text, companions):
    fixed = {"M", "N2", "O2"}
    if source == "wrf-chem":
        fixed.add("H2O")
    if source in {"mcm", "wrf-chem"}:
        fixed.update(_kpp_fixed(text))
        for data in companions:
            if b"#DEFFIX" in data.upper():
                fixed.update(_kpp_fixed(data.decode("utf-8-sig")))
    if source == "cmaq":
        fixed.update(("H2", "CH4", "H2O"))
    return fixed


def labeled(text, source, companions=()):
    """Read York/WRF KPP equations or CMAQ MECHDEF in molecular units."""
    is_cmaq = source == "cmaq"
    start = "REACTIONS[CM] =" if is_cmaq else "#EQUATIONS"
    labels = r"<([^>]+)>" if source != "wrf-chem" else r"\{([^}]+)\}"
    sep = "#" if is_cmaq else ":"
    elements = _mcm_elements(text) if source == "mcm" else {}
    definitions = _mcm_rates(companions) if source == "mcm" else {}
    fixed = _fixed_species(source, text, companions)
    for line, stmt in _statements(text, start, "END MECH" if is_cmaq else None):
        m = re.fullmatch(labels + r"\s*(.+)", stmt)
        if not m or sep not in m[2]:
            raise ValueError(f"line {line}: invalid {source} reaction")
        equation, rate = m[2].split(sep, 1)
        rate = rate.strip()
        kinetics, references = _resolve(rate, source, definitions)
        issue = "" if kinetics else "requires source rate expression or forcing"
        if "hv" in equation.lower() or "J(" in rate.upper() or "/<" in rate:
            issue = "requires a source photolysis frequency"
        clean_eq = equation.split("%")[0]
        entry = channel(m[1], line, clean_eq.strip(), rate, kinetics, issue)
        entry.equation = equation.strip()
        if references:
            entry.metadata["rate_definitions"] = references
        if "%" in equation:
            entry.reason = "requires CMAQ special rate operator"
        entry.metadata["fixed_species"] = sorted(fixed)
        if source == "mcm":
            entry.metadata["elements"] = {
                name: elements[name]
                for name in set(entry.reactants) | set(entry.products)
                if name in elements
            }
        yield entry
