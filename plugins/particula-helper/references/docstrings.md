# Scientific docstrings

Use Google-style docstrings with an accurate short summary and only relevant
sections. Read the function/class body and callers before describing behavior.
Do not invent validation, units, equations, return shapes or exceptions.

For modules, explain the physical purpose and cite primary model/equation
references. For functions, cover physical meaning, units, valid domains,
array axes/broadcasting, return units/shapes, mutation and raised errors.
For classes, describe responsibility, construction constraints and public
attributes; explain identity/ownership when consequential. Avoid repeating
inherited or obvious type information.

Use `Args:`, `Returns:`, `Raises:`, `Attributes:`, and `Examples:` where useful.
Use actual public argument names. Do not use the retired templates' alternative
`Arguments:` headings, mandatory reference sections without a source, universal
backslash continuation rules, or long boilerplate lists. Keep lines within the
repository's 80-character convention and let Ruff enforce its configured rules.

Scientific documentation should distinguish radius from diameter, count from
number density/PDF, particle mass from total inventory, and represented volume
from computational capacity. State sign conventions and inert/species order
when applicable. Explain assumptions and applicability; do not present fitted
or illustrative values as measured constants.

Include equations only when they clarify the contract, defining each symbol
and linking the primary source. Examples must use supported native APIs and
remain short enough to check. Larger scientific workloads belong in registered
Python examples with saved outputs. Do not change behavior to match old prose;
correct prose or separately justify the intended behavior change.
