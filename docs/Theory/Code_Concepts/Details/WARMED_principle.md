# WARMED Code

*A concise philosophy for building and maintaining Particula.*

| Letter | Focus | One–line guideline |
|--------|-------|--------------------|
| **W** | **Writing** | Write code that is direct, minimal, and fits the problem. |
| **A** | **Agreeing** | Discuss and settle on _how_ a feature is implemented **before** merging. |
| **R** | **Reading** | Code and variable names must explain themselves; comments fill the gaps, not the voids. |
| **M** | **Modifying** | Any competent dev should be able to extend or swap a component in minutes. |
| **E** | **Executing** | Favor vectorization and avoid hidden `for`‑loops. |
| **D** | **Debugging** | Fail fast with helpful messages and provide deterministic tests. |

See Casey Muratori’s [Where Does Bad Code Come From?](https://youtu.be/7YpFGkG-u1w?si=ihTtVUKebJ1zJ2yy) talk for a deep dive into the WARMED principles.

## Why WARMED instead of CLEAN?

CLEAN Code is not an effective guide. Aerosol scientists are usually **both** the developer **and** the user. WARMED shifts the emphasis from enterprise‑scale maintainability toward day‑to‑day research agility:

* Really short iterations: prototype ➔ publish ➔ archive.
* Minimal ceremony: no “service layers” or factory jungles.
* Maximum clarity for humans **and** language models—LLMs can audit, explain, and even refactor WARMED‑style code.

See Casey Muratori’s ["Clean" Code, Horrible Performance](https://youtu.be/tD5NrevFtbU?si=ZJBEnhqVGYqAerXM) talk for a deep dive into why CLEAN runs into problems.

## Practical safeguards already in place

* Builder classes check required parameters and types before any heavy work starts.
* `@validate_inputs` decorators enforce domain‑specific invariants (positive radius, non‑negative concentration, finite Coulomb potential, …).
* Exhaustive docstrings + LLMs docs make auto‑completion meaningful in modern IDEs.

## Where WARMED shows up in the repo

```text
particula/
 ├─ dynamics/…/turbulent_shear_kernel.py   ← readable, single‑purpose functions
 ├─ dynamics/…/coagulation_builder/        ← builders enforce “A” & “M”
 ├─ util/validate_inputs.py                ← fast‑fail debugging
 └─ docs/                                  ← you are here (R)
```

---
