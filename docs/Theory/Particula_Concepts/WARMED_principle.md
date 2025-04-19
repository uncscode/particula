# The WARMED Developer Principle

*A concise philosophy for building and maintaining Particula.*

| Letter | Focus | One–line guideline |
|--------|-------|--------------------|
| **W** | **Writing** | Write code that is direct, minimal, and fits the problem. |
| **A** | **Agreeing** | Discuss and settle on _how_ a feature is implemented **before** merging. |
| **R** | **Reading** | Code must explain itself; comments fill the gaps, not the voids. |
| **M** | **Modifying** | Any competent dev should be able to extend or swap a component in minutes. |
| **E** | **Executing** | Keep NumPy/SciPy as the only hard deps; favor vectorization and avoid hidden `for`‑loops. |
| **D** | **Debugging** | Fail fast with helpful messages and provide deterministic tests. |

## Why WARMED instead of CLEAN?

CLEAN Code \citep{UncleBobCLEAN2009} is an excellent *general* guide, but aerosol scientists are usually **both** the developer **and** the user.  
WARMED shifts the emphasis from enterprise‑scale maintainability toward day‑to‑day research agility:

* Really short iterations: prototype ➔ publish ➔ archive.
* Minimal ceremony: no “service layers” or factory jungles.
* Maximum clarity for humans **and** language models—LLMs can audit, explain, and even refactor WARMED‑style code.

## Practical safeguards already in place

* Builder classes check required parameters and types before any heavy work starts.
* `@validate_inputs` decorators enforce domain‑specific invariants (positive radius, non‑negative concentration, finite Coulomb potential, …).
* Exhaustive docstrings + Sphinx docs make auto‑completion meaningful in modern IDEs.

## Where WARMED shows up in the repo

```text
particula/
 ├─ dynamics/…/turbulent_shear_kernel.py   ← readable, single‑purpose functions
 ├─ dynamics/…/coagulation_builder/        ← builders enforce “A” & “M”
 ├─ util/validate_inputs.py                ← fast‑fail debugging
 └─ docs/                                  ← you are here (R)
```

> *Bottom line:* WARMED code is easy to write, easy to read, trivial to tweak, quick to run, and painless to debug—exactly what a researcher needs.

---
