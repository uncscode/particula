# One-line Vision

**Objective**
Give users a *single* statement—`par.use_backend("taichi")`, `par.use_backend("warp")`, or nothing at all—to decide whether each heavy-duty kernel runs as plain-Python/NumPy or on an accelerated backend.  All builders, simulators, and analysis notebooks continue to call the same public functions (`par.coagulation_gain_rate`, `par.foo`, …).  In other words: “flip a switch, get speed, keep code.”


#### Design constraints we start with

| Constraint              | Why it matters                                                           |
| ----------------------- | ------------------------------------------------------------------------ |
| **One-liner for users** | Notebooks shouldn’t need `if backend:` blocks or different import paths. |
| **Clean public API**    | Research scripts and tests written today must still run tomorrow.        |
| **Graceful fallback**   | If Taichi/c++ isn’t installed, everything still works (slower).         |
| **Low maintenance**     | Contributors shouldn’t touch every file each time we add a new backend.  |
| **Lazy dependencies**   | Heavy libraries load only when that backend is active.                   |

Early idea: insert an `if par.backend.is_enabled(): …` test inside every compute routine.
*Problem:* scales poorly (hundreds of duplicated `if` blocks), scatters backend logic across the codebase, and complicates testing.


#### Why we chose a **dispatch-decorator + registry**

See the [Dispatch Decorator](#dispatch-decorator) section below for a full code example.

| Alternative                                         | Drawbacks we avoided                                                |
| --------------------------------------------------- | ------------------------------------------------------------------- |
| **Inline `if` statements**                          | Boilerplate in every function; easy to miss one path; hard to grep. |
| **Monkey-patching modules**                         | Breaks static analysis, doc generation, and can confuse IDEs.       |
| **Separate “backend” namespace (`par.taichi.foo`)** | Forces users to learn new call sites and duplicate builder logic.   |

**Dispatch layer** centralizes the decision in *one* lightweight wrapper:

```python
@par.dispatchable        # decorates the Python reference impl
def coagulation_gain_rate(...):
    ...

@par.register("coagulation_gain_rate", backend="taichi")
def _fast_taichi_impl(...):
    ...
```

* The wrapper looks up `par.get_backend()`.
* If an accelerated version is registered for that name, it calls it.
* Otherwise it falls back to the original Python body.

Outcome: **zero duplication, single source-of-truth docstring, plug-and-play backends.**

#### How integration feels to each stakeholder

*Users*

```python
import particula as par
par.use_backend("taichi")      # one line, opt-in
result = par.coagulation_gain_rate(r, c, K)
```

If Taichi isn’t available—or if that particular kernel hasn’t been ported yet—the same call silently executes the pure-Python path.

*Backend Developer*

```python
from particula import register

@register("coagulation_gain_rate", backend="warp")
def gain_rate_warp(...):
    ...
```

No changes to public modules, no touching docstrings, no conflicts with other backends.

*Core maintainers*

* Only the tiny `_dispatch.py` needs to know about backend state.
* Tests iterate over the registry to verify numerical consistency.
* Doc generation shows *one* canonical signature per function.


#### Key benefits

* **Scalability:** add Taichi today, C++ tomorrow, SIMD next year—public API untouched.
* **Safety:** automatic Python fallback guarantees correctness over performance.
* **Maintainability:** backend logic lives in dedicated files; diff-friendly and testable.
* **Developer ergonomics:** decorating existing functions is a two-second task.
* **Performance isolation:** heavy imports load only when a user explicitly selects them.

---

## Dispatch Decorator

Our goal is to have **one line for users, zero friction for contributors.**
What changes is *how* we wire the backends. Instead of sprinkling `if par.backend.is_enabled(): …` inside every kernel, we wrap each public-API function in a **single decorator that does automatic dispatch**:

```python
import particula as par

par.use_backend("taichi")        # or "warp", "cpp", …
# builder code stays identical ↓
gain = par.coagulation_gain_rate(radius, conc, kernel)
```

Behind the scenes:

1. **The original Python implementation is the source-of-truth.**
   It is defined once, carries the docstring, and is decorated with `@par.dispatchable`.

2. **The decorator installs a lightweight wrapper** that:

   1. Looks up the currently active backend (`par.get_backend()`).
   2. Checks a registry (`_registry[func_name].get(backend)`).
   3. **Calls the registered accelerated version if it exists; otherwise falls back** to the original Python body.

3. **Accelerated versions live in separate modules** and register themselves with:

   ```python
   @par.register("coagulation_gain_rate", backend="taichi")
   def coagulation_gain_rate_taichi(radius, conc, kernel):
       # Taichi kernel here …
   ```

4. **No accelerated version? No problem.**
   The wrapper silently calls the pure-Python code, so functionality never breaks.

### Why this approach meets our requirements

| Requirement                                     | How dispatch-decorator delivers                                                                                                                                                   |
| ----------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| *“Single enable statement for users.”*          | `par.use_backend("taichi")` sets one global flag (with an optional context manager for thread-safety).                                                                            |
| *“Builder APIs and call sites stay unchanged.”* | All backend logic lives inside the decorator wrapper; public signatures are untouched.                                                                                            |
| *“Minimal boilerplate for new kernels.”*        | Add `@par.dispatchable` to the existing Python function → done.  If/when you write a faster Taichi/Warp/C++ version, just drop it into `_backend_taichi.py` with `@par.register`. |
| *“Automatic Python fallback.”*                  | The wrapper uses `registry.get(backend, original_python_func)`, so the pure-Python path is always available.                                                                      |

### Example skeleton

```python
# -------- particula/_dispatch.py -----------------
_backend = "python"
_registry: dict[str, dict[str, callable]] = defaultdict(dict)

def use_backend(name: str):  # user API
    global _backend
    _backend = name.lower()

def get_backend() -> str:
    return _backend

def dispatchable(func):
    """Decorator that enables backend dispatch with Python fallback."""
    func_name = func.__name__

    def wrapper(*args, **kwargs):
        impl = _registry.get(func_name, {}).get(_backend, func)
        return impl(*args, **kwargs)

    # register default python implementation
    _registry.setdefault(func_name, {})["python"] = func
    wrapper.__doc__ = func.__doc__
    return wrapper

def register(func_name: str, *, backend: str):
    """Decorator factory for accelerated implementations."""
    def decorator(accel_func):
        _registry.setdefault(func_name, {})[backend.lower()] = accel_func
        return accel_func
    return decorator
```

```python
# -------- particula/public_api.py ----------------
from particula._dispatch import dispatchable

@dispatchable
def coagulation_gain_rate(radius, concentration, kernel):
    """Compute gain rate by trapezoidal integration (pure Python)."""
    return 0.5 * np.trapz(
        kernel * concentration[:, None] * concentration,
        radius,
        axis=1,
    )
```

```python
# -------- particula/_backend_taichi.py -----------
from particula._dispatch import register
import taichi as ti

@register("coagulation_gain_rate", backend="taichi")
def coagulation_gain_rate_taichi(radius, concentration, kernel):
    # Taichi implementation here …
    return ti_kernel(radius, concentration, kernel)
```

### What users see

```python
import particula as par
par.use_backend("taichi")          # one-liner switch

gain_rate = par.coagulation_gain_rate(r, c, K)  # transparently fast
```

If the Taichi module is unavailable or that particular kernel hasn’t been ported yet, **the call runs the original Python version without any code change or error**.

---

## Bottom line

By decorating existing Python functions with `@dispatchable`, we *add* optional acceleration instead of *rewriting* for it.  The registry-based dispatch keeps maintenance low, guarantees a safe fallback path, and preserves the clean public API that users already know.
