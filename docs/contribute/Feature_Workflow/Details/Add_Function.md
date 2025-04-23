# Add Functions

## Where

Create or update a module under particula/<area>/.

## Checklist

* Python – no I/O or global state changes.
* Follow [Code Specifications](../Code_Specifications/index.md) for formatting.
* Input validation via util.validate_inputs when applicable.
* Logging: use `logger = logging.getLogger("particula")` and log at DEBUG.

## Steps

1. Create a new issue on GitHub and assign it to yourself.
   1. Create a branch on your forked repo for this issue.
2. Add a new function(s) to the appropriate module in particula/<area>/.
   1. If the module is new, add it to `__init__.py`.
   2. If the function is a helper, add prefix `_` to the function name.
   3. Write a docstring.
4. Add type hints for all parameters and return values.
5. Add Unit Tests for the function (see [Add Unit Tests](Add_Unit_Test.md)).
6. Commit this file in a branch.
7. Create your pull‑request to the main repo.
