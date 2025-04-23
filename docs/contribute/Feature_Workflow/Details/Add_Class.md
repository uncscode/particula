# Add Class

Create a new class to implement the feature.  
This is typically a wrapper around the functions you wrote in the previous step.

## Choose a base

- `BuilderABC` – when key/parameter checking is useful.
- `dataclass` – for simple, immutable data containers.
- Regular class – for everything else.

## Required Sections

- `__init__` with complete type hints.
- Public interface first (getters, setters, actions).
- Docstring that includes an “Examples” subsection.
- Follow [Code Specifications](../Code_Specifications/index.md) for formatting.

## Steps

1. Create a new issue on GitHub and assign it to yourself.
2. Create a branch on your forked repo for this issue.
3. Add a new class(s) to the appropriate module in particula/<area>/.
   - If the module is new, add it to `__init__.py`.
   - Use `ABC` for abstract classes and `BuilderABC` for builders.
   - Call your functions in the class methods and keep most calculations in the functions (not directly in the class).
4. Write a docstring.
5. Add type hints for all parameters and return values.
6. Add unit tests for the class (see [Add Unit Tests](Add_Unit_Test.md)).
7. Commit this file in a branch.
8. Create your pull‑request to the main repo.
