# Feature Workflow

The overall workflow depends on the extent of the feature being added.
This outline covers how to add a completely new feature to the library.  
However, depending on your situation some of the steps may be skipped, if in doubt, please start a discussion in the
[GitHub Discussions](https://github.com/uncscode/particula/discussions).

## Full Workflow

The overall workflow for adding a new feature to the library is as follows:

**Feature Proposal**: Create a new discussion in the [GitHub Discussions](https://github.com/uncscode/particula/discussions) to propose your vision and goals.

- Include a description of the feature, its purpose, and how it fits into the library.
- Add manuscripts or references to contextualize the feature and equations to implement.
- Discuss the feature with the community to gather feedback and suggestions.
- Get the okay from the maintainer(s) to proceed with the implementation.

---

1. [Add Theory](Add_Theory.md): Write the theory behind the feature.
2. [Add Functions](Add_Function.md): Write the functions that implement the feature.
3. [Add Unit Tests](Add_Unit_Test.md): Write unit tests to ensure the function works as intended.
4. [Add Class](Add_Class.md): Write the class that implements the functions.
5. [Add Unit Tests](Add_Unit_Test.md): Write unit tests for the class to ensure it works as intended.
6. [Add to init](Add_to_init.md): Add the new functions/class to the `__init__.py` file to make it accessible from the package.
7. [Add Examples](Add_Example.md): Write examples to demonstrate the feature.

> _Note:_ Each step would be one or more issues, e.g., one for the theory, 1+ for the functions, etc.
