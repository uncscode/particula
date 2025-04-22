# Docstring Specification: Python Functions

> **Analyze all function definitions provided, improve the docstrings, and ensure clarity, consistency, and adherence to best practices.**  

---  

## High-Level Objective  

- **Improve Documentation Clarity:**  
  Ensure that all function docstrings follow a structured and readable format, using clear parameter descriptions and return values.  

- **Include Mathematical Equations:**  
  When applicable, include equations in Unicode format to enhance readability and scientific accuracy.  

- **Ensure Consistency:**  
  Maintain a uniform style for docstrings, ensuring proper indentation, spacing, and structure.

- **ADD EXAMPLES:**  
  Add examples to demonstrate function usage, if applicable.

- **Include References:**  
  Add a **"References"** section when applicable, citing sources such as Wikipedia pages, journal articles, or books for scientific validity.  

---  

## Mid-Level Objectives  

- **Standardized Docstring Format:**  
  Use the following format:  

  ```python
  def function_name(param1: type, param2: type) -> return_type:
      """
      Brief description of what the function does.

      A description of the function, including the purpose
      and methodology. Can be multiple lines. Where calculated as:

      - φ = (γ × β) / c
          - φ is Description of φ.
          - γ is Description of γ.
          - β is Description of β.
          - c is Description of the constant.

      Arguments:
          - param1 : Description of param1.
          - param2 : Description of param2.

      Returns:
          - Description of the return value.

      Examples:
          ``` py title="Example title"
          import package_name as np
          np.function_name(2, 3)
          # Output: 1.5
          ```

          ``` py title="Example Usage 2 array input"
          import package_name as np
          np.function_name(np.array([4,5,5]), np.array([2,3,4]))
          # Output: array([4.0, 1.66666667, 1.25])
          ```

      References:
          - Author Name, "Title of the Article," Journal Name, Volume, Issue, Year.
              [DOI](url_link)
          - "Article Title",
            [Wikipedia](link)
      """
      return (param1 * param2) / CONSTANT
  ```
  
- **Mathematical Equation Representation:**  
  - If needed, include mathematical equations in **Unicode format** (e.g., `C = (P × M) / (R × T)`) for broader compatibility.  

- **Consistent Spacing and Formatting:**  
  - Ensure a **space after `:`** in Arguments: descriptions.  
  - Maintain proper indentation and line breaks, (`- parameter : Description`)
  - Use examples to demonstrate function usage.
  - Use a **"References"** section to cite sources.  

---  

## Implementation Steps  

### 1. Analyze the Function Docstrings  

Review each function's existing docstring to identify issues, including:  
- Incorrect parameter names  
- Inconsistent formatting  
- Missing or unclear descriptions  
- Lack of references when needed  

---  

### 2. Update Docstrings for Clarity and Readability  

- UPDATE function descriptions for clarity.  
- All parameter descriptions follow the same style (`- parameter : Description`). Using a hyphen, colon, and space before the description.
- UPDATE all parameters and return values are described.
- CHECK all line lengths are 79 characters or less, for readability.
- USE **equations in Unicode format** when applicable.  
  - DEFINE each variable and the equation in a clear and concise manner, in style '- variable is describe variable'.
- ADD **References** for scientific accuracy.  

**Example Before:**  
```python
def calculate_concentration(partial_pressure, molar_mass, temperature):
    """Calculate the concentration of a gas from its partial pressure, molar mass, and temperature using the ideal gas law.

    Parameters:
    pressure (float or NDArray[np.float64]): Partial pressure of the gas
    in Pascals (Pa).
    molar_mass (float or NDArray[np.float64]): Molar mass of the gas in kg/mol
    temperature (float or NDArray[np.float64]): Temperature in Kelvin.

    Returns:
    - concentration (float or NDArray[np.float64]): Concentration of the gas
    in kg/m^3.
    """
    return (partial_pressure * molar_mass) / (float(GAS_CONSTANT) * temperature)
```

**Example After (With Reference):**  
```python
def get_calculate_concentration(
    partial_pressure: Union[float, NDArray[np.float64]],
    molar_mass: Union[float, NDArray[np.float64]],
    temperature: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate the concentration of a gas using the ideal gas law.

    The concentration is determined using the equation:

    - C = (P × M) / (R × T)
        - C is the concentration in kg/m³,
        - P is the partial pressure in Pascals (Pa),
        - M is the molar mass in kg/mol,
        - R is the universal gas constant (J/(mol·K)),
        - T is the temperature in Kelvin.

    Arguments:
        - partial_pressure : Partial pressure of the gas in Pascals (Pa).
        - molar_mass : Molar mass of the gas in kg/mol.
        - temperature : Temperature in Kelvin.

    Returns:
        - Concentration of the gas in kg/m³.

    Examples:
        ``` py title="Example Usage"
        import particula as par
        par.gas.get_calculate_concentration(1.5, 0.02897, 298)
        # Output: 1.6175
        ```

    References:
        - "Ideal Gas Law,"
          [Wikipedia](https://en.wikipedia.org/wiki/Ideal_gas_law)
        - J. D. Lee, *Physical Chemistry*, 5th ed., Oxford University
          Press, 2019.
    """
    return (partial_pressure * molar_mass) / (float(GAS_CONSTANT) * temperature)
```
---

### 3. Apply Consistency Rules

Ensure that:  
- **All Argument descriptions** follow the same style (`parameter : Description`).  
- **Equations are formatted in Unicode** for clarity.  
- **A "References" section is included** when citing sources.  
- **All docstrings use a structured format** for easy readability.  

---

## Final Checklist

- [ ] **Docstrings follow a consistent format** (brief description, equation, arguments, return values, references).  
- [ ] **Equations are formatted properly** (Unicode, based on the context).  
- [ ] **Parameter names and descriptions are accurate**.  
- [ ] **Proper spacing and indentation are used**.  
- [ ] **References are included when applicable**.  
- [ ] **All functions have complete and clear docstrings**.  

---