# Docstring Specification: Python Classes, Attributes, and Methods

**Instruction**  
Analyze all Class definitions provided and improve their docstrings for clarity, consistency, and adherence to best practices.

---

## High-Level Objectives

1. **Improve Documentation Clarity**  
   - Ensure all function docstrings follow a clear, structured format.  
   - Use concise parameter descriptions and clearly stated return values.

2. **Include Mathematical Equations**  
   - When applicable, present equations in **Unicode format** for broader compatibility.

3. **Ensure Consistency**  
   - Maintain uniform style for docstrings: proper indentation, spacing, formatting.

4. **Add Examples**  
   - Provide code snippets to demonstrate usage of classes and methods.

5. **Include References**  
   - Insert a **"References"** section where needed, citing reliable sources (e.g., Wikipedia, journal articles, or books).

---

## Mid-Level Objectives

1. **Standardized Docstring Format**  
   Use the following template as a guide:

   ```python
   class ExampleClassName:
       """
       Short description of what the class is or does.

       Longer description of the class, including its purpose and functionality.
       This can be multiple lines. Discuss why the class is important and how it
       fits into a larger API or system.

       Attributes:
           - param1 : Description of param1.
           - param2 : Description of param2.

       Methods:
           - method_name: Brief description or context.
           - another_method: Brief description or context.

       Examples:
           ```py title="Example Usage"
           import particula as par
           example_object = par.ExampleClassName(param1, param2)
           output = example_object.method_name(value1, value2)
           # Output: ...
           ```

       References:
           - Author Name, "Title of the Article," Journal Name,
             Volume, Issue, Year.
             [DOI](link)
           - "Article Title,"
             [Wikipedia](URL).
       """

       def __init__(self, param1, param2):
           """
           Initialize the ExampleClassName with parameters.

           Arguments:
               - param1 : Description of param1.
               - param2 : Description of param2.

           Returns:
               - None
           """
           self.param1 = param1
           self.param2 = param2

       def method_name(self, value1, value2):
           """
           Brief description of what the method does.

           A longer description of the method, including its purpose
           and methodology. Can be multiple lines. For example:

           - φ = (γ × β) / c
               - φ is Description of φ.
               - γ is Description of γ.
               - β is Description of β.
               - c is Description of the constant.

           Arguments:
               - value1 : Description of value1.
               - value2 : Description of value2.

           Returns:
               - Description of the return value.

           Examples:
               ```py title="Example"
               example_object.method_name(2, 3)
               # Output: 1.5
               ```

               ```py title="Example Usage with Arrays"
               example_object.method_name(np.array([4,5,5]), np.array([2,3,4]))
               # Output: array([4.0, 1.66666667, 1.25])
               ```

           References:
               - Author Name, "Title of the Article," Journal Name,
                 Volume, Issue, Year.
                 [DOI](link)
               - "Article Title,"
                 [Wikipedia](link).
           ```

2. **Mathematical Equation Representation**  
   - Include relevant mathematical equations in **Unicode format** (e.g., `C = (P × M) / (R × T)`).

3. **Consistent Spacing and Formatting**  
   - Insert a **space** after each colon in parameter descriptions (`- parameter : Description`).  
   - Maintain proper line breaks and indentation.  
   - Provide usage examples under an **"Examples"** subheading.  
   - Include a **"References"** section whenever citing sources.

---

## Implementation Steps

1. **Analyze Existing Docstrings**  
   - Identify missing or incorrect parameter names.  
   - Check for inconsistent formatting, unclear descriptions, or missing references.

2. **Update Docstrings for Clarity**  
   - Use a single, consistent style for parameter listings: `- parameter : Description`.  
   - Verify each parameter and return value is accurately described.  
   - Keep line lengths under 79 characters when possible.  
   - Use equations in Unicode format where relevant.  
   - Add references for scientific or technical validity.

3. **Apply Consistency Rules**  
   - All arguments must follow the `parameter : Description` style.  
   - Equations should be in Unicode format.  
   - Insert a "References" section if any sources are used.  
   - Ensure docstrings are consistently structured.

---

## Final Checklist

- [ ] Docstrings follow the uniform format (brief description, parameters, returns, etc.).  
- [ ] Equations are presented in Unicode format if applicable.  
- [ ] Parameter names and descriptions are accurate and consistent.  
- [ ] Spacing and indentation are correct.  
- [ ] References are included when needed.  
- [ ] Every class and method has a complete, clear docstring.

---
