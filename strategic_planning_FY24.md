# Strategic Planning FY24 for Particula Integration

## Review of Particula's Utilization

Particula has been instrumental in conducting coagulation simulations and educating on atmospheric science concepts over the last year. Feedback indicates a need for more intuitive class structures and simplified code extraction for standalone use.

In parallel, Datacula's project has encountered issues with the rigidity of data structures, particularly when adjusting data analysis processes. The need to re-import and reprocess large datasets when changes are made is a significant bottleneck.

## Insights and Reflections

- **Particula's Rigidity**: The tight coupling within the particula's class structure, connecting environment, vapors, and particles, limits extensibility and complicates comprehension.
- **Datacula's Inflexibility**: Similar to Particula, Datacula suffers from tight coupling of data to its analysis processes.
- **Complexity in Modification**: Adjusting the coagulation calculations is difficult due to its dependency on other class structures.

## Objectives for FY24

### Refactoring Particula

The goal is to restructure Particula to facilitate ease of modification and to integrate Datacula for shared functionalities. The proposed structure includes:

#### Discrete Concepts (Functions)

- **Role**: To provide standalone functionality like volume alteration or distribution integration usable in external projects.
- **Criteria**:
  - **Reuse**: Extracted when a block is utilized thrice or more.
  - **Independence**: Users can integrate concepts without additional code dependencies.
  - **Documentation**: Full documentation with references and commentary in line if the steps are complex.
  - **Standardization**: All inputs are in SI units, with internal conversions as needed.
  - **Testing**: Unit tests to ensure functionality.

#### Interface Concepts (Procedures)

- **Role**: To represent procedural workflows, for example, calculating coagulation rates or kappa.
- **Integration**: Utilizes multiple discrete functions or other interfaces minimally.
- **Accessibility**: Requires installation for full functionality (or coping 3-5 files), suitable for learning from.
- **Documentation**: Code doc strings. Accompanied by an illustrative Jupyter notebook, including usage cases, significance, and interrelation with the codebase. What it is, Why it is important, and how it relates to other parts of the code. At least one example of how to use it.
- **Testing**: Tests, but just to make sure it loads and all the sub-functions are called.

#### Systems-Level Concepts (Classes)

- **Role**: To abstract routine tasks, such as ODE coagulation simulation using particle classes.
- **Utility**: Serves users requiring higher abstraction without delving into interface mechanics. These poeple already know what they are calling and how the procedures work.
- **Documentation**: Detailed Jupyter notebooks explaining the class's purpose, its importance, and connection to other code components. What it is, Why it is important, and how it relates to other parts of the code. At least three examples of how to use it.
- **Testing**: Validated through Jupyter notebook execution.

By adhering to this plan, we aim to boost Particula's adaptability, making it more user-friendly and less time-consuming for large-scale data analysis. The refactoring will be instrumental in fulfilling the dual needs of process simplification and educational clarity.

## Conclusion

The proposed refactoring strategy for FY24 is crafted to address the current limitations of Particula and Datacula. It prioritizes modularity, intuitive understanding, and efficient data handling to aid researchers and educators alike.

# Long-Term Vision

## GPU Acceleration

Create a interface that uses, Pytourch, for GPU acceleration. This will allow for faster simulations of particle centric models.

