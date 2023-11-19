# FY24 Plan

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

## FY24 Deliverables

### Particula v0.1.0 Release

Scheduled for late winter/early spring, the inaugural release of Particula will have two manuscript publications. This release will encompass:

**Comprehensive Documentation**: An in-depth guide detailing the codebase structure and operational instructions.

**Interactive Jupyter Notebook**: Featuring practical examples to facilitate user engagement with the software.
Key Features in Particula v0.1.0

Paper 1: A Particula model of the Microphysics and Chemistry of Aerosols
- Equilibrium Analysis: Contrasting Thermodynamic and Dynamic Equilibria.
- Chemical Processes:
  - Implementing Volatility Basis Set (VBS) for chemical reactions, in collaboration with CMU/Neil?
- Addressing Non-ideal mixing using BAT/AIOMFAC models, in partnership with McGill-Zuend?
- Atmospheric Phenomena Simulation:
  - Strategies for simulating smoke, smog, or cloud formation.
  - Techniques: Sectional Method and Super Droplet (Direct Simulation).
- Processes: Emphasizing Coagulation, Condensation, Evaporation, and Nucleation.
  - Impact Analysis: Investigating how initial emissions influence cloud formation and updraft velocity.
Paper 2: Data Integration and Experimental Validation
- Data pipeline in Particula: Aligning modeling with experimental data form a better understanding of the phenomena.
- Remote Sensing Data Incorporation: Exploring integration with DOE-Radar Observations for cloud droplet analysis or other remote sensing data like AERONET.
- Case Study: Examining the Saharan Dust transport to the DOE Houston site in 2022?

### Collaborative Publications (Secondary role in helping setup the codebase for collaborators)

- LANL Collaboration: CLoud Chamber Characterization and Coagulation Experiments
  - Size dependent wall loss correction
  - Dry (no-humidity) experiments for coagulation at high concentrations (1000 ug/m3)
    - NaCl (coagulation/wall loss code validation)
    - Sucrose, PEG (coagulation/wall loss code validation)
    - Smoldering Smoke
    - Flaming Smoke (soot): seeing if there is a morphology effect on coagulation
    - Dust + Smoke: Ash fallout analog
    - Dust + Sucrose
- MTU Collaboration: Dust Coagulation and Transport
  - Using a Lagrangian model to simulate charge separation (static) and settling of dust.
  - Showing settling lifetimes and how they are effected by charge separation.
    - Role of humidity?
  - Theoretical bases for future chamber experiments on dust.
- MTU Collaboration: Soot in Clouds
  - Using a Lagrangian model to simulate cloud formation, smoke and soot particles.
  - Specifically, exploring the parameter space of initial conditions (emissions) and how much smoke is nucleated vs coagulated to cloud droplets.
    - Under different turbulence conditions/fields (taken as an a priori input, not simulated).
    - How many droplet activation cycles do particles go through when in a cloud?
    - What comes down in the rain? and what stays after the cloud evaporates?
- CMU Collaboration (CLOUD/Neil)
  - They probably have a lot of data, so we could get them started on the code base to look at something interesting.

### Major Milestones for FY24

- [ ] Refactor Sectional Method
  - [ ] Wall loss correction, experimental data integration
- [ ] Add Non-ideal Mixing (BAT/AIOMFAC integration, maybe web api)
  - [ ] Add Thermodynamic Equilibrium
- [ ] Add Super Droplet Method
  - [ ] Rough estimates of memory on an RTX A6000 16 GB using pytorch can handle about 5000 particles tracking (position and velocity only) with coagulation checks, time steps of 0.000008 sec, 1000 sec simulation time
- [ ] Add Chemistry (this could come later, but simple VBS bin shift might be achievable)
  - [ ] Add Volatility Basis Set (VBS) for chemical reactions
- [ ] Publish Particula v0.1.0
  - [ ] Paper 1: A Particula model of the Microphysics and Chemistry of Aerosols
  - [ ] Paper 2: Data Integration and Experimental Validation
