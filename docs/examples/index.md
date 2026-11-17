# Examples

Choose a calculation by the question you want to answer. Each worked example
connects physical assumptions, code, and saved results, then explains what the
result supports and how to investigate further. If the topic name is unfamiliar,
use the [topic explorer](../guide/topics/index.md) to move between a broad idea,
its prerequisites, and related questions.

## Start with the meaning of the state

Follow the [first CPU simulation](../guide/getting-started/first-simulation.md),
then work through [fundamentals](fundamentals/index.md). In particular,
[size distributions](fundamentals/particles/aerosol_distributions.py) explains why
count, concentration, and distribution density must be distinguished before
comparing any plots. [Gas and environment](fundamentals/gas/index.md) connects
stored vapor mass with saturation.

## Choose a scientific question

| Question | Start here | What to compare |
| --- | --- | --- |
| Why do particles gain or lose material? | [Condensation and other processes](processes/index.md) | Gas plus particle inventories, size-dependent growth, and timestep sensitivity. |
| Can number decrease while material is conserved? | [Coagulation representations](processes/coagulation/coagulation_4_compared.py) | Number, each material inventory, and the meaning of the distribution ordinate. |
| Where do particles enter or leave a chamber? | [Nucleation](processes/nucleation/index.md) and [wall loss](processes/wall-loss/index.md) | Source and sink budgets, size-dependent survival, and stochastic spread. |
| How does composition alter equilibrium? | [Thermodynamics](thermodynamics/index.md) | Activity, curvature, phase behavior, and the assumptions behind an equilibrium endpoint. |
| How does a size population interact with light? | [Sphere optics](optics/sphere_optics.py) | Single-particle efficiencies, population-weighted coefficients, and optical depth. |
| What changes when processes act together? | [Scientific simulations](simulations/index.md) | Controlled comparisons, external forcing, and limitations that a final plot can hide. |
| How can many independent histories be practical? | [GPU studies](gpu/index.md) | Per-box response, ensemble variability, numerical checks, and complete-workflow cost. |

## Build toward your research question

Start with one mechanism and a quantity you can check independently. Combine
processes only after you can explain their separate effects. A plausible graph
or conserved total does not establish numerical convergence or agreement with
measurements. Some retained simulations deliberately expose model or input
failures; their explanations identify what must change before research use.

For a new study, state the proposed comparison, hold appropriate inputs fixed,
and choose a diagnostic that could distinguish the competing explanations.
Use [Theory](../theory/index.md) for the physical basis and the
[Guide](../guide/index.md) for supported interfaces. The
[agent plugins](../guide/agent-plugins/index.md) can help carry this structure into
simulation design, execution, and analysis.

## Run and adapt an example

Published pages render saved outputs from maintained Python sources. Download
the Python file or the helper ZIP when a study uses shared code or data. Follow
the page's environment and execution requirements; CUDA publication runs require
CUDA, while the scaling report plots existing measurements on the CPU.

The [GPU index](gpu/index.md) distinguishes prepared scientific studies from
small direct-kernel and resident-interface demonstrations. Longer simulation
workloads retain their full inputs and numerical checks; begin with a small
process example before adapting them. See [installation](../guide/getting-started/installation/index.md)
for setup and [example authoring](../../website/AUTHORING.md) for the source and
saved-result workflow.
