# Explore aerosol topics

Follow a scientific question through a topic, its prerequisites, and related
calculations. The website provides illustrated topic cards with the next
2–4 choices.
Select a topic or subtopic to continue into its narrower branches; topics can
nest as deeply as the subject needs. These links also provide a starting point
when reading the documentation offline.

- **Describe an aerosol:** [population representations](../data-model/index.md), [size distributions](../../studies/tutorials/building-an-aerosol/particles/aerosol_distributions.py), [activity](../processes/activity.md), [optical properties](../optics/index.md).
- **Understand processes:** [condensation](../processes/condensation.md), [coagulation](../processes/coagulation.md), [sources and losses](../processes/index.md).
- **Numerical modeling:** [numerical methods](../processes/numerical-integration.md), [CPU simulation](../getting-started/first-simulation.md), [GPU execution](../gpu/index.md), [verification and comparison](../../contribute/validation/index.md).
- **Explore applications:** [scientific simulations](../../studies/examples/coupled-populations/index.md), [environmental sweeps](../../studies/examples/coupled-populations/environment/gpu_environment_sweep.py).

Population representations live under Describe an aerosol and remain related
to Numerical modeling: choices about counts, weights and resolution affect
numerical accuracy and interpretation.

Topic names, alternate terms, and page associations are maintained together in
`website/topics.json` in the source checkout. Guide, Examples, Theory, API, and
Contribute remain the documentation sections; topics connect material across them.
