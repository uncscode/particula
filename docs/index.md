import StudyCarousel from '@components/StudyCarousel.astro';
import { CardGrid, LinkCard } from '@astrojs/starlight/components';


<a id="particula"></a>

<StudyCarousel
  label="Featured studies"
  featured={[
    "cloud-ion-humidity",
    "examples-simulations-cloud-chamber-multi-cycle",
    "pendant-drop-evaporation",
    "nacl-scavenging",
    "pi-chamber-spectra",
    "shear-drop-spectrum",
    "sea2cloud-events",
    "pinic-isomers",
    "mam4xx-coagulation",
    "mt-hua-salts",
  ]}
/>

## Dig deeper into the science

<CardGrid>
  <LinkCard title="Run your first simulation" description="A native aerosol and a small CPU condensation step." href="/particula/guide/getting-started/first-simulation/" />
  <LinkCard title="Agent plugins" description="Set up Particula Helper in Codex, Claude, or OpenCode to design simulations, analyze results, and develop models." href="/particula/guide/agent-plugins/" />
  <LinkCard title="Guide" description="Construction, ownership, and the CPU/GPU support boundary." href="/particula/guide/" />
  <LinkCard title="Studies" description="Find runnable Python scripts, literature comparisons, validation evidence and benchmarks." href="/particula/studies/" />
  <LinkCard title="Theory" description="Equations and assumptions behind the processes." href="/particula/theory/" />
  <LinkCard title="API reference" description="Signatures and docstrings extracted directly with Griffe." href="/particula/api/" />
  <LinkCard title="Contribute" description="Develop models and maintain reproducible examples." href="/particula/contribute/" />
</CardGrid>
