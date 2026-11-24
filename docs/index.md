import StudyCarousel from '@components/StudyCarousel.astro';
import { CardGrid, LinkCard } from '@astrojs/starlight/components';

<a id="particula"></a>

<StudyCarousel
  label="Featured literature studies"
  literatureLink
  featured={[
    "cloud-1833",
    "mt-soledad",
    "manchester-campaign-performance",
    "antarctic-ccn",
    "dunne-cloud",
    "sea2cloud-events",
  ]}
/>

## Work with Particula

<CardGrid>
  <LinkCard title="Build a simulation" description="Choose a particle population and combine processes on the CPU or GPU." href="/particula/guide/" />
  <LinkCard title="Run a study" description="Start from a worked calculation, inspect its outputs, and adapt the Python script." href="/particula/studies/" />
  <LinkCard title="Explore the theory" description="Trace condensation, collisions, and transport back to their governing equations." href="/particula/theory/" />
  <LinkCard title="Look up an API" description="Find the inputs, units, and return values for a Particula function." href="/particula/api/" />
  <LinkCard title="Use Particula Helper" description="Set up an assistant to design simulations and analyze results." href="/particula/guide/agent-plugins/" />
  <LinkCard title="Contribute" description="Add a physical model, test a calculation, or share a worked example." href="/particula/contribute/" />
</CardGrid>
