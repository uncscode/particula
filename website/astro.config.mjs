import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import { unified } from '@astrojs/markdown-remark';
import starlightPydocs, { pydocsSidebarGroup } from 'starlight-pydocs';
import { pluginCollapsibleSections } from '@expressive-code/plugin-collapsible-sections';
import remarkMath from 'remark-math';
import rehypeMathjax from 'rehype-mathjax';
import remarkMermaid from './scripts/remark-mermaid.mjs';
import watchSources from './scripts/watch-sources.mjs';

export default defineConfig({
  site: 'https://uncscode.github.io',
  base: '/particula',
  trailingSlash: 'always',
  vite: { plugins: [watchSources()] },
  markdown: {
    processor: unified({
      remarkPlugins: [remarkMath, remarkMermaid],
      rehypePlugins: [rehypeMathjax],
    }),
  },
  integrations: [starlight({
    title: 'Particula',
    description: 'Aerosol science, from first principles to simulation.',
    logo: { light: './public/images/icon_light_clear.svg', dark: './public/images/icon_dark_clear.svg', replacesTitle: false },
    favicon: '/images/icon_dark.svg',
    social: [{ icon: 'github', label: 'GitHub', href: 'https://github.com/uncscode/particula' }],
    customCss: ['./src/styles/custom.css'],
    components: { Head: './src/components/Head.astro' },
    expressiveCode: { plugins: [pluginCollapsibleSections()], shiki: { langs: ['python', 'bash', 'json', 'text'] } },
    plugins: [starlightPydocs({
      packages: [{
        name: 'particula', base: 'API/particula', source: { file: '.cache/griffe.json' },
        docstringStyle: 'google',
        filters: { private: true },
        sourceLink: { host: 'github', repo: 'uncscode/particula', ref: 'main', root: '..' },
      }],
    })],
    sidebar: [
      { label: 'Guide', items: [
        { label: 'Start here', slug: 'guide' },
        { label: 'First CPU simulation', slug: 'guide/getting-started/first-simulation' },
        { label: 'Construct an aerosol', slug: 'guide/data-model/construction' },
        { label: 'CPU and GPU support', slug: 'guide/gpu/support' },
      ] },
      { label: 'Examples', items: [
        { label: 'Explore examples', slug: 'examples' },
        { label: 'Size distributions', slug: 'examples/fundamentals/particles/aerosol_distributions' },
        { label: 'Condensation', slug: 'examples/processes/condensation/condensation_1_bin' },
        { label: 'GPU coagulation', slug: 'examples/gpu/direct/gpu_coagulation_direct' },
        { label: 'Saved-result states', slug: 'examples/result-states' },
      ] },
      { label: 'Theory', items: [
        { label: 'Explore the science', slug: 'theory' },
        { label: 'Thermodynamic activity', slug: 'theory/thermodynamics/activity-theory' },
      ] },
      { label: 'API', collapsed: true, items: [{ label: 'API overview', slug: 'api' }, pydocsSidebarGroup] },
      { label: 'Contribute', items: [{ label: 'Contribute', slug: 'contribute' }, { label: 'Author a Python example', slug: 'contribute/script-examples' }] },
    ],
  })],
});
