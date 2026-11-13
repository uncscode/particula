import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import { unified } from '@astrojs/markdown-remark';
import starlightPydocs, { pydocsSidebarGroup } from 'starlight-pydocs';
import { pluginCollapsibleSections } from '@expressive-code/plugin-collapsible-sections';
import remarkMath from 'remark-math';
import remarkMkdocs from './scripts/remark-mkdocs.mjs';
import remarkDocLinks from './scripts/remark-doc-links.mjs';
import rehypeMathjax from 'rehype-mathjax';
import remarkMermaid from './scripts/remark-mermaid.mjs';
import watchSources from './scripts/watch-sources.mjs';
import navigation from './navigation.json';

export default defineConfig({
  site: 'https://uncscode.github.io',
  base: '/particula',
  trailingSlash: 'always',
  vite: { plugins: [watchSources()] },
  markdown: {
    processor: unified({
      remarkPlugins: [remarkMath, remarkMkdocs, remarkDocLinks, remarkMermaid],
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
    components: {
      Head: './src/components/Head.astro',
      Header: './src/components/Header.astro',
      Hero: './src/components/Hero.astro',
      Sidebar: './src/components/Sidebar.astro',
      PageFrame: './src/components/PageFrame.astro',
      TwoColumnContent: './src/components/TwoColumnContent.astro',
    },
    expressiveCode: { plugins: [pluginCollapsibleSections()], shiki: { langs: ['python', 'bash', 'json', 'text'] } },
    plugins: [starlightPydocs({
      packages: [{
        name: 'particula', base: 'API/particula', source: { file: '.cache/griffe.json' },
        docstringStyle: 'google',
        filters: { private: true },
        sourceLink: { host: 'github', repo: 'uncscode/particula', ref: 'main', root: '..' },
      }],
    })],
    sidebar: navigation.map((group) => group.label === 'API'
      ? { label: 'API', collapsed: true, items: [{ label: 'API overview', slug: 'api' }, pydocsSidebarGroup] }
      : group),
  })],
});
