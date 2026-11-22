import test from 'node:test';
import assert from 'node:assert/strict';
import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkDocLinks from './remark-doc-links.mjs';

test('keeps published-site links in the local preview without changing literals', () => {
  const value = '[API][reference]\n\n[reference]: https://uncscode.github.io/particula/API/particula/aerosol/?view=all#Aerosol\n\n```text\nhttps://uncscode.github.io/particula/API/\n```\n\n[Elsewhere](https://uncscode.github.io/another-project/)';
  const processor = unified().use(remarkParse).use(remarkDocLinks);
  const tree = processor.runSync(processor.parse(value), { path: '/site/src/content/docs/examples/test.md' });
  assert.equal(tree.children[1].url, '/particula/API/particula/aerosol/?view=all#Aerosol');
  assert.equal(tree.children[2].value, 'https://uncscode.github.io/particula/API/');
  assert.equal(tree.children[3].children[0].url, 'https://uncscode.github.io/another-project/');
});

test('gives registered TOCs their print size and leaves report plots at native size', () => {
  const processor = unified().use(remarkParse).use(remarkDocLinks);
  const source = '![TOC](/particula/images/dunne-cloud/formation-toc.png)\n\n![Data](/particula/images/dunne-cloud/warm-ensemble.png)';
  const tree = processor.runSync(processor.parse(source), { path: '/site/src/content/docs/studies/test.md' });
  assert.deepEqual(tree.children[0].children[0].data.hProperties, { width: 288, height: 288 });
  assert.equal(tree.children[1].children[0].data, undefined);
});

test('groups the registered study image with its abstract, preserving footnotes and links', () => {
  const processor = unified().use(remarkParse).use(remarkDocLinks);
  const value = '![Droplets](/particula/images/pi-chamber-lidar/graphical-abstract.svg)[^emblem]\n\nRead the [measured gain](#gain).\n\n## Next\n\n[^emblem]: Attribution';
  const tree = processor.parse(value);
  const [figure, abstract, heading] = tree.children;
  processor.runSync(tree, { path: '/site/src/content/docs/studies/literature/pi-chamber-lidar/index.md' });
  assert.equal(tree.children[0].data.hName, 'section');
  assert.deepEqual(tree.children[0].children, [figure, abstract]);
  assert.equal(tree.children[1], heading);
});

test('pairs a Python study after its execution banner, only on its own page', () => {
  const processor = unified().use(remarkParse).use(remarkDocLinks);
  const value = 'Archived run.\n\n## Comparison\n\n![Growth](/particula/images/glyoxal-chamber/graphical-abstract.svg)\n\nMeasured growth.\n\nSecond paragraph.';
  const study = 'studies/literature/evaporation-and-uptake/glyoxal_chamber.md';
  const render = (source, page) => processor.runSync(processor.parse(source), { path: '/site/src/content/docs/' + page });
  assert.deepEqual(render(value, study).children.map(n => n.type), ['paragraph', 'heading', 'studyIntro', 'paragraph']);
  assert.ok(render(value, 'studies/literature/index.md').children.every(n => n.type !== 'studyIntro'));
  assert.ok(render(value.replace('Measured growth.', '## Heading'), study).children.every(n => n.type !== 'studyIntro'));
});
