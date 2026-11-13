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
