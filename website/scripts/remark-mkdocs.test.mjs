import test from 'node:test';
import assert from 'node:assert/strict';
import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkMkdocs from './remark-mkdocs.mjs';

const transform = (value) => {
  const processor = unified().use(remarkParse).use(remarkMkdocs);
  return processor.runSync(processor.parse(value), { value });
};

test('preserves the body and title of an indented MkDocs note', () => {
  const tree = transform('!!! note "Wall losses"\n    Size-dependent settling.\n    Preserve the second line.');
  assert.equal(tree.children[0].data.hName, 'aside');
  assert.equal(tree.children[0].children[0].children[0].value, 'Wall losses');
  assert.match(JSON.stringify(tree), /Size-dependent settling/);
  assert.match(JSON.stringify(tree), /Preserve the second line/);
});

test('does not rewrite literal examples inside code fences', () => {
  const tree = transform('```text\n!!! note "Literal"\n    Body\n```');
  assert.equal(tree.children[0].type, 'code');
  assert.match(tree.children[0].value, /!!! note/);
});

test('retains explicit heading IDs and expanded disclosure state', () => {
  const tree = transform('## Science {#legacy-science}\n\n???+ note "Details"\n    Content');
  assert.equal(tree.children[0].data.hProperties.id, 'legacy-science');
  assert.equal(tree.children[1].data.hName, 'details');
  assert.equal(tree.children[1].data.hProperties.open, true);
});
