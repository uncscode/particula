import test from 'node:test';
import assert from 'node:assert/strict';
import { apiStudySections } from '../src/components/study-api.mjs';

test('API study links follow their named symbol inside the original section', () => {
  const html = '<h1 id="top">Module</h1><section><h2 id="a.B"><code>B</code></h2><p>Docs</p></section>';
  const links = [{ symbol: 'a.B', title: 'First' }, { symbol: 'a.B', title: 'Second' }];
  const sections = apiStudySections(html, links);
  assert.equal(sections.map(item => item.html).join(''), html);
  assert(sections[0].html.endsWith('<code>B</code></h2>'));
  assert.deepEqual(sections[0].related, links);
  assert.deepEqual(sections[1].related, []);
  assert.throws(() => apiStudySections(html, [{ symbol: 'a.Missing' }]), /a.Missing/);
});
