import test from 'node:test';
import assert from 'node:assert/strict';
import { ancestors, children, searchTopics, topicPath, resourcePath } from '../src/components/topic-model.mjs';

test('recognizable phrases retrieve the intended scientific subject', () => {
  for (const [query, id] of [['humidity growth', 'condensation'], ['distribution normalization', 'size'], ['DNS', 'turbulent'], ['air viscosity', 'transport'], ['vapour pressure', 'vapor'], ['Mie', 'optics']]) {
    assert.equal(searchTopics(query)[0].topic.id, id, query);
  }
  assert.deepEqual(searchTopics(''), []);
  assert.deepEqual(searchTopics('unfindable-nonsense-xxx'), []);
});
test('topic paths and resource links respect the deployment base', () => {
  assert.equal(topicPath('aerosols'), '/particula/guide/topics/');
  assert.equal(topicPath('condensation', '/'), '/guide/topics/condensation/');
  assert.equal(resourcePath('docs/index.md'), '/particula/');
  assert.equal(resourcePath('docs/guide/gpu/index.md#support'), '/particula/guide/gpu/#support');
  assert.equal(resourcePath('docs/examples/demo.py'), '/particula/examples/demo/');
});
test('hierarchy has predictable root, children, and breadcrumbs', () => {
  assert.deepEqual(children('aerosols').map(t => t.id), ['state', 'processes', 'methods', 'applications']);
  assert.deepEqual(ancestors('condensation').map(t => t.id), ['aerosols', 'processes', 'exchange', 'condensation']);
  assert.deepEqual(ancestors('missing'), []);
});


test('organized topics remain discoverable beyond four selections from the root', () => {
  assert.deepEqual(ancestors('gpu-foundations-warp').map(t => t.id), [
    'aerosols', 'methods', 'execution', 'gpu', 'gpu-foundations', 'gpu-foundations-warp',
  ]);
  assert.equal(searchTopics('graph capture')[0].topic.id, 'gpu-resident');
  assert.equal(searchTopics('GPU slots')[0].topic.id, 'gpu-foundations-data');
  assert.deepEqual(children('gpu').map(t => t.id), [
    'gpu-direct', 'gpu-prepared', 'gpu-resident', 'gpu-foundations',
  ]);
});
