import test from 'node:test';
import assert from 'node:assert/strict';
import { BusinessNodeStore } from '../src/semantic_annotations.js';

test('business node store numbers nodes by Chinese category and preserves world coordinates', () => {
  const store = new BusinessNodeStore();
  const supply = store.add('supply', [1, 2, 3]);
  const delivery = store.add('delivery', [-4, 5, -6]);
  assert.equal(supply.id, '供货点_001');
  assert.equal(delivery.id, '送货点_001');
  assert.deepEqual(store.document().nodes[1].position, [-4, 5, -6]);
  assert.deepEqual(store.document().axis_order, ['east', 'up', 'north']);
});

test('business node store supports edit, delete, undo, and per-type limits', () => {
  const store = new BusinessNodeStore();
  const node = store.add('resupply', [0, 10, 0]);
  store.update(node.id, { position: [3, 11, 4], qa_status: 'UNCHECKED' });
  assert.deepEqual(store.nodes.get(node.id).position, [3, 11, 4]);
  store.remove(node.id);
  assert.equal(store.nodes.size, 0);
  store.undo();
  assert.deepEqual(store.nodes.get(node.id).position, [3, 11, 4]);
  store.undo();
  assert.deepEqual(store.nodes.get(node.id).position, [0, 10, 0]);
  const full = new BusinessNodeStore(Array.from({ length: 20 }, (_, index) => ({
    id: `补给点_${String(index + 1).padStart(3, '0')}`,
    type: 'resupply', position: [index, 0, index], qa_status: 'UNCHECKED',
  })));
  assert.throws(() => full.add('resupply', [0, 0, 0]), /上限/);
});

test('changing category assigns a category-consistent id and undo restores it', () => {
  const store = new BusinessNodeStore();
  const node = store.add('supply', [1, 2, 3]);
  const changed = store.update(node.id, { type: 'delivery' });
  assert.equal(changed.id, '送货点_001');
  assert.equal(store.nodes.has('供货点_001'), false);
  assert.equal(store.nodes.get('送货点_001').type, 'delivery');
  store.undo();
  assert.equal(store.nodes.has('供货点_001'), true);
  assert.equal(store.nodes.get('供货点_001').type, 'supply');
});

test('replace rejects imported documents over a category limit', () => {
  assert.throws(() => new BusinessNodeStore(Array.from({ length: 21 }, (_, index) => ({
    id: `补给点_${String(index + 1).padStart(3, '0')}`,
    type: 'resupply',
    position: [index, 0, index],
  }))), /超过/);
});
