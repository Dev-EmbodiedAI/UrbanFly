import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import { PathVisualizer } from '../src/path.js';
import { PresentationBudget } from '../src/presentation_budget.js';
import { DroneSensorSuite } from '../src/drone_sensors.js';
import { RuntimeHealth } from '../src/runtime_health.js';
import { hostLifecycle, bindHostLifecycle, presentationHidden } from '../src/host_lifecycle.js';
import { DigitalTwinRenderer } from '../src/digital_twin.js';
import { SceneManager } from '../src/scene.js';
import { SemanticFleetRenderer } from '../src/semantic_fleet.js';

test('trace reuses GPU objects and preserves the last 6000 points in order', () => {
  const scene = new THREE.Scene();
  const paths = new PathVisualizer(scene);
  paths.updateExecutedTrace('uav', [0, 0, 0]);
  const trace = paths.executedTraces.get('uav');
  const line = trace.line;
  const geometry = line.geometry;
  const material = line.material;
  for (let i = 1; i <= 13000; i++) paths.updateExecutedTrace('uav', [i, 2, 3]);
  assert.equal(trace.line, line);
  assert.equal(line.geometry, geometry);
  assert.equal(line.material, material);
  assert.equal(scene.children.length, 1);
  assert.equal(trace.count, 6000);
  assert.equal(trace.attribute.array.length, 36000);
  assert.equal(trace.attribute.updateRanges.length, 0);
  const { start, count } = geometry.drawRange;
  for (let i = 0; i < count; i++) assert.equal(trace.attribute.getX(start + i), 7001 + i);
  let disposed = 0;
  geometry.addEventListener('dispose', () => disposed++);
  material.addEventListener('dispose', () => disposed++);
  paths.clearAll();
  assert.equal(disposed, 2);
  assert.equal(scene.children.length, 0);
});

test('trace rejects invalid and stationary points', () => {
  const paths = new PathVisualizer(new THREE.Scene());
  paths.updateExecutedTrace('uav', [NaN, 0, 0]);
  assert.equal(paths.executedTraces.size, 0);
  paths.updateExecutedTrace('uav', [0, 0, 0]);
  paths.updateExecutedTrace('uav', [0.01, 0, 0]);
  assert.equal(paths.executedTraces.get('uav').count, 1);
});

test('path throttle runs before touching waypoint coordinates', () => {
  const paths = new PathVisualizer(new THREE.Scene());
  paths.pathMetadata.set('uav', { updatedAt: performance.now(), signature: 'old' });
  paths.updatePath('uav', [{ get pos() { throw new Error('should be throttled'); } }, {}]);
});

test('display budgets prioritize sensors and skip hidden display work', () => {
  const budget = new PresentationBudget();
  assert.equal(budget.shouldRender(0, { busy: true }), true);
  assert.equal(budget.shouldRender(16.7, { busy: true }), false);
  assert.equal(budget.shouldRender(33.4, { busy: true }), true);
  assert.equal(budget.targetFps, 30);
  assert.equal(budget.shouldRender(100, { hidden: true }), false);
  assert.equal(budget.shouldRender(101, {}), true);
  assert.equal(budget.targetFps, 60);
  assert.equal(budget.shouldRender(200, { idle: true, interacting: true }), true);
  assert.equal(budget.targetFps, 60);
});

test('render metrics are bounded and stale samples expire', () => {
  const budget = new PresentationBudget();
  for (let i = 0; i < 1000; i++) budget.record(i * 20, 4);
  assert.equal(budget.frames.length, 300);
  assert.equal(budget.statistics(19980).fps, 50);
  assert.equal(budget.statistics(19980).render_submission_p95_ms, 4);
  assert.equal(budget.statistics(30000).frames_in_window, 0);
});

test('30 FPS pacing does not collapse to 25 FPS on a 50 Hz host', () => {
  const budget = new PresentationBudget();
  let frames = 0;
  for (let now = 0; now < 10000; now += 20) {
    if (budget.shouldRender(now, { idle: true })) frames++;
  }
  assert.ok(frames >= 299 && frames <= 301, `unexpected frame count ${frames}`);
});

test('policy bridge never takes ownership of manual streaming', () => {
  const suite = Object.create(DroneSensorSuite.prototype);
  suite.streamingEnabled = false;
  const api = suite._createPublicApi();
  api.setBridgeEnabled(true);
  api.setBridgeEnabled(false);
  assert.equal(suite.streamingEnabled, false);
  api.startStreaming();
  api.setBridgeEnabled(true);
  api.setBridgeEnabled(false);
  assert.equal(suite.streamingEnabled, true);
  api.stopStreaming();
  assert.equal(suite.streamingEnabled, false);
});

test('third-person recording capture is explicit and bounded', () => {
  const suite = Object.create(DroneSensorSuite.prototype);
  suite.thirdPersonCapture = { enabled: false, distance_m: 22, height_m: 9 };
  const api = suite._createPublicApi();
  api.setThirdPersonCapture({ enabled: true, distance_m: 999, height_m: -5 });
  assert.deepEqual(suite.thirdPersonCapture, { enabled: true, distance_m: 60, height_m: 4 });
  api.setThirdPersonCapture({ enabled: false });
  assert.equal(suite.thirdPersonCapture.enabled, false);
});

test('health polling is single-flight and abortable', async () => {
  const original = { window: globalThis.window, document: globalThis.document, fetch: globalThis.fetch };
  globalThis.window = { location: { search: '' } };
  globalThis.document = { getElementById: () => null, removeEventListener() {} };
  let calls = 0;
  let pending;
  globalThis.fetch = (_url, options) => {
    calls++;
    return new Promise((_resolve, reject) => {
      pending = options.signal;
      options.signal.addEventListener('abort', () => reject(new Error('aborted')));
    });
  };
  try {
    const health = new RuntimeHealth({ endpoint: '/api/health', network: { connected: false }, sensors: {} });
    const first = health.refresh();
    await health.refresh();
    assert.equal(calls, 1);
    health.stop();
    await first;
    assert.equal(pending.aborted, true);
    assert.equal(health.refreshActive, false);
  } finally {
    Object.assign(globalThis, original);
  }
});

test('native minimize hides presentation even if Page Visibility says visible', () => {
  const original = { window: globalThis.window, document: globalThis.document };
  let receive;
  globalThis.window = { chrome: { webview: { addEventListener: (_type, handler) => { receive = handler; } } } };
  globalThis.document = { hidden: false };
  try {
    bindHostLifecycle();
    receive({ data: { type: 'host_visibility', hidden: true } });
    assert.equal(presentationHidden(), true);
    receive({ data: { type: 'unrelated', hidden: false } });
    assert.equal(presentationHidden(), true);
    receive({ data: { type: 'host_visibility', hidden: false } });
    assert.equal(presentationHidden(), false);
  } finally {
    hostLifecycle.hidden = false;
    Object.assign(globalThis, original);
  }
});

test('unchanged stopped view does not redraw; invalidation restores presentation', () => {
  const original = globalThis.document;
  globalThis.document = { hidden: false };
  try {
    const scene = new SceneManager({}, {});
    scene.displayDirty = false;
    assert.equal(scene.shouldRender(100, { idle: true, busy: false }), false);
    assert.equal(scene.displayIdle, true);
    scene.invalidate();
    assert.equal(scene.shouldRender(101, { idle: true, busy: false }), true);
    scene.displayDirty = false;
    assert.equal(scene.shouldRender(200, { idle: false, busy: true }), true);
  } finally { globalThis.document = original; }
});

test('asset warmup uploads shared texture once and compiles against real scene lights', async () => {
  const twin = Object.create(DigitalTwinRenderer.prototype);
  const texture = new THREE.Texture();
  const group = new THREE.Group();
  for (let i = 0; i < 2; i++) group.add(new THREE.Mesh(
    new THREE.BufferGeometry(), new THREE.MeshBasicMaterial({ map: texture }),
  ));
  let uploads = 0;
  let compiles = 0;
  twin.scene = new THREE.Scene();
  twin.sceneManager = { camera: new THREE.PerspectiveCamera(), renderer: {
    initTexture(value) { assert.equal(value, texture); uploads++; },
    async compileAsync(root, camera, scene) {
      assert.equal(root, group);
      assert.equal(camera, twin.sceneManager.camera);
      assert.equal(scene, twin.scene);
      compiles++;
    },
  } };
  await twin._warmVisualScene(group);
  assert.equal(uploads, 1);
  assert.equal(compiles, 1);
});

test('semantic overlays are presentation-only, stable, and removed on event expiry', () => {
  const scene = new THREE.Scene();
  const renderer = new SemanticFleetRenderer(scene);
  const snapshot = {
    enabled: true,
    active_events: [{
      event_id: 'nf-1',
      event_type: 'no_fly_zone',
      position: [10, 30, -20],
      radius_m: 25,
      severity: 0.8,
    }],
  };
  renderer.update(snapshot);
  assert.equal(renderer.group.userData.excludeFromSensors, true);
  assert.equal(renderer.objects.size, 1);
  const first = renderer.objects.get('nf-1').object;
  assert.deepEqual(first.position.toArray(), [10, 30, -20]);
  renderer.update(snapshot);
  assert.equal(renderer.objects.get('nf-1').object, first);

  let disposed = 0;
  first.traverse((child) => child.geometry?.addEventListener('dispose', () => disposed++));
  renderer.update({ enabled: true, active_events: [] });
  assert.equal(renderer.objects.size, 0);
  assert.ok(disposed >= 2);
});

test('semantic overlays reject malformed coordinates without leaking scene objects', () => {
  const renderer = new SemanticFleetRenderer(new THREE.Scene());
  renderer.update({
    enabled: true,
    active_events: [{
      event_id: 'bad', event_type: 'weather_hazard', position: [NaN, 0, 0], radius_m: 10,
    }],
  });
  assert.equal(renderer.objects.size, 0);
  assert.equal(renderer.group.children.length, 0);
});
