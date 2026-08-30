import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import { CityDisplayLod, SENSOR_CITY_LAYER, DISPLAY_CITY_LAYER } from '../src/city_display_lod.js';
import { RenderBenchmark } from '../src/render_benchmark.js';
import { SceneManager } from '../src/scene.js';

const meshGroup = () => {
  const root = new THREE.Group();
  root.add(new THREE.Mesh(new THREE.BoxGeometry(), new THREE.MeshBasicMaterial()));
  return root;
};
const bounds = { minimum: [0, 0, 0], maximum: [250, 50, 250] };

test('sensor always sees original high mesh, never overview across LOD switches', () => {
  const lod = new CityDisplayLod();
  const high = meshGroup(), overview = meshGroup();
  const geometry = high.children[0].geometry;
  const material = high.children[0].material;
  const sensor = new THREE.PerspectiveCamera();
  sensor.layers.enable(SENSOR_CITY_LAYER);
  const display = new THREE.PerspectiveCamera();
  display.layers.enable(DISPLAY_CITY_LAYER);
  lod.addOverview('tile', overview);
  lod.addHigh('tile', high, bounds);
  for (const enabled of [true, false]) for (const fullDetail of [true, false]) {
    lod.enabled = enabled;
    for (const x of [100, 400, 1000, -1000]) {
      lod.update(new THREE.Vector3(x, 30, 100), { fullDetail });
      assert.equal(sensor.layers.test(high.children[0].layers), true);
      assert.equal(sensor.layers.test(overview.children[0].layers), false);
      assert.notEqual(display.layers.test(high.children[0].layers), display.layers.test(overview.children[0].layers));
      assert.equal(high.visible, true);
      assert.equal(high.children[0].geometry, geometry);
      assert.equal(high.children[0].material, material);
    }
  }
});

test('distance to tile bounds and hysteresis prevent repeated boundary switches', () => {
  const lod = new CityDisplayLod();
  lod.addOverview('tile', meshGroup());
  lod.addHigh('tile', meshGroup(), bounds);
  const move = (x) => { lod.update(new THREE.Vector3(x, 20, 100)); return lod.statistics().display_high_tiles; };
  assert.equal(move(1000), 0);
  assert.equal(move(450), 1);
  assert.equal(move(480), 1);
  assert.equal(move(500), 0);
  assert.equal(move(480), 0);
});

test('missing overview falls back to high; loading overview stays display-only', () => {
  const lod = new CityDisplayLod();
  lod.addHigh('high-only', meshGroup(), bounds);
  lod.addOverview('loading', meshGroup());
  lod.update(new THREE.Vector3(10000, 10000, 10000));
  assert.equal(lod.statistics().display_high_tiles, 1);
  assert.equal(lod.statistics().display_overview_tiles, 1);
  assert.equal(lod.statistics().sensor_high_tiles, 1);
});

test('render benchmark measures real intervals, excludes warmup and terminates', () => {
  const benchmark = new RenderBenchmark({ warmupMs: 100, durationMs: 1000 });
  benchmark.start(0);
  let report;
  for (let now = 0; now <= 1100; now += 20) report = benchmark.record(now, 3, 256, 300000);
  assert.equal(benchmark.active, false);
  assert.equal(report.fps, 50);
  assert.equal(report.frames, 51);
  assert.equal(report.frame_interval_p95_ms, 20);
  assert.equal(report.render_cpu_p95_ms, 3);
  assert.equal(report.draw_calls_median, 256);
  assert.equal(benchmark.record(1200, 3, 256, 300000), null);
});

test('cancelling benchmark cannot produce a qualified report', () => {
  const benchmark = new RenderBenchmark();
  benchmark.start(0);
  benchmark.cancel();
  assert.equal(benchmark.record(15000, 1, 1, 1), null);
});

test('display metrics exclude previous sensor draws and include all composer passes', () => {
  const manager = new SceneManager({}, { width: 1280, height: 650 });
  manager.camera = new THREE.PerspectiveCamera();
  manager.controls = { update() {} };
  const info = {
    autoReset: true, render: { calls: 500, triangles: 1000000 },
    reset() { this.render.calls = 0; this.render.triangles = 0; },
  };
  manager.renderer = { info, render() {
    if (info.autoReset) info.reset();
    info.render.calls += 10;
    info.render.triangles += 100;
  } };
  manager.render();
  assert.equal(manager.lastDrawCalls, 10);
  assert.equal(manager.lastTriangles, 100);
  assert.equal(info.autoReset, true);
  manager.quality = 'detail';
  manager.composer = { render() { manager.renderer.render(); manager.renderer.render(); } };
  manager.render();
  assert.equal(manager.lastDrawCalls, 20);
  assert.equal(manager.lastTriangles, 200);
  assert.equal(info.autoReset, true);
  manager.composer.render = () => { throw new Error('render failure'); };
  assert.throws(() => manager.render(), /render failure/);
  assert.equal(info.autoReset, true);
});
