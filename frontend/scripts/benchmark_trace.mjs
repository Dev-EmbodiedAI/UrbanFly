import * as THREE from 'three';
import { PathVisualizer } from '../src/path.js';
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname } from 'node:path';

// Reproduce the pre-optimization update algorithm, CPU only (no GPU/context).
function legacy(count) {
  const points = [];
  let line;
  for (let i = 0; i < count; i++) {
    points.push(new THREE.Vector3(i, 2, 3));
    if (line) { line.geometry.dispose(); line.material.dispose(); }
    line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(points),
      new THREE.LineBasicMaterial({ color: 0x29ff78, transparent: true, opacity: 0.95 }));
  }
  line.geometry.dispose(); line.material.dispose();
}
function current(count) {
  const paths = new PathVisualizer(new THREE.Scene());
  for (let i = 0; i < count; i++) paths.updateExecutedTrace('probe', [i, 2, 3]);
  paths.clearAll();
}
const measure = (callback, count) => {
  const start = performance.now(); callback(count); return performance.now() - start;
};
legacy(100); current(100);
const report = { kind: 'CPU microbenchmark; not a display FPS claim', points: 3000,
  legacy_ms: [], fixed_buffer_ms: [] };
for (let run = 0; run < 5; run++) {
  report.legacy_ms.push(measure(legacy, report.points));
  report.fixed_buffer_ms.push(measure(current, report.points));
}
report.legacy_median_ms = [...report.legacy_ms].sort((a, b) => a - b)[2];
report.fixed_buffer_median_ms = [...report.fixed_buffer_ms].sort((a, b) => a - b)[2];
console.log(JSON.stringify(report, null, 2));
if (process.argv[2]) {
  mkdirSync(dirname(process.argv[2]), { recursive: true });
  writeFileSync(process.argv[2], JSON.stringify(report, null, 2), { flag: 'wx' });
}
