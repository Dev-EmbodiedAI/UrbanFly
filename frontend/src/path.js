/**
 * 路径可视化
 * ==========
 * 用更克制的航迹带和关键节点标识来表达飞行计划。
 */

import * as THREE from 'three';

export class PathVisualizer {
  constructor(scene) {
    this.scene = scene;
    this.paths = new Map();
    this.pathMetadata = new Map();
    this.worldModelPredictions = new Map();
    this.executedTraces = new Map();
    this.maxPaths = 30;
    this.pathUpdateIntervalMs = 250;
    this.worldModelUpdateIntervalMs = 200;
  }

  updateExecutedTrace(droneId, position) {
    if (!Array.isArray(position) || position.length !== 3) return;
    const point = position.map(Number);
    if (!point.every(Number.isFinite)) return;
    let trace = this.executedTraces.get(droneId);
    if (!trace) {
      // Mirror the ring so a single contiguous draw range preserves chronology
      // across wraparound, without rebuilding geometry or shifting 6000 points.
      const capacity = 6000;
      const attribute = new THREE.BufferAttribute(new Float32Array(capacity * 6), 3);
      attribute.setUsage(THREE.DynamicDrawUsage);
      const geometry = new THREE.BufferGeometry().setAttribute('position', attribute);
      const line = new THREE.Line(geometry, new THREE.LineBasicMaterial({
        color: 0x29ff78, transparent: true, opacity: 0.95,
      }));
      line.renderOrder = 20;
      line.frustumCulled = false;
      line.userData.excludeFromSensors = true;
      trace = { capacity, attribute, line, count: 0, next: 0, previous: null };
      this.scene.add(line);
      this.executedTraces.set(droneId, trace);
    }
    if (trace.previous && point.reduce(
      (sum, value, axis) => sum + (value - trace.previous[axis]) ** 2, 0,
    ) < 0.01) return;
    for (const offset of [trace.next * 3, (trace.next + trace.capacity) * 3]) {
      trace.attribute.array.set(point, offset);
    }
    // A full bounded upload also avoids unbounded updateRanges while hidden.
    trace.attribute.needsUpdate = true;
    trace.previous = point;
    trace.next = (trace.next + 1) % trace.capacity;
    trace.count = Math.min(trace.count + 1, trace.capacity);
    trace.line.geometry.setDrawRange(trace.count < trace.capacity ? 0 : trace.next, trace.count);
  }

  updatePath(droneId, waypoints, colorHex = 0x58b4ff) {
    if (!waypoints || waypoints.length < 2) {
      this._removePath(droneId);
      return;
    }

    const now = performance.now();
    const previousMetadata = this.pathMetadata.get(droneId);
    if (previousMetadata && now - previousMetadata.updatedAt < this.pathUpdateIntervalMs) return;
    const signature = `${colorHex}:${
      waypoints.map((waypoint) => {
        const point = Array.isArray(waypoint) ? waypoint : waypoint.pos;
        return point.map((value) => Math.round(Number(value) * 4)).join(',');
      }).join('|')
    }`;
    if (previousMetadata?.signature === signature) return;

    this._removePath(droneId);

    const points = waypoints.map((w) => {
      if (Array.isArray(w)) return new THREE.Vector3(w[0], w[1], w[2]);
      if (w.pos) return new THREE.Vector3(w.pos[0], w.pos[1], w.pos[2]);
      return new THREE.Vector3(w[0], w[1], w[2]);
    });

    const curve = new THREE.CatmullRomCurve3(points, false, 'catmullrom', 0.45);
    const segments = Math.min(240, Math.max(80, waypoints.length * 10));

    const core = new THREE.Mesh(
      new THREE.TubeGeometry(curve, segments, 0.22, 8, false),
      new THREE.MeshStandardMaterial({
        color: colorHex,
        emissive: colorHex,
        emissiveIntensity: 0.42,
        roughness: 0.24,
        metalness: 0.1,
        transparent: true,
        opacity: 0.82,
      })
    );

    const shell = new THREE.Mesh(
      new THREE.TubeGeometry(curve, Math.floor(segments * 0.66), 0.58, 8, false),
      new THREE.MeshBasicMaterial({
        color: colorHex,
        transparent: true,
        opacity: 0.09,
        depthWrite: false,
      })
    );

    const markers = new THREE.Group();
    core.userData.excludeFromSensors = true;
    shell.userData.excludeFromSensors = true;
    markers.userData.excludeFromSensors = true;
    const markerMat = new THREE.MeshStandardMaterial({
      color: 0xe9f6ff,
      emissive: 0x8ed4ff,
      emissiveIntensity: 0.7,
      roughness: 0.22,
      metalness: 0.12,
    });

    const startMarker = new THREE.Mesh(new THREE.SphereGeometry(0.42, 12, 12), markerMat);
    startMarker.position.copy(points[0]);
    markers.add(startMarker);

    const endMarker = new THREE.Mesh(
      new THREE.OctahedronGeometry(0.55, 0),
      new THREE.MeshStandardMaterial({
        color: colorHex,
        emissive: colorHex,
        emissiveIntensity: 0.95,
        roughness: 0.16,
        metalness: 0.12,
      })
    );
    endMarker.position.copy(points[points.length - 1]);
    markers.add(endMarker);

    for (let i = 1; i < points.length - 1; i++) {
      const waypoint = new THREE.Mesh(
        new THREE.SphereGeometry(0.18, 8, 8),
        new THREE.MeshBasicMaterial({
          color: 0xc1e6ff,
          transparent: true,
          opacity: 0.65,
        })
      );
      waypoint.position.copy(points[i]);
      markers.add(waypoint);
    }

    this.scene.add(core);
    this.scene.add(shell);
    this.scene.add(markers);

    this.paths.set(droneId, { core, shell, markers });
    this.pathMetadata.set(droneId, { signature, updatedAt: now });
    if (this.paths.size > this.maxPaths) {
      const oldestKey = this.paths.keys().next().value;
      this._removePath(oldestKey);
    }
  }

  _removePath(droneId) {
    const path = this.paths.get(droneId);
    if (!path) return;

    this.scene.remove(path.core);
    this.scene.remove(path.shell);
    this.scene.remove(path.markers);
    this._disposeObject(path.core);
    this._disposeObject(path.shell);
    this._disposeObject(path.markers);
    this.paths.delete(droneId);
    this.pathMetadata.delete(droneId);
  }

  updateWorldModel(droneId, currentPosition, worldModel) {
    if (!worldModel?.enabled || !worldModel.selected_trajectory_world_m?.length) {
      this.clearWorldModel(droneId);
      return;
    }
    // Candidate rollouts remain available in telemetry and in the dedicated
    // World Model panel. Long-range recordings keep them out of the shared
    // 3-D scene so the stable global route and factual trace stay readable.
    if (worldModel.scene_candidate_overlay === false) {
      this.clearWorldModel(droneId);
      return;
    }
    const existing = this.worldModelPredictions.get(droneId);
    if (existing?.sequence === worldModel.decision_sequence) return;
    const now = performance.now();
    if (
      existing
      && now - existing.updatedAt < this.worldModelUpdateIntervalMs
    ) {
      return;
    }
    this.clearWorldModel(droneId);

    const group = new THREE.Group();
    group.name = `WorldModelRollouts:${droneId}`;
    group.userData.excludeFromSensors = true;
    const origin = Array.isArray(currentPosition)
      ? new THREE.Vector3(...currentPosition)
      : new THREE.Vector3();

    // The main comparison fixes the candidate set at 15.  Draw every
    // non-selected trajectory so the recording exposes the real search fan,
    // not a cosmetically reduced subset.
    const alternatives = (worldModel.top_candidates || []).slice(1);
    for (const candidate of alternatives) {
      const points = [
        origin,
        ...(candidate.trajectory_world_m || []).map(
          (point) => new THREE.Vector3(...point),
        ),
      ];
      if (points.length < 2) continue;
      const geometry = new THREE.BufferGeometry().setFromPoints(points);
      const line = new THREE.Line(
        geometry,
        new THREE.LineBasicMaterial({
          color: candidate.predicted_collision ? 0xff5263 : 0x6fa8ff,
          transparent: true,
          opacity: candidate.predicted_collision ? 0.38 : 0.28,
          depthWrite: false,
        }),
      );
      group.add(line);
    }

    const selectedPoints = [
      origin,
      ...worldModel.selected_trajectory_world_m.map(
        (point) => new THREE.Vector3(...point),
      ),
    ];
    if (selectedPoints.length >= 2) {
      const selectedCurve = new THREE.CatmullRomCurve3(
        selectedPoints,
        false,
        'catmullrom',
        0.35,
      );
      const selected = new THREE.Mesh(
        new THREE.TubeGeometry(selectedCurve, 48, 0.12, 7, false),
        new THREE.MeshBasicMaterial({
          color: 0xffd166,
          transparent: true,
          opacity: 0.92,
        }),
      );
      group.add(selected);
      const endpoint = new THREE.Mesh(
        new THREE.SphereGeometry(0.34, 10, 10),
        new THREE.MeshBasicMaterial({ color: 0xffe6a3 }),
      );
      endpoint.position.copy(selectedPoints[selectedPoints.length - 1]);
      group.add(endpoint);
    }

    this.scene.add(group);
    this.worldModelPredictions.set(droneId, {
      sequence: worldModel.decision_sequence,
      updatedAt: now,
      group,
    });
  }

  clearWorldModel(droneId) {
    const prediction = this.worldModelPredictions.get(droneId);
    if (!prediction) return;
    this.scene.remove(prediction.group);
    prediction.group.traverse((object) => {
      object.geometry?.dispose?.();
      if (Array.isArray(object.material)) {
        object.material.forEach((material) => material.dispose?.());
      } else {
        object.material?.dispose?.();
      }
    });
    this.worldModelPredictions.delete(droneId);
  }

  clearAll() {
    for (const id of this.paths.keys()) {
      this._removePath(id);
    }
    this.paths.clear();
    for (const id of [...this.worldModelPredictions.keys()]) {
      this.clearWorldModel(id);
    }
    for (const trace of this.executedTraces.values()) {
      if (trace.line) {
        this.scene.remove(trace.line);
        trace.line.geometry.dispose();
        trace.line.material.dispose();
      }
    }
    this.executedTraces.clear();
  }

  _disposeObject(root) {
    const disposedGeometries = new Set();
    const disposedMaterials = new Set();
    root.traverse((object) => {
      if (object.geometry && !disposedGeometries.has(object.geometry)) {
        disposedGeometries.add(object.geometry);
        object.geometry.dispose();
      }
      const materials = Array.isArray(object.material)
        ? object.material
        : [object.material];
      for (const material of materials) {
        if (!material || disposedMaterials.has(material)) continue;
        disposedMaterials.add(material);
        material.dispose();
      }
    });
  }
}
