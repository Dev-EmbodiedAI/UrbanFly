import * as THREE from 'three';

const EVENT_STYLES = {
  temporary_obstacle: { color: 0xff9b42, opacity: 0.28 },
  no_fly_zone: { color: 0xff4d6d, opacity: 0.20 },
  weather_hazard: { color: 0x48cae4, opacity: 0.18 },
  drone_failure: { color: 0xffd166, opacity: 0.42 },
  goal_landmark: { color: 0x80ed99, opacity: 0.34 },
};

function finiteVector(value) {
  return Array.isArray(value)
    && value.length >= 3
    && value.slice(0, 3).every((number) => Number.isFinite(Number(number)));
}

function disposeObject(object) {
  object.traverse((child) => {
    child.geometry?.dispose();
    if (Array.isArray(child.material)) child.material.forEach((material) => material.dispose());
    else child.material?.dispose();
  });
}

/**
 * Presentation-only overlays for accepted semantic events. These objects are
 * deliberately excluded from the RGB-D sensor layer: a VLM must infer events
 * from the real scene, not from its own debug annotation.
 */
export class SemanticFleetRenderer {
  constructor(scene) {
    this.scene = scene;
    this.objects = new Map();
    this.group = new THREE.Group();
    this.group.name = 'UrbanFlySemanticFleetOverlays';
    this.group.userData.excludeFromSensors = true;
    this.scene.add(this.group);
  }

  update(snapshot) {
    const events = snapshot?.enabled ? (snapshot.active_events || []) : [];
    const alive = new Set();
    for (const event of events) {
      if (!event?.event_id || !finiteVector(event.position)) continue;
      const id = String(event.event_id);
      alive.add(id);
      const signature = JSON.stringify([
        event.event_type,
        event.position.slice(0, 3).map(Number),
        Number(event.radius_m || 1),
        Number(event.severity || 0),
      ]);
      let record = this.objects.get(id);
      if (!record || record.signature !== signature) {
        if (record) {
          this.group.remove(record.object);
          disposeObject(record.object);
        }
        const object = this._create(event);
        record = { object, signature };
        this.objects.set(id, record);
        this.group.add(object);
      }
      record.object.userData.event = event;
    }
    for (const [id, record] of this.objects) {
      if (alive.has(id)) continue;
      this.group.remove(record.object);
      disposeObject(record.object);
      this.objects.delete(id);
    }
  }

  _create(event) {
    const style = EVENT_STYLES[event.event_type] || EVENT_STYLES.goal_landmark;
    const radius = Math.max(1, Number(event.radius_m || 1));
    const [x, y, z] = event.position.map(Number);
    const material = new THREE.MeshBasicMaterial({
      color: style.color,
      transparent: true,
      opacity: style.opacity,
      depthWrite: false,
      side: THREE.DoubleSide,
    });
    const wireMaterial = new THREE.LineBasicMaterial({
      color: style.color,
      transparent: true,
      opacity: 0.82,
    });
    const root = new THREE.Group();
    root.name = `SemanticEvent_${event.event_type}_${event.event_id}`;
    root.userData.excludeFromSensors = true;

    let geometry;
    let body;
    if (event.event_type === 'no_fly_zone') {
      const height = Math.max(80, y * 2, 160);
      geometry = new THREE.CylinderGeometry(radius, radius, height, 48, 1, true);
      body = new THREE.Mesh(geometry, material);
      body.position.y = height * 0.5 - y;
    } else if (event.event_type === 'weather_hazard') {
      geometry = new THREE.SphereGeometry(radius, 32, 18);
      body = new THREE.Mesh(geometry, material);
    } else if (event.event_type === 'drone_failure') {
      geometry = new THREE.OctahedronGeometry(Math.max(3, radius * 2), 1);
      body = new THREE.Mesh(geometry, material);
    } else if (event.event_type === 'goal_landmark') {
      geometry = new THREE.TorusGeometry(radius, Math.max(0.35, radius * 0.08), 10, 48);
      body = new THREE.Mesh(geometry, material);
      body.rotation.x = Math.PI / 2;
    } else {
      geometry = new THREE.CylinderGeometry(radius, radius, Math.max(12, radius * 1.5), 32);
      body = new THREE.Mesh(geometry, material);
    }
    root.add(body);

    const edges = new THREE.LineSegments(new THREE.EdgesGeometry(geometry), wireMaterial);
    edges.position.copy(body.position);
    edges.rotation.copy(body.rotation);
    root.add(edges);
    root.position.set(x, y, z);
    return root;
  }

  dispose() {
    for (const { object } of this.objects.values()) disposeObject(object);
    this.objects.clear();
    this.group.removeFromParent();
  }
}

export { EVENT_STYLES };
