import * as THREE from 'three';

export class DynamicActorRenderer {
  constructor(scene) {
    this.scene = scene;
    this.objects = new Map();
    this.group = new THREE.Group();
    this.group.name = 'UrbanFlyDynamicActors';
    this.scene.add(this.group);
  }

  update(actorStates = []) {
    const alive = new Set();
    for (const actor of actorStates) {
      const id = String(actor.id);
      alive.add(id);
      let object = this.objects.get(id);
      if (!object) {
        object = this._create(actor);
        this.objects.set(id, object);
        this.group.add(object);
      }
      const [x, y, z] = actor.pos;
      object.position.set(x, y, z);
      const [vx, , vz] = actor.vel || [0, 0, 0];
      if (Math.hypot(vx, vz) > 1e-4) object.rotation.y = Math.atan2(vx, vz);
      object.userData.actorState = actor;
    }
    for (const [id, object] of this.objects) {
      if (alive.has(id)) continue;
      this.group.remove(object);
      object.traverse((child) => {
        child.geometry?.dispose();
        child.material?.dispose();
      });
      this.objects.delete(id);
    }
  }

  _create(actor) {
    const pedestrian = actor.actor_type === 'pedestrian';
    const material = new THREE.MeshStandardMaterial({
      color: pedestrian ? 0x8a63ff : 0xffa52f,
      roughness: 0.72,
      metalness: pedestrian ? 0 : 0.08,
    });
    let object;
    if (pedestrian) {
      object = new THREE.Mesh(new THREE.CapsuleGeometry(0.32, 1.15, 5, 10), material);
    } else {
      const extent = actor.bbox_extent || [2.2, 0.9, 0.95];
      object = new THREE.Mesh(
        new THREE.BoxGeometry(extent[0] * 2, extent[1] * 2, extent[2] * 2),
        material,
      );
    }
    object.castShadow = false;
    object.receiveShadow = true;
    object.name = `DynamicActor_${actor.id}`;
    return object;
  }
}
