/**
 * 无人机管理器
 * ============
 * 创建、更新和动画化无人机模型。
 */

import * as THREE from 'three';

const DRONE_COLORS = {
  heavy: 0xff705f,
  standard: 0x58b4ff,
  light: 0x57d68f,
};

export class DroneManager {
  constructor(scene) {
    this.scene = scene;
    this.drones = new Map();
    this.rotors = new Map();
    this.payloadIndicators = new Map();
    this.navLights = new Map();

    this.coreGeo = new THREE.BoxGeometry(1.7, 0.34, 1.12);
    this.canopyGeo = new THREE.CapsuleGeometry(0.36, 0.55, 3, 8);
    this.armGeo = new THREE.CylinderGeometry(0.04, 0.06, 1.08, 10);
    this.rotorHubGeo = new THREE.CylinderGeometry(0.08, 0.08, 0.06, 14);
    this.rotorBladeGeo = new THREE.BoxGeometry(0.62, 0.018, 0.09);
    this.rotorDiscGeo = new THREE.CylinderGeometry(0.48, 0.48, 0.016, 28);
    this.landingGeo = new THREE.CylinderGeometry(0.025, 0.025, 0.72, 6);
    this.payloadGeo = new THREE.BoxGeometry(0.4, 0.22, 0.4);
  }

  update(droneStates) {
    const activeIds = new Set(droneStates.map(d => d.id));
    const now = performance.now() * 0.001;

    for (const data of droneStates) {
      let group = this.drones.get(data.id);
      if (!group) {
        group = this._createDrone(data);
        this.drones.set(data.id, group);
      }

      const pos = data.pos;
      const bob = data.state === 'hovering' || data.state === 'idle' ? Math.sin(now * 2.2 + this._hash(data.id)) * 0.12 : 0;
      group.position.set(pos[0], pos[1] + bob, pos[2]);
      if (Array.isArray(data.orientation) && data.orientation.length === 4) {
        const [w, x, y, z] = data.orientation;
        group.quaternion.set(x, y, z, w).normalize();
      } else {
        group.rotation.set(0, THREE.MathUtils.degToRad(data.yaw || 0), 0);
      }

      const rotors = this.rotors.get(data.id);
      if (rotors) {
        for (let index = 0; index < rotors.length; index++) {
          const rotor = rotors[index];
          const omega = data.motor_omega?.[index];
          const visualSpeed = Number.isFinite(omega)
            ? THREE.MathUtils.clamp(omega / 90, 12, 42)
            : 14 + (data.speed || 0) * 2.5;
          const spinDirection = index % 2 === 0 ? 1 : -1;
          rotor.blades.rotation.y += visualSpeed * 0.16 * spinDirection;
          rotor.disc.material.opacity = 0.18 + Math.min(0.22, visualSpeed * 0.005);
        }
      }

      const indicator = this.payloadIndicators.get(data.id);
      if (indicator) {
        const hasPayload = data.payload > 0.1;
        indicator.visible = hasPayload;
        indicator.material.color.setHex(
          data.state === 'picking_up' ? 0xffb24d :
          data.state === 'delivering' ? 0x55ffa2 :
          0x7cd7ff
        );
        indicator.material.emissive.setHex(
          data.state === 'picking_up' ? 0xff8e1a :
          data.state === 'delivering' ? 0x1ad982 :
          0x2aa0ff
        );
      }

      const navLights = this.navLights.get(data.id);
      if (navLights) {
        const pulse = 0.65 + Math.sin(now * 5.5 + this._hash(data.id)) * 0.35;
        navLights.left.material.emissiveIntensity = pulse;
        navLights.right.material.emissiveIntensity = pulse;
      }

      this._updateStateEffect(group, data);
    }

    for (const [id, group] of this.drones) {
      if (!activeIds.has(id)) {
        this.scene.remove(group);
        this._disposeDroneMaterials(group);
        this.drones.delete(id);
        this.rotors.delete(id);
        this.payloadIndicators.delete(id);
        this.navLights.delete(id);
      }
    }
  }

  _disposeDroneMaterials(group) {
    const disposed = new Set();
    group.traverse((object) => {
      const materials = Array.isArray(object.material)
        ? object.material
        : [object.material];
      for (const material of materials) {
        if (!material || disposed.has(material)) continue;
        disposed.add(material);
        material.dispose();
      }
    });
  }

  _createDrone(data) {
    const group = new THREE.Group();
    group.userData.excludeFromSensors = true;
    group.name = `drone-${data.id}`;

    const color = DRONE_COLORS[data.drone_type] || 0x58b4ff;
    const darkAccent = this._mixColors(color, 0x08111b, 0.7);

    const bodyMat = new THREE.MeshPhysicalMaterial({
      color,
      roughness: 0.34,
      metalness: 0.5,
      clearcoat: 0.42,
      clearcoatRoughness: 0.3,
      emissive: 0x07131e,
      emissiveIntensity: 0.18,
    });

    const body = new THREE.Mesh(this.coreGeo, bodyMat);
    body.castShadow = true;
    body.receiveShadow = true;
    group.add(body);

    const canopy = new THREE.Mesh(
      this.canopyGeo,
      new THREE.MeshPhysicalMaterial({
        color: this._mixColors(color, 0xffffff, 0.24),
        roughness: 0.18,
        metalness: 0.18,
        transmission: 0.18,
        transparent: true,
        opacity: 0.9,
        clearcoat: 0.7,
        clearcoatRoughness: 0.1,
      })
    );
    canopy.rotation.z = Math.PI / 2;
    canopy.position.y = 0.2;
    canopy.castShadow = true;
    group.add(canopy);

    const armMat = new THREE.MeshStandardMaterial({
      color: darkAccent,
      metalness: 0.75,
      roughness: 0.26,
    });

    const rotors = [];
    for (let i = 0; i < 4; i++) {
      const angle = i * Math.PI / 2 + Math.PI / 4;
      const arm = new THREE.Mesh(this.armGeo, armMat);
      arm.rotation.z = Math.PI / 2;
      arm.position.set(Math.cos(angle) * 0.55, 0.03, Math.sin(angle) * 0.55);
      group.add(arm);

      const rotorGroup = new THREE.Group();
      rotorGroup.position.set(Math.cos(angle) * 1.05, 0.18, Math.sin(angle) * 1.05);

      const hub = new THREE.Mesh(
        this.rotorHubGeo,
        new THREE.MeshStandardMaterial({
          color: 0x12181f,
          metalness: 0.82,
          roughness: 0.28,
        })
      );
      rotorGroup.add(hub);

      const blades = new THREE.Group();
      for (let j = 0; j < 2; j++) {
        const blade = new THREE.Mesh(
          this.rotorBladeGeo,
          new THREE.MeshStandardMaterial({
            color: 0xc7d6e3,
            metalness: 0.1,
            roughness: 0.38,
          })
        );
        blade.rotation.y = (j * Math.PI) / 2;
        blades.add(blade);
      }
      rotorGroup.add(blades);

      const disc = new THREE.Mesh(
        this.rotorDiscGeo,
        new THREE.MeshBasicMaterial({
          color: 0xa8d8ff,
          transparent: true,
          opacity: 0.22,
          depthWrite: false,
        })
      );
      disc.rotation.x = Math.PI / 2;
      rotorGroup.add(disc);

      group.add(rotorGroup);
      rotors.push({ blades, disc });
    }
    this.rotors.set(data.id, rotors);

    const landingOffsets = [
      [-0.48, -0.28, -0.32],
      [0.48, -0.28, -0.32],
      [-0.48, -0.28, 0.32],
      [0.48, -0.28, 0.32],
    ];
    for (const [x, y, z] of landingOffsets) {
      const leg = new THREE.Mesh(
        this.landingGeo,
        new THREE.MeshStandardMaterial({
          color: 0x9aa7b6,
          metalness: 0.45,
          roughness: 0.4,
        })
      );
      leg.position.set(x, y, z);
      leg.rotation.z = 0.18 * Math.sign(x || 1);
      group.add(leg);
    }

    const payload = new THREE.Mesh(
      this.payloadGeo,
      new THREE.MeshPhysicalMaterial({
        color: 0x7cd7ff,
        emissive: 0x2aa0ff,
        emissiveIntensity: 0.48,
        roughness: 0.28,
        metalness: 0.2,
      })
    );
    payload.position.set(0, -0.32, 0);
    payload.visible = false;
    group.add(payload);
    this.payloadIndicators.set(data.id, payload);

    const leftLight = this._createNavLight(0xff5252);
    leftLight.position.set(-0.9, 0.12, 0.1);
    group.add(leftLight);

    const rightLight = this._createNavLight(0x5bd1ff);
    rightLight.position.set(0.9, 0.12, -0.1);
    group.add(rightLight);

    this.navLights.set(data.id, { left: leftLight, right: rightLight });

    this.scene.add(group);
    return group;
  }

  _createNavLight(color) {
    return new THREE.Mesh(
      new THREE.SphereGeometry(0.07, 10, 10),
      new THREE.MeshStandardMaterial({
        color,
        emissive: color,
        emissiveIntensity: 0.85,
        roughness: 0.12,
      })
    );
  }

  _updateStateEffect(group, data) {
    const body = group.children[0];
    if (!body) return;

    if (data.state === 'charging') {
      body.material.emissive.setHex(0x124d2e);
      body.material.emissiveIntensity = 0.4;
    } else if (data.state === 'emergency' || data.battery < 0.15) {
      body.material.emissive.setHex(0x5a1111);
      body.material.emissiveIntensity = 0.46;
    } else if (data.state === 'delivering') {
      body.material.emissive.setHex(0x0b2840);
      body.material.emissiveIntensity = 0.26;
    } else {
      body.material.emissive.setHex(0x07131e);
      body.material.emissiveIntensity = 0.18;
    }
  }

  _mixColors(a, b, t = 0.5) {
    const c1 = new THREE.Color(a);
    const c2 = new THREE.Color(b);
    c1.lerp(c2, t);
    return c1;
  }

  _hash(input) {
    let value = 0;
    const text = String(input);
    for (let i = 0; i < text.length; i++) {
      value = (value * 31 + text.charCodeAt(i)) % 997;
    }
    return value / 997;
  }
}
