/**
 * 相机管理器
 * ==========
 * 支持三种模式：轨道、跟随无人机、正交俯视。
 */

import * as THREE from 'three';

export class CameraManager {
  constructor(camera, controls) {
    this.camera = camera;
    this.controls = controls;
    this.mode = 'orbit';
    this.followTarget = null;
    this.followDistance = 45;
    this.followHeight = 18;
    this.followLookAhead = 7;
    this.transitionRate = 2.8;
    this.topDownHeight = 1150;
    this._targetPosition = new THREE.Vector3();
    this._forward = new THREE.Vector3();
    this._offset = new THREE.Vector3();
    this._desiredCameraPosition = new THREE.Vector3();
    this._lookTarget = new THREE.Vector3();
  }

  setMode(mode, targetId = null) {
    this.mode = mode;
    this.followTarget = targetId;

    if (mode === 'topdown') {
      this.camera.position.set(0, this.topDownHeight, 10);
      this.camera.lookAt(0, 0, 0);
      this.controls.enabled = false;
    } else if (mode === 'orbit') {
      this.controls.enabled = true;
      this.controls.target.set(0, 25, 0);
    } else if (mode === 'follow') {
      this.controls.enabled = false;
    }
  }

  update(drones, dt) {
    if (this.mode !== 'follow' || !this.followTarget) {
      return;
    }

    const target = drones.find(d => d.id === this.followTarget);
    if (!target) {
      this.mode = 'orbit';
      this.controls.enabled = true;
      return;
    }

    const targetPos = this._targetPosition.set(
      target.pos[0], target.pos[1], target.pos[2],
    );

    const yawRad = THREE.MathUtils.degToRad(target.yaw || 0);
    const forward = this._forward.set(
      Math.cos(yawRad),
      0,
      Math.sin(yawRad),
    );
    const offset = this._offset
      .copy(forward)
      .multiplyScalar(-this.followDistance)
      .setY(this.followHeight);

    const desiredCamPos = this._desiredCameraPosition
      .copy(targetPos)
      .add(offset);
    const transition = 1 - Math.exp(-this.transitionRate * Math.max(dt, 0.001));
    this.camera.position.lerp(desiredCamPos, transition);

    const lookTarget = this._lookTarget
      .copy(targetPos)
      .addScaledVector(forward, this.followLookAhead);
    lookTarget.y += 2.5;
    this.camera.lookAt(lookTarget);
  }
}
