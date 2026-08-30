/**
 * UrbanFly 机载 RGB-D 传感器
 * ============================
 * 从 UrbanFly 当前摄影测量场景进行离屏渲染。RGB 与 DepthPerspective 使用
 * 同一台透视相机、同一时间戳和同一位姿；深度值以米为单位。
 */

import * as THREE from 'three';
import { DISPLAY_CITY_LAYER, SENSOR_CITY_LAYER } from './city_display_lod.js';

const DEFAULT_CONFIG = {
  front_center: {
    body_pose: {
      position: [1.05, 0.02, 0],
      roll_pitch_yaw_degrees: [0, -8, 0],
    },
    capture_settings: {
      width: 160,
      height: 90,
      fov_degrees: 90,
      near_clip_m: 0.3,
      far_clip_m: 120,
      frame_rate_hz: 10,
      motion_blur: 0,
    },
    image_types: ['Scene', 'DepthPerspective'],
    depth_unit: 'meter',
    gimbal_stabilization: { roll: false, pitch: false, yaw: false },
  },
};

export class DroneSensorSuite {
  constructor(sceneManager, droneManager) {
    this.sceneManager = sceneManager;
    this.droneManager = droneManager;
    this.config = structuredClone(DEFAULT_CONFIG);
    this.activeDroneId = null;
    this.elapsed = 0;
    this.sequence = 0;
    this.latestFrame = null;
    this.enabled = true;
    this.streamingEnabled = false;
    this.frameSubscribers = new Set();
    this.captureCount = 0;
    this.captureTimeMs = 0;
    this.bridgeSender = null;
    this.bridgeCaptureStartedSender = null;
    this.bridgeEnabled = false;
    this.bridgeEncoding = false;
    this.bridgeFrames = 0;
    this.bridgeDroppedFrames = 0;
    this.bridgeSkippedBusy = 0;
    this.lastBridgeCaptureSimTime = null;
    this.bridgeEncodeTimeMs = 0;
    this.bridgeWorker = new Worker(
      new URL('./sensor_packet_worker.js', import.meta.url),
      { type: 'module', name: 'urbanfly-rgbd-packetizer' },
    );
    this.bridgeWorker.onmessage = (event) => this._handleBridgeWorkerMessage(event.data);
    this.bridgeWorker.onerror = (event) => {
      this.bridgeEncoding = false;
      this.bridgeDroppedFrames += 1;
      console.error('[Sensors] RGB-D packet worker failed:', event.message);
    };
    this.perturbation = { camera_noise_std: 0, frame_drop_probability: 0 };

    this.rgbCanvas = document.getElementById('sensor-rgb');
    this.depthCanvas = document.getElementById('sensor-depth');
    this.statusElement = document.getElementById('sensor-status');
    this.droneSelect = document.getElementById('sensor-drone-select');
    this.rgbContext = this.rgbCanvas?.getContext('2d');
    this.depthContext = this.depthCanvas?.getContext('2d');

    this.sensorCamera = new THREE.PerspectiveCamera();
    this.sensorCamera.layers.enable(SENSOR_CITY_LAYER);
    this.thirdPersonCamera = new THREE.PerspectiveCamera(52, 16 / 9, 0.5, 4000);
    this.thirdPersonCamera.layers.enable(DISPLAY_CITY_LAYER);
    this.overviewCamera = new THREE.OrthographicCamera(-1000, 1000, 560, -560, 1, 3000);
    this.overviewCamera.layers.enable(DISPLAY_CITY_LAYER);
    this.overviewCamera.position.set(0, 1200, 0);
    this.overviewCamera.up.set(0, 0, -1);
    this.overviewCamera.lookAt(0, 0, 0);
    this.overviewCamera.updateMatrixWorld(true);
    this.thirdPersonCapture = { enabled: false, distance_m: 22, height_m: 9 };
    this.thirdPersonTarget = null;
    this.thirdPersonPixels = null;
    this.overviewTarget = null;
    this.overviewPixels = null;
    this.rgbTarget = null;
    this.depthTarget = null;
    this.rgbPixels = null;
    this.depthPackedPixels = null;
    this.depthMeters = null;
    this.depthMaterial = new THREE.MeshDepthMaterial({
      depthPacking: THREE.RGBADepthPacking,
      blending: THREE.NoBlending,
    });

    this._configureTargets();
    this._bindControls();
    window.urbanFlySensorSuite = this;
    window.urbanFlySensors = this._createPublicApi();
    document.documentElement.dataset.sensorApi = 'urbanflySensors';
  }

  applyConfig(config) {
    if (!config?.front_center) return;
    const before = JSON.stringify(this.config.front_center.capture_settings);
    this.config = structuredClone(config);
    const after = JSON.stringify(this.config.front_center.capture_settings);
    if (before !== after) this._configureTargets();
  }

  applyPerturbation(perturbation = {}) {
    this.perturbation = {
      camera_noise_std: Math.max(0, Math.min(0.1, Number(perturbation.camera_noise_std || 0))),
      frame_drop_probability: Math.max(0, Math.min(0.5, Number(perturbation.frame_drop_probability || 0))),
    };
  }

  update(droneStates, dt, simulationTime) {
    if (!this.enabled || !droneStates?.length) return;
    this._syncDroneOptions(droneStates);
    const drone = droneStates.find((item) => item.id === this.activeDroneId)
      || droneStates[0];
    if (!drone) return;
    this.activeDroneId = drone.id;

    // The policy RGB-D bridge is driven directly by sim_state below. Keeping
    // it out of requestAnimationFrame makes lockstep capture independent of
    // tab visibility and display refresh rate.
    if (this.bridgeEnabled) return;

    const settings = this.config.front_center.capture_settings;
    this.elapsed += dt;
    const interval = 1 / Math.max(1, settings.frame_rate_hz || 8);
    if (this.elapsed < interval) return;
    this.elapsed %= interval;
    const draw = (((this.captureCount + 1) * 1664525 + 1013904223) >>> 0) / 4294967296;
    if (draw < this.perturbation.frame_drop_probability) {
      this.captureCount += 1;
      this.bridgeDroppedFrames += 1;
      return;
    }

    const droneObject = this.droneManager.drones.get(drone.id);
    if (!droneObject) return;
    this.lastCaptureContext = { drone, droneObject, simulationTime };
    const previewVisible = document.getElementById('tab-sensors')
      ?.classList.contains('active');
    if (!previewVisible && !this.streamingEnabled && this.frameSubscribers.size === 0) {
      return;
    }
    this._capture(drone, droneObject, simulationTime, previewVisible);
  }

  captureBridgeState(droneStates, simulationTime) {
    if (!this.enabled || !this.bridgeEnabled || !droneStates?.length) return false;
    const simTime = Number(simulationTime);
    if (!Number.isFinite(simTime)) return false;
    if (
      this.lastBridgeCaptureSimTime !== null
      && Math.abs(simTime - this.lastBridgeCaptureSimTime) < 1e-9
    ) {
      return false;
    }
    if (this.bridgeEncoding) {
      this.bridgeSkippedBusy += 1;
      return false;
    }

    this._syncDroneOptions(droneStates);
    const drone = droneStates.find((item) => item.id === this.activeDroneId)
      || droneStates[0];
    if (!drone) return false;
    this.activeDroneId = drone.id;
    const droneObject = this.droneManager.drones.get(drone.id);
    if (!droneObject) return false;

    const draw = (((this.captureCount + 1) * 1664525 + 1013904223) >>> 0) / 4294967296;
    if (draw < this.perturbation.frame_drop_probability) {
      this.captureCount += 1;
      this.bridgeDroppedFrames += 1;
      return false;
    }

    this.lastBridgeCaptureSimTime = simTime;
    this.bridgeCaptureStartedSender?.({
      sim_time: simTime,
      vehicle_name: drone.id,
      source: 'sim_state_event',
    });
    this.lastCaptureContext = { drone, droneObject, simulationTime: simTime };
    const previewVisible = document.getElementById('tab-sensors')
      ?.classList.contains('active');
    this._capture(drone, droneObject, simTime, previewVisible);
    return true;
  }

  _configureTargets() {
    const settings = this.config.front_center.capture_settings;
    const width = settings.width;
    const height = settings.height;
    this.rgbTarget?.dispose();
    this.depthTarget?.dispose();
    this.thirdPersonTarget?.dispose();
    this.overviewTarget?.dispose();
    this.rgbTarget = new THREE.WebGLRenderTarget(width, height, {
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      format: THREE.RGBAFormat,
      type: THREE.UnsignedByteType,
      depthBuffer: true,
    });
    this.rgbTarget.texture.colorSpace = THREE.SRGBColorSpace;
    this.depthTarget = new THREE.WebGLRenderTarget(width, height, {
      minFilter: THREE.NearestFilter,
      magFilter: THREE.NearestFilter,
      format: THREE.RGBAFormat,
      type: THREE.UnsignedByteType,
      depthBuffer: true,
    });
    this.rgbPixels = new Uint8Array(width * height * 4);
    this.depthPackedPixels = new Uint8Array(width * height * 4);
    this.depthMeters = new Float32Array(width * height);
    this.thirdPersonTarget = new THREE.WebGLRenderTarget(640, 360, {
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      format: THREE.RGBAFormat,
      type: THREE.UnsignedByteType,
      depthBuffer: true,
    });
    this.thirdPersonTarget.texture.colorSpace = THREE.SRGBColorSpace;
    this.thirdPersonPixels = new Uint8Array(640 * 360 * 4);
    this.overviewTarget = new THREE.WebGLRenderTarget(640, 360, {
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      format: THREE.RGBAFormat,
      type: THREE.UnsignedByteType,
      depthBuffer: true,
    });
    this.overviewTarget.texture.colorSpace = THREE.SRGBColorSpace;
    this.overviewPixels = new Uint8Array(640 * 360 * 4);
    if (this.rgbCanvas) {
      this.rgbCanvas.width = width;
      this.rgbCanvas.height = height;
    }
    if (this.depthCanvas) {
      this.depthCanvas.width = width;
      this.depthCanvas.height = height;
    }
    this.sensorCamera.fov = settings.fov_degrees;
    this.sensorCamera.aspect = width / height;
    this.sensorCamera.near = settings.near_clip_m;
    this.sensorCamera.far = settings.far_clip_m;
    this.sensorCamera.updateProjectionMatrix();
    this.thirdPersonCamera.aspect = 640 / 360;
    this.thirdPersonCamera.updateProjectionMatrix();
  }

  _captureThirdPerson(drone, droneObject) {
    if (!this.thirdPersonCapture.enabled) return null;
    const renderer = this.sceneManager.renderer;
    const target = new THREE.Vector3();
    droneObject.getWorldPosition(target);
    const yaw = THREE.MathUtils.degToRad(Number(drone.yaw || 0));
    const forward = new THREE.Vector3(Math.cos(yaw), 0, Math.sin(yaw));
    this.thirdPersonCamera.position.copy(target)
      .addScaledVector(forward, -this.thirdPersonCapture.distance_m)
      .add(new THREE.Vector3(0, this.thirdPersonCapture.height_m, 0));
    this.thirdPersonCamera.up.set(0, 1, 0);
    this.thirdPersonCamera.lookAt(
      target.clone().addScaledVector(forward, 7).add(new THREE.Vector3(0, 2.5, 0)),
    );
    this.thirdPersonCamera.updateMatrixWorld(true);
    const previousTarget = renderer.getRenderTarget();
    const previousXr = renderer.xr.enabled;
    const previousShadowAutoUpdate = renderer.shadowMap.autoUpdate;
    try {
      renderer.xr.enabled = false;
      renderer.shadowMap.autoUpdate = false;
      renderer.setRenderTarget(this.thirdPersonTarget);
      renderer.clear();
      renderer.render(this.sceneManager.scene, this.thirdPersonCamera);
      renderer.readRenderTargetPixels(this.thirdPersonTarget, 0, 0, 640, 360, this.thirdPersonPixels);
    } finally {
      renderer.setRenderTarget(previousTarget);
      renderer.xr.enabled = previousXr;
      renderer.shadowMap.autoUpdate = previousShadowAutoUpdate;
    }
    return this._rgbWithoutAlpha(this.thirdPersonPixels);
  }

  _captureOverview() {
    if (!this.thirdPersonCapture.enabled) return null;
    const renderer = this.sceneManager.renderer;
    const previousTarget = renderer.getRenderTarget();
    const previousXr = renderer.xr.enabled;
    const previousShadowAutoUpdate = renderer.shadowMap.autoUpdate;
    try {
      renderer.xr.enabled = false;
      renderer.shadowMap.autoUpdate = false;
      renderer.setRenderTarget(this.overviewTarget);
      renderer.clear();
      renderer.render(this.sceneManager.scene, this.overviewCamera);
      renderer.readRenderTargetPixels(this.overviewTarget, 0, 0, 640, 360, this.overviewPixels);
    } finally {
      renderer.setRenderTarget(previousTarget);
      renderer.xr.enabled = previousXr;
      renderer.shadowMap.autoUpdate = previousShadowAutoUpdate;
    }
    return this._rgbWithoutAlpha(this.overviewPixels);
  }

  _capture(drone, droneObject, simulationTime, drawPreview = true) {
    const captureStarted = performance.now();
    const renderer = this.sceneManager.renderer;
    const scene = this.sceneManager.scene;
    const settings = this.config.front_center.capture_settings;
    const mount = this.config.front_center.body_pose;
    const mountPosition = new THREE.Vector3(...mount.position);
    droneObject.localToWorld(mountPosition);

    const pitch = THREE.MathUtils.degToRad(
      mount.roll_pitch_yaw_degrees?.[1] || 0,
    );
    const forward = new THREE.Vector3(Math.cos(pitch), Math.sin(pitch), 0)
      .applyQuaternion(droneObject.quaternion)
      .normalize();
    const up = new THREE.Vector3(0, 1, 0)
      .applyQuaternion(droneObject.quaternion)
      .normalize();
    this.sensorCamera.position.copy(mountPosition);
    this.sensorCamera.up.copy(up);
    this.sensorCamera.lookAt(mountPosition.clone().add(forward));
    this.sensorCamera.updateMatrixWorld(true);

    const previousTarget = renderer.getRenderTarget();
    const previousOverride = scene.overrideMaterial;
    const previousXr = renderer.xr.enabled;
    const previousShadowAutoUpdate = renderer.shadowMap.autoUpdate;
    const sensorExcludedObjects = [];
    scene.traverse((object) => {
      if (object.userData?.excludeFromSensors && object.visible) {
        sensorExcludedObjects.push(object);
        object.visible = false;
      }
    });
    renderer.xr.enabled = false;
    renderer.shadowMap.autoUpdate = false;

    try {
      renderer.setRenderTarget(this.rgbTarget);
      renderer.clear();
      renderer.render(scene, this.sensorCamera);
      renderer.readRenderTargetPixels(
        this.rgbTarget,
        0,
        0,
        settings.width,
        settings.height,
        this.rgbPixels,
      );

      scene.overrideMaterial = this.depthMaterial;
      renderer.setRenderTarget(this.depthTarget);
      renderer.clear();
      renderer.render(scene, this.sensorCamera);
      renderer.readRenderTargetPixels(
        this.depthTarget,
        0,
        0,
        settings.width,
        settings.height,
        this.depthPackedPixels,
      );
    } finally {
      scene.overrideMaterial = previousOverride;
      renderer.setRenderTarget(previousTarget);
      renderer.xr.enabled = previousXr;
      renderer.shadowMap.autoUpdate = previousShadowAutoUpdate;
      for (const object of sensorExcludedObjects) object.visible = true;
    }

    this._decodePerspectiveDepth(settings);
    this._applyRgbNoise();
    const thirdPersonRgb = this._captureThirdPerson(drone, droneObject);
    const overviewRgb = this._captureOverview();
    if (drawPreview) {
      this._drawRgb(settings);
      this._drawDepth(settings);
    }

    const timestamp = Math.round(
      (Number.isFinite(simulationTime) ? simulationTime : performance.now() / 1000)
      * 1e9,
    );
    const fy = settings.height / (
      2 * Math.tan(THREE.MathUtils.degToRad(settings.fov_degrees) / 2)
    );
    const fx = fy;
    const cameraQuaternion = this.sensorCamera.quaternion;
    this.latestFrame = {
      sequence: this.sequence++,
      timestamp,
      vehicle_name: drone.id,
      camera_name: 'front_center',
      width: settings.width,
      height: settings.height,
      camera_position: this.sensorCamera.position.toArray(),
      camera_orientation: [
        cameraQuaternion.w,
        cameraQuaternion.x,
        cameraQuaternion.y,
        cameraQuaternion.z,
      ],
      intrinsics: {
        fx,
        fy,
        cx: (settings.width - 1) / 2,
        cy: (settings.height - 1) / 2,
        fov_degrees: settings.fov_degrees,
        near_clip_m: settings.near_clip_m,
        far_clip_m: settings.far_clip_m,
      },
      image_data_uint8: this._rgbWithoutAlpha(),
      third_person_width: thirdPersonRgb ? 640 : 0,
      third_person_height: thirdPersonRgb ? 360 : 0,
      third_person_image_data_uint8: thirdPersonRgb,
      overview_width: overviewRgb ? 640 : 0,
      overview_height: overviewRgb ? 360 : 0,
      overview_image_data_uint8: overviewRgb,
      image_data_float: this.depthMeters.slice(),
      depth_type: 'DepthPerspective',
      depth_unit: 'meter',
      vehicle_pose: {
        position: [...drone.pos],
        orientation: drone.orientation
          ? [...drone.orientation]
          : [1, 0, 0, 0],
      },
      depth_stats: { ...this.depthStats },
      dynamics: {
        model: drone.dynamics_model || 'kinematic-demo',
        roll_degrees: drone.roll || 0,
        pitch_degrees: drone.pitch || 0,
        yaw_degrees: drone.yaw || 0,
        angular_velocity_rad_s: drone.angular_velocity
          ? [...drone.angular_velocity]
          : [0, 0, 0],
        motor_omega_rad_s: drone.motor_omega
          ? [...drone.motor_omega]
          : [0, 0, 0, 0],
        total_thrust_n: drone.total_thrust || 0,
        electrical_power_w: drone.power_w || 0,
      },
    };
    this.captureCount += 1;
    this.captureTimeMs = performance.now() - captureStarted;
    if (this.bridgeEnabled && this.bridgeSender) {
      this._publishBridgeFrame(this.latestFrame, drone);
    }
    for (const callback of this.frameSubscribers) {
      queueMicrotask(() => {
        try {
          callback(this.latestFrame);
        } catch (error) {
          console.error('[Sensors] Frame subscriber failed:', error);
        }
      });
    }
    if (this.statusElement) {
      this.statusElement.textContent = (
        `${drone.id} · ${settings.width}×${settings.height} · `
        + `${settings.frame_rate_hz} Hz · #${this.latestFrame.sequence} · `
        + `R ${Number(drone.roll || 0).toFixed(1)}° `
        + `P ${Number(drone.pitch || 0).toFixed(1)}°`
      );
      this.statusElement.dataset.depthMin = this.depthStats.minimum_m.toFixed(3);
      this.statusElement.dataset.depthMean = this.depthStats.mean_m.toFixed(3);
      this.statusElement.dataset.depthMax = this.depthStats.maximum_m.toFixed(3);
    }
  }

  _applyRgbNoise() {
    const sigma = this.perturbation.camera_noise_std * 255;
    if (sigma <= 0) return;
    let state = ((this.sequence + 1) * 2654435761) >>> 0;
    for (let index = 0; index < this.rgbPixels.length; index += 4) {
      state = (1664525 * state + 1013904223) >>> 0;
      const noise = ((state / 4294967296) * 2 - 1) * sigma * 1.732;
      this.rgbPixels[index] = Math.max(0, Math.min(255, this.rgbPixels[index] + noise));
      this.rgbPixels[index + 1] = Math.max(0, Math.min(255, this.rgbPixels[index + 1] + noise));
      this.rgbPixels[index + 2] = Math.max(0, Math.min(255, this.rgbPixels[index + 2] + noise));
    }
  }

  _decodePerspectiveDepth(settings) {
    const width = settings.width;
    const height = settings.height;
    const near = settings.near_clip_m;
    const far = settings.far_clip_m;
    const tanHalfFovY = Math.tan(
      THREE.MathUtils.degToRad(settings.fov_degrees) / 2,
    );
    const tanHalfFovX = tanHalfFovY * (width / height);
    const unpackScale = 255 / 256;
    let minimum = Number.POSITIVE_INFINITY;
    let maximum = 0;
    let sum = 0;

    for (let y = 0; y < height; y++) {
      const ny = ((y + 0.5) / height) * 2 - 1;
      for (let x = 0; x < width; x++) {
        const index = (y * width + x) * 4;
        const r = this.depthPackedPixels[index] / 255;
        const g = this.depthPackedPixels[index + 1] / 255;
        const b = this.depthPackedPixels[index + 2] / 255;
        const a = this.depthPackedPixels[index + 3] / 255;
        const depthBuffer = unpackScale * (
          r / 16777216 + g / 65536 + b / 256 + a
        );
        const viewZ = (near * far) / ((far - near) * depthBuffer - far);
        const nx = ((x + 0.5) / width) * 2 - 1;
        const rayScale = Math.sqrt(
          1
          + (nx * tanHalfFovX) ** 2
          + (ny * tanHalfFovY) ** 2
        );
        const distance = Math.min(
          far,
          Math.max(near, -viewZ * rayScale),
        );
        this.depthMeters[y * width + x] = distance;
        minimum = Math.min(minimum, distance);
        maximum = Math.max(maximum, distance);
        sum += distance;
      }
    }
    this.depthStats = {
      minimum_m: minimum,
      mean_m: sum / (width * height),
      maximum_m: maximum,
    };
  }

  _drawRgb(settings) {
    if (!this.rgbContext) return;
    const image = this.rgbContext.createImageData(settings.width, settings.height);
    this._flipRgbaRows(
      this.rgbPixels,
      image.data,
      settings.width,
      settings.height,
    );
    this.rgbContext.putImageData(image, 0, 0);
  }

  _drawDepth(settings) {
    if (!this.depthContext) return;
    const width = settings.width;
    const height = settings.height;
    const image = this.depthContext.createImageData(width, height);
    for (let y = 0; y < height; y++) {
      const sourceY = height - 1 - y;
      for (let x = 0; x < width; x++) {
        const source = sourceY * width + x;
        const target = (y * width + x) * 4;
        const depth = this.depthMeters[source];
        const normalized = Math.min(1, Math.log1p(depth) / Math.log1p(120));
        const color = this._turbo(1 - normalized);
        image.data[target] = color[0];
        image.data[target + 1] = color[1];
        image.data[target + 2] = color[2];
        image.data[target + 3] = 255;
      }
    }
    this.depthContext.putImageData(image, 0, 0);
  }

  _flipRgbaRows(source, target, width, height) {
    const stride = width * 4;
    for (let y = 0; y < height; y++) {
      const sourceOffset = (height - 1 - y) * stride;
      target.set(source.subarray(sourceOffset, sourceOffset + stride), y * stride);
    }
  }

  _rgbWithoutAlpha(pixels = this.rgbPixels) {
    const output = new Uint8Array((pixels.length / 4) * 3);
    for (let source = 0, target = 0; source < pixels.length; source += 4) {
      output[target++] = pixels[source];
      output[target++] = pixels[source + 1];
      output[target++] = pixels[source + 2];
    }
    return output;
  }

  _publishBridgeFrame(frame, drone) {
    if (this.bridgeEncoding) {
      this.bridgeDroppedFrames += 1;
      return;
    }
    this.bridgeEncoding = true;
    try {
      const { width, height } = frame;
      const clip = 120;
      const depthCompression = 'none';
      // Keep the Dataset v1 bridge latency bounded. Chromium's asynchronous
      // CompressionStream flush can take multiple simulator seconds on the
      // acceptance runtime even for a small frame. At 160x90 the lossless
      // uint16 depth payload is only 28.8 kB, so raw transport is both factual
      // and comfortably below the server's packet limit.
      const goal = drone.path_remaining?.[0] || drone.pos;
      const yaw = THREE.MathUtils.degToRad(Number(drone.yaw || 0));
      const dx = Number(goal[0] || 0) - Number(drone.pos[0] || 0);
      const dy = Number(goal[1] || 0) - Number(drone.pos[1] || 0);
      const dz = Number(goal[2] || 0) - Number(drone.pos[2] || 0);
      const header = {
        schema: 'urbanfly-sensor-packet-v2',
        sequence: frame.sequence,
        timestamp_ns: frame.timestamp,
        sim_time: frame.timestamp / 1e9,
        vehicle_name: frame.vehicle_name,
        width,
        height,
        rgb_codec: 'raw_rgb8',
        rgb_length: width * height * 3,
        depth_codec: 'u16',
        depth_compression: depthCompression,
        depth_length: width * height * 2,
        depth_scale_m: clip / 65535,
        intrinsics: frame.intrinsics,
        camera_position: frame.camera_position,
        camera_orientation: frame.camera_orientation,
        vehicle_pose: frame.vehicle_pose,
        linear_velocity_world_mps: drone.vel || [0, 0, 0],
        angular_velocity_world_rps: drone.angular_velocity || [0, 0, 0],
        linear_velocity_body_flu_mps: [
          Math.cos(yaw) * Number(drone.vel?.[0] || 0)
            + Math.sin(yaw) * Number(drone.vel?.[2] || 0),
          Math.sin(yaw) * Number(drone.vel?.[0] || 0)
            - Math.cos(yaw) * Number(drone.vel?.[2] || 0),
          Number(drone.vel?.[1] || 0),
        ],
        angular_velocity_body_flu_rps: [
          Number(drone.angular_velocity?.[0] || 0),
          -Number(drone.angular_velocity?.[2] || 0),
          Number(drone.angular_velocity?.[1] || 0),
        ],
        goal_world_m: goal,
        goal_body_flu_m: [
          Math.cos(yaw) * dx + Math.sin(yaw) * dz,
          Math.sin(yaw) * dx - Math.cos(yaw) * dz,
          dy,
        ],
        dynamics: frame.dynamics,
        yaw_degrees: Number(drone.yaw || 0),
        world_model: drone.world_model || null,
        drone_state: drone.state || 'unknown',
      };
      const rgbRgba = this.rgbPixels.slice();
      const depthMeters = this.depthMeters.slice();
      this.bridgeWorker.postMessage(
        {
          header,
          width,
          height,
          depthClipM: clip,
          rgbRgbaBuffer: rgbRgba.buffer,
          depthMetersBuffer: depthMeters.buffer,
        },
        [rgbRgba.buffer, depthMeters.buffer],
      );
    } catch (error) {
      this.bridgeDroppedFrames += 1;
      this.bridgeEncoding = false;
      console.error('[Sensors] Binary bridge frame failed:', error);
    }
  }

  _handleBridgeWorkerMessage(message) {
    this.bridgeEncoding = false;
    if (message?.type !== 'packet' || !(message.packet instanceof ArrayBuffer)) {
      this.bridgeDroppedFrames += 1;
      console.error('[Sensors] RGB-D packet worker error:', message?.error || 'invalid result');
      return;
    }
    this.bridgeEncodeTimeMs = Number(message.encode_ms || 0);
    if (this.bridgeSender?.(message.packet) === false) {
      this.bridgeDroppedFrames += 1;
      return;
    }
    this.bridgeFrames += 1;
  }

  _turbo(value) {
    const x = Math.max(0, Math.min(1, value));
    const r = Math.max(0, Math.min(255, 255 * (1.5 - Math.abs(4 * x - 3))));
    const g = Math.max(0, Math.min(255, 255 * (1.5 - Math.abs(4 * x - 2))));
    const b = Math.max(0, Math.min(255, 255 * (1.5 - Math.abs(4 * x - 1))));
    return [r, g, b];
  }

  _syncDroneOptions(drones) {
    if (!this.droneSelect) return;
    const signature = drones.map((drone) => drone.id).join('|');
    if (this.droneSelect.dataset.signature === signature) return;
    this.droneSelect.dataset.signature = signature;
    this.droneSelect.replaceChildren();
    for (const drone of drones) {
      const option = document.createElement('option');
      option.value = drone.id;
      option.textContent = `${drone.id} · ${drone.drone_type}`;
      option.selected = drone.id === this.activeDroneId;
      this.droneSelect.append(option);
    }
    if (!this.activeDroneId && drones[0]) this.activeDroneId = drones[0].id;
    this.droneSelect.value = this.activeDroneId;
  }

  _bindControls() {
    this.droneSelect?.addEventListener('change', () => {
      this.activeDroneId = this.droneSelect.value;
      this.elapsed = 1.0;
    });
  }

  _createPublicApi() {
    return Object.freeze({
      getConfig: () => structuredClone(this.config),
      setActiveDrone: (droneId) => {
        this.activeDroneId = droneId;
        if (this.droneSelect) this.droneSelect.value = droneId;
      },
      getCameraInfo: () => {
        if (!this.latestFrame) return null;
        return {
          camera_name: this.latestFrame.camera_name,
          vehicle_name: this.latestFrame.vehicle_name,
          pose: {
            position: [...this.latestFrame.camera_position],
            orientation: [...this.latestFrame.camera_orientation],
          },
          intrinsics: { ...this.latestFrame.intrinsics },
        };
      },
      simGetImages: (requests = []) => {
        if (this.lastCaptureContext) {
          const { drone, droneObject, simulationTime } = this.lastCaptureContext;
          this._capture(drone, droneObject, simulationTime, false);
        }
        return requests.map((request) => {
        if (!this.latestFrame) return null;
        const imageType = request.image_type || request.imageType || 'Scene';
        const common = {
          camera_name: 'front_center',
          vehicle_name: this.latestFrame.vehicle_name,
          timestamp: this.latestFrame.timestamp,
          width: this.latestFrame.width,
          height: this.latestFrame.height,
          camera_position: [...this.latestFrame.camera_position],
          camera_orientation: [...this.latestFrame.camera_orientation],
        };
        if (imageType === 'DepthPerspective') {
          return {
            ...common,
            image_type: imageType,
            pixels_as_float: true,
            image_data_float: this.latestFrame.image_data_float.slice(),
            depth_unit: 'meter',
          };
        }
        return {
          ...common,
          image_type: 'Scene',
          pixels_as_float: false,
          compress: false,
          image_data_uint8: this.latestFrame.image_data_uint8.slice(),
        };
        });
      },
      latest: () => this.latestFrame,
      startStreaming: () => {
        this.streamingEnabled = true;
        this.elapsed = 1.0;
      },
      stopStreaming: () => {
        this.streamingEnabled = false;
      },
      subscribe: (callback) => {
        if (typeof callback !== 'function') {
          throw new TypeError('sensor subscriber must be a function');
        }
        this.frameSubscribers.add(callback);
        this.elapsed = 1.0;
        return () => this.frameSubscribers.delete(callback);
      },
      statistics: () => ({
        captures: this.captureCount,
        latest_capture_ms: this.captureTimeMs,
        streaming: this.streamingEnabled,
        subscribers: this.frameSubscribers.size,
        ui_independent: true,
        bridge_enabled: this.bridgeEnabled,
        bridge_frames: this.bridgeFrames,
        bridge_dropped_frames: this.bridgeDroppedFrames,
        bridge_skipped_busy: this.bridgeSkippedBusy,
        bridge_capture_source: 'sim_state_event',
        last_bridge_sim_time: this.lastBridgeCaptureSimTime,
        latest_bridge_encode_ms: this.bridgeEncodeTimeMs,
        bridge_worker: true,
      }),
      setBridgeSender: (sender) => {
        if (sender !== null && typeof sender !== 'function') {
          throw new TypeError('bridge sender must be a function or null');
        }
        this.bridgeSender = sender;
      },
      setBridgeCaptureStartedSender: (sender) => {
        if (sender !== null && typeof sender !== 'function') {
          throw new TypeError('bridge capture sender must be a function or null');
        }
        this.bridgeCaptureStartedSender = sender;
      },
      setBridgeEnabled: (enabled) => {
        this.bridgeEnabled = Boolean(enabled);
        // Manual preview streaming and the policy bridge have separate owners.
        // Closing a collector must not silently leave expensive preview capture on.
        this.elapsed = 1.0;
        this.lastBridgeCaptureSimTime = null;
      },
      setThirdPersonCapture: (settings = {}) => {
        this.thirdPersonCapture = {
          enabled: Boolean(settings.enabled),
          distance_m: Math.max(10, Math.min(60, Number(settings.distance_m || 22))),
          height_m: Math.max(4, Math.min(30, Number(settings.height_m || 9))),
        };
      },
    });
  }
}
