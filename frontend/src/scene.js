/**
 * Three.js 场景管理
 * =================
 * 负责相机、灯光、地表和整体环境氛围。
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { PresentationBudget } from './presentation_budget.js';
import { presentationHidden } from './host_lifecycle.js';
import { DISPLAY_CITY_LAYER } from './city_display_lod.js';
import { RenderBenchmark } from './render_benchmark.js';

export class SceneManager {
  constructor(container, canvas) {
    this.container = container;
    this.canvas = canvas;
    this.scene = null;
    this.camera = null;
    this.renderer = null;
    this.controls = null;
    this.clock = new THREE.Clock();
    this.composer = null;
    this.sunLight = null;
    this.digitalTwinViewer = null;
    this.digitalTwinDebugRenderer = null;
    this.presentation = new PresentationBudget();
    this.interactingUntil = 0;
    this.quality = 'smooth';
    this.displayDirty = true;
    this.displayIdle = false;
    this.cityDisplayLod = null;
    this.benchmark = new RenderBenchmark();
    this.lastDrawCalls = 0;
    this.lastTriangles = 0;
  }

  init() {
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x9fb6c9);
    this.scene.fog = null;

    const aspect = this.container.clientWidth / this.container.clientHeight;
    this.camera = new THREE.PerspectiveCamera(52, aspect, 0.5, 4000);
    this.camera.layers.enable(DISPLAY_CITY_LAYER);
    this.camera.position.set(460, 280, 520);
    this.camera.lookAt(0, 90, 0);

    this.renderer = new THREE.WebGLRenderer({
      canvas: this.canvas,
      antialias: true,
      alpha: false,
      powerPreference: 'high-performance',
    });
    this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
    // Keep the same scene and effects while avoiding a 4x fragment workload
    // on high-DPI displays.
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.08;

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(0, 90, 0);
    this.controls.maxPolarAngle = Math.PI / 2.08;
    this.controls.minPolarAngle = 0.28;
    this.controls.minDistance = 35;
    this.controls.maxDistance = 1000;
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.055;
    this.controls.autoRotate = false;
    this.controls.addEventListener('change', () => {
      this.interactingUntil = performance.now() + 250;
      this.invalidate();
    });
    this.controls.addEventListener('start', () => this.invalidate());

    this._setupLights();
    this._setupAtmosphere();
    // Allocate the postprocessing targets only when the operator requests them.
    document.getElementById('display-quality')?.addEventListener('change', (event) => {
      this.setQuality(event.target.value);
    });
    document.getElementById('display-lod')?.addEventListener('change', (event) => {
      if (this.cityDisplayLod) this.cityDisplayLod.enabled = event.target.checked;
      this.invalidate();
    });

    window.addEventListener('resize', () => this._onResize());
  }

  applyAppearance(perturbation = {}) {
    const exposure = Number(perturbation.exposure_ev || 0);
    const fog = Number(perturbation.fog_density || 0);
    const temperature = Number(perturbation.color_temperature_k || 6500);
    this.renderer.toneMappingExposure = 1.08 * (2 ** exposure);
    this.scene.fog = fog > 0
      ? new THREE.FogExp2(new THREE.Color(0xb9c7ce), fog * 0.008)
      : null;
    if (this.sunLight) {
      const warm = THREE.MathUtils.clamp((6500 - temperature) / 4000, -1, 1);
      this.sunLight.color.setRGB(
        THREE.MathUtils.clamp(1 + 0.12 * warm, 0.75, 1),
        THREE.MathUtils.clamp(0.95 - 0.03 * Math.abs(warm), 0.75, 1),
        THREE.MathUtils.clamp(0.88 - 0.18 * warm, 0.65, 1),
      );
    }
  }

  _setupLights() {
    const ambient = new THREE.AmbientLight(0xffffff, 0.42);
    this.scene.add(ambient);

    const hemisphere = new THREE.HemisphereLight(0xd9eeff, 0x52644f, 0.72);
    this.scene.add(hemisphere);

    const fillA = new THREE.DirectionalLight(0xd7edff, 0.38);
    fillA.position.set(-260, 140, -160);
    this.scene.add(fillA);

    const fillB = new THREE.DirectionalLight(0xaad8ff, 0.16);
    fillB.position.set(90, 68, -260);
    this.scene.add(fillB);

    const rim = new THREE.DirectionalLight(0xffd4a5, 0.34);
    rim.position.set(-180, 110, 280);
    this.scene.add(rim);

    const sun = new THREE.DirectionalLight(0xfff0d1, 2.4);
    sun.position.set(280, 210, 80);
    sun.castShadow = true;
    sun.shadow.mapSize.width = 4096;
    sun.shadow.mapSize.height = 4096;
    sun.shadow.camera.near = 10;
    sun.shadow.camera.far = 1800;
    sun.shadow.camera.left = -550;
    sun.shadow.camera.right = 550;
    sun.shadow.camera.top = 550;
    sun.shadow.camera.bottom = -550;
    sun.shadow.bias = -0.00015;
    this.scene.add(sun);
    this.sunLight = sun;
  }

  _setupAtmosphere() {
    const skyGeo = new THREE.SphereGeometry(2400, 48, 32);
    const skyMat = new THREE.ShaderMaterial({
      side: THREE.BackSide,
      depthWrite: false,
      uniforms: {
        topColor: { value: new THREE.Color(0x6093be) },
        horizonColor: { value: new THREE.Color(0xbdd6e7) },
        bottomColor: { value: new THREE.Color(0xdfe7e6) },
      },
      vertexShader: `
        varying vec3 vWorldPosition;
        void main() {
          vec4 worldPosition = modelMatrix * vec4(position, 1.0);
          vWorldPosition = worldPosition.xyz;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        uniform vec3 topColor;
        uniform vec3 horizonColor;
        uniform vec3 bottomColor;
        varying vec3 vWorldPosition;
        void main() {
          float h = normalize(vWorldPosition).y * 0.5 + 0.5;
          vec3 color = mix(bottomColor, horizonColor, smoothstep(0.0, 0.45, h));
          color = mix(color, topColor, smoothstep(0.45, 1.0, h));
          gl_FragColor = vec4(color, 1.0);
        }
      `,
    });
    const sky = new THREE.Mesh(skyGeo, skyMat);
    sky.name = 'AtmosphereSky';
    this.scene.add(sky);

    const haze = new THREE.Mesh(
      new THREE.CylinderGeometry(760, 1080, 180, 64, 1, true),
      new THREE.MeshBasicMaterial({
        color: 0xd8e6ec,
        transparent: true,
        opacity: 0.08,
        side: THREE.DoubleSide,
        depthWrite: false,
      })
    );
    haze.position.y = 72;
    this.scene.add(haze);

    const sunGlow = new THREE.Sprite(
      new THREE.SpriteMaterial({
        color: 0xffcb82,
        transparent: true,
        opacity: 0.24,
        depthWrite: false,
      })
    );
    sunGlow.scale.set(180, 180, 1);
    sunGlow.position.copy(this.sunLight.position.clone().setLength(1000));
    this.scene.add(sunGlow);
  }

  _setupGround() {
    const baseGround = new THREE.Mesh(
      new THREE.PlaneGeometry(3600, 3600),
      new THREE.MeshStandardMaterial({
        color: 0x071019,
        roughness: 0.96,
        metalness: 0.02,
      })
    );
    baseGround.rotation.x = -Math.PI / 2;
    baseGround.position.y = -0.6;
    baseGround.receiveShadow = true;
    this.scene.add(baseGround);

    const cityPad = new THREE.Mesh(
      new THREE.CircleGeometry(980, 80),
      new THREE.MeshStandardMaterial({
        map: this._createGroundTexture(),
        color: 0x264057,
        roughness: 0.98,
        metalness: 0.03,
      })
    );
    cityPad.rotation.x = -Math.PI / 2;
    cityPad.position.y = -0.18;
    cityPad.receiveShadow = true;
    this.scene.add(cityPad);

    const scanRing = new THREE.Mesh(
      new THREE.RingGeometry(700, 880, 96),
      new THREE.MeshBasicMaterial({
        color: 0x5fd9ff,
        transparent: true,
        opacity: 0.09,
        side: THREE.DoubleSide,
        depthWrite: false,
      })
    );
    scanRing.rotation.x = -Math.PI / 2;
    scanRing.position.y = -0.05;
    this.scene.add(scanRing);

    const outerGrid = new THREE.GridHelper(2200, 40, 0x16324e, 0x0d1d2f);
    outerGrid.position.y = -0.14;
    outerGrid.material.transparent = true;
    outerGrid.material.opacity = 0.18;
    this.scene.add(outerGrid);
  }

  _setupPostProcessing() {
    const size = new THREE.Vector2(
      this.container.clientWidth,
      this.container.clientHeight
    );

    this.composer = new EffectComposer(this.renderer);
    this.composer.addPass(new RenderPass(this.scene, this.camera));

    const bloom = new UnrealBloomPass(size, 0.48, 0.72, 0.84);
    bloom.threshold = 0.86;
    bloom.strength = 0.18;
    bloom.radius = 0.48;
    this.composer.addPass(bloom);
  }

  setDigitalTwinViewer(viewer, debugRenderer = null) {
    this.digitalTwinViewer = viewer;
    this.digitalTwinDebugRenderer = debugRenderer;
  }

  enableProceduralGround() {
    if (this.scene.getObjectByName('ProceduralGround')) return;
    const before = this.scene.children.length;
    this._setupGround();
    for (let i = before; i < this.scene.children.length; i++) {
      this.scene.children[i].name ||= 'ProceduralGround';
    }
  }

  _createGroundTexture() {
    const size = 1024;
    const canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d');

    const gradient = ctx.createRadialGradient(size / 2, size / 2, 80, size / 2, size / 2, size / 2);
    gradient.addColorStop(0, '#274056');
    gradient.addColorStop(0.55, '#182838');
    gradient.addColorStop(1, '#0b121b');
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, size, size);

    ctx.strokeStyle = 'rgba(88, 168, 220, 0.12)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= size; i += 32) {
      ctx.beginPath();
      ctx.moveTo(i, 0);
      ctx.lineTo(i, size);
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(0, i);
      ctx.lineTo(size, i);
      ctx.stroke();
    }

    ctx.strokeStyle = 'rgba(154, 220, 255, 0.08)';
    ctx.lineWidth = 2;
    for (let i = 0; i <= size; i += 128) {
      ctx.beginPath();
      ctx.moveTo(i, 0);
      ctx.lineTo(i, size);
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(0, i);
      ctx.lineTo(size, i);
      ctx.stroke();
    }

    for (let i = 0; i < 2400; i++) {
      const x = Math.random() * size;
      const y = Math.random() * size;
      const alpha = Math.random() * 0.05;
      ctx.fillStyle = `rgba(255,255,255,${alpha})`;
      ctx.fillRect(x, y, 1, 1);
    }

    const texture = new THREE.CanvasTexture(canvas);
    texture.wrapS = THREE.RepeatWrapping;
    texture.wrapT = THREE.RepeatWrapping;
    texture.anisotropy = 8;
    texture.colorSpace = THREE.SRGBColorSpace;
    return texture;
  }

  shouldRender(now, { busy, idle }) {
    if (this.benchmark.active) {
      if (busy || presentationHidden()) this.cancelBenchmark();
      else idle = false;
    }
    this.displayIdle = idle && !this.displayDirty && now >= this.interactingUntil;
    // A stopped, unchanged city has nothing to redraw. Orbit events, assets,
    // quality changes, resize and real sim_state all explicitly invalidate it.
    if (this.displayIdle && !presentationHidden()) return false;
    return this.presentation.shouldRender(now, {
      hidden: presentationHidden(), busy, idle, interacting: now < this.interactingUntil,
    });
  }

  invalidate() {
    this.displayDirty = true;
    this.displayIdle = false;
  }

  setQuality(quality) {
    this.quality = quality === 'detail' ? 'detail' : 'smooth';
    const ratio = Math.min(window.devicePixelRatio, this.quality === 'detail' ? 1.5 : 1);
    this.renderer.setPixelRatio(ratio);
    if (this.quality === 'detail' && !this.composer) this._setupPostProcessing();
    this.composer?.setPixelRatio(ratio);
    this.invalidate();
  }

  statistics() {
    return {
      ...this.presentation.statistics(performance.now()),
      pixel_ratio: this.renderer.getPixelRatio(),
      geometries: this.renderer.info.memory.geometries,
      textures: this.renderer.info.memory.textures,
      draw_calls: this.lastDrawCalls,
      display_triangles: this.lastTriangles,
      idle: this.displayIdle,
      ...this.cityDisplayLod?.statistics(),
    };
  }

  render({ busy = false } = {}) {
    this.displayDirty = false;
    const started = performance.now();
    this.controls.update();
    this.cityDisplayLod?.update(this.camera.position, { fullDetail: this.quality === 'detail' });
    const previousAutoReset = this.renderer.info.autoReset;
    this.renderer.info.autoReset = false;
    this.renderer.info.reset();
    try {
      if (this.digitalTwinViewer) {
        this.digitalTwinViewer.update();
        this.digitalTwinViewer.render();
        if (this.digitalTwinDebugRenderer) this.digitalTwinDebugRenderer();
      } else if (this.composer && this.quality === 'detail' && !busy) {
        this.composer.render();
      } else {
        // Collection omits display bloom. Sensor targets/lighting and the
        // authoritative high-detail city layer remain unchanged.
        this.renderer.render(this.scene, this.camera);
      }
      this.lastDrawCalls = this.renderer.info.render.calls;
      this.lastTriangles = this.renderer.info.render.triangles;
    } finally { this.renderer.info.autoReset = previousAutoReset; }
    const elapsed = performance.now() - started;
    this.presentation.record(started, elapsed);
    const report = this.benchmark.record(started, elapsed, this.lastDrawCalls, this.lastTriangles);
    if (report) {
      this.onBenchmarkComplete?.({ ...report, quality: this.quality,
        pixel_ratio: this.renderer.getPixelRatio(),
        viewport: [this.canvas.width, this.canvas.height],
        ...this.cityDisplayLod?.statistics() });
    }
  }

  cancelBenchmark() {
    if (!this.benchmark.active) return;
    this.benchmark.cancel();
    this.onBenchmarkComplete?.({ schema: 'urbanfly-display-benchmark-v1', cancelled: true });
  }

  _onResize() {
    this.invalidate();
    const w = this.container.clientWidth;
    const h = this.container.clientHeight;
    if (w <= 0 || h <= 0) return;

    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
    if (this.composer) {
      this.composer.setSize(w, h);
    }
  }
}
