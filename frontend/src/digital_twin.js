import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { MeshoptDecoder } from 'three/addons/libs/meshopt_decoder.module.js';
import { CityDisplayLod } from './city_display_lod.js';
import {
  acceleratedRaycast,
  computeBoundsTree,
  disposeBoundsTree,
} from 'three-mesh-bvh';

THREE.BufferGeometry.prototype.computeBoundsTree = computeBoundsTree;
THREE.BufferGeometry.prototype.disposeBoundsTree = disposeBoundsTree;
THREE.Mesh.prototype.raycast = acceleratedRaycast;

const DEFAULT_SCENE = 'CityCentral1km';
// The immutable source directory retains its licensed dataset identifier.
// Product-facing scene names remain generic "城市" aliases.
const LICENSED_ASSET_SCENE = 'HelsinkiCentral1km';
const OPERATION_SIZE_METERS = 1000;
const DEFAULT_ESDF_ALTITUDE = 30;

function fetchJson(url) {
  return fetch(url).then((response) => {
    if (!response.ok) throw new Error(`${url} returned ${response.status}`);
    return response.json();
  });
}

function yieldToMainThread() {
  return new Promise((resolve) => {
    if ('requestIdleCallback' in window) {
      window.requestIdleCallback(() => resolve(), { timeout: 48 });
      return;
    }
    window.setTimeout(resolve, 0);
  });
}

export class DigitalTwinRenderer {
  constructor(sceneManager, onStatus = () => {}) {
    this.sceneManager = sceneManager;
    this.scene = sceneManager.scene;
    this.onStatus = onStatus;
    this.sceneName = DEFAULT_SCENE;
    this.manifest = null;
    this.colliders = [];
    this.zoningData = null;
    this.esdfAltitude = DEFAULT_ESDF_ALTITUDE;
    this.assetRoot = null;
    this.loader = null;
    this.collisionDiagnosticsLoaded = false;
    this.esdfSliceLoaded = false;
    this.displayLod = new CityDisplayLod();
    this.sceneManager.cityDisplayLod = this.displayLod;

    this.overviewGroup = new THREE.Group();
    this.overviewGroup.name = 'CityOverviewMesh';
    this.scene.add(this.overviewGroup);

    this.meshGroup = new THREE.Group();
    this.meshGroup.name = 'CityPhotogrammetryMesh';
    this.scene.add(this.meshGroup);

    this.operationalGroup = new THREE.Group();
    this.operationalGroup.name = 'OperationalVolume';
    this.operationalGroup.userData.excludeFromSensors = true;
    this.scene.add(this.operationalGroup);

    this.collisionDebugGroup = new THREE.Group();
    this.collisionDebugGroup.name = 'CityTriangleCollisionDebug';
    this.collisionDebugGroup.userData.excludeFromSensors = true;
    this.collisionDebugGroup.visible = false;
    this.scene.add(this.collisionDebugGroup);

    this.esdfDebugGroup = new THREE.Group();
    this.esdfDebugGroup.name = 'CityEsdfDebug';
    this.esdfDebugGroup.userData.excludeFromSensors = true;
    this.esdfDebugGroup.visible = false;
    this.scene.add(this.esdfDebugGroup);

    this.zoningGroup = new THREE.Group();
    this.zoningGroup.name = 'CityFunctionalZoning';
    this.zoningGroup.userData.excludeFromSensors = true;
    this.zoningGroup.visible = false;
    this.scene.add(this.zoningGroup);

    this._bindDiagnosticsControls();
  }

  async load(sceneName = DEFAULT_SCENE) {
    this.sceneName = sceneName;
    const root = `/data/helsinki_mesh/${LICENSED_ASSET_SCENE}`;
    this.assetRoot = root;

    this.onStatus({
      phase: 'loading',
      title: '正在载入城市实景网格',
      detail: '读取场景清单与压缩瓦片',
    });

    this.manifest = await fetchJson(`${root}/manifest.json`);
    this._configureCamera(this.manifest.camera);
    this._createOperationalVolume(this.manifest.operation_size_m);
    this._populateEsdfOptions();

    const loader = this._createGltfLoader();
    this._clearGroup(this.overviewGroup, true);
    this._clearGroup(this.meshGroup);
    const overviewTiles = this.manifest.visual.overview?.tiles ?? [];
    for (let index = 0; index < overviewTiles.length; index += 1) {
      await yieldToMainThread();
      const tile = overviewTiles[index];
      this.onStatus({
        phase: 'loading',
        title: `正在载入 1 km 概览 ${index + 1}/${overviewTiles.length}`,
        detail: `${tile.name} · L${tile.source_lod}`,
      });
      const gltf = await loader.loadAsync(`${root}/overview/${tile.uri}`);
      gltf.scene.name = `CityOverview_${tile.name}`;
      this._prepareVisualScene(gltf.scene);
      this.displayLod.addOverview(tile.name, gltf.scene);
      await this._warmVisualScene(gltf.scene);
      this.overviewGroup.add(gltf.scene);
      this.sceneManager.invalidate();
    }

    const zoning = await this._loadZoning(root);
    const visualTiles = [...this.manifest.visual.tiles].sort((a, b) => {
      const centerA = [
        (a.bounds.minimum[0] + a.bounds.maximum[0]) * 0.5,
        (a.bounds.minimum[2] + a.bounds.maximum[2]) * 0.5,
      ];
      const centerB = [
        (b.bounds.minimum[0] + b.bounds.maximum[0]) * 0.5,
        (b.bounds.minimum[2] + b.bounds.maximum[2]) * 0.5,
      ];
      return Math.hypot(...centerA) - Math.hypot(...centerB);
    });
    for (let index = 0; index < visualTiles.length; index += 1) {
      await yieldToMainThread();
      const tile = visualTiles[index];
      this.onStatus({
        phase: 'loading',
        title: `正在载入实景网格 ${index + 1}/${visualTiles.length}`,
        detail: `${tile.name} · ${(tile.bytes / 1048576).toFixed(1)} MiB`,
      });
      const gltf = await loader.loadAsync(`${root}/visual/${tile.uri}`);
      gltf.scene.name = `CityVisual_${tile.name}`;
      this._prepareVisualScene(gltf.scene);
      await this._warmVisualScene(gltf.scene);
      this.displayLod.addHigh(tile.name, gltf.scene, tile.bounds);
      this.meshGroup.add(gltf.scene);
      this.sceneManager.invalidate();
      // Retain the existing L18 assets for distant display only. The L21
      // meshes always remain on the sensor layer, irrespective of camera LOD.
      this.displayLod.update(this.sceneManager.camera.position);
      await yieldToMainThread();
    }

    this.loader = loader;
    const collision = this.manifest.collision;
    this._updateCollisionStats(collision);
    this.onStatus({
      phase: 'ready',
      title: '城市实景网格已就绪',
      detail: `${this.manifest.visual.triangles.toLocaleString()} 三角面 · Meshopt`,
    });

    return {
      sceneName,
      operationSize: this.manifest.operation_size_m,
      source: '城市摄影测量网格',
      license: this.manifest.source.license,
      visual: this.manifest.visual,
      collision,
      zoning,
    };
  }

  update() {}

  render() {}

  intersectsSphere(center, radius) {
    const localCenter = new THREE.Vector3();
    const inverse = new THREE.Matrix4();
    const closest = new THREE.Vector3();
    for (const mesh of this.colliders) {
      inverse.copy(mesh.matrixWorld).invert();
      localCenter.copy(center).applyMatrix4(inverse);
      const localRadius = radius / mesh.matrixWorld.getMaxScaleOnAxis();
      const sphere = new THREE.Sphere(localCenter, localRadius);
      let hit = false;
      mesh.geometry.boundsTree.shapecast({
        intersectsBounds: (box) => box.intersectsSphere(sphere),
        intersectsTriangle: (triangle) => {
          triangle.closestPointToPoint(localCenter, closest);
          hit = closest.distanceToSquared(localCenter) <= localRadius ** 2;
          return hit;
        },
      });
      if (hit) return true;
    }
    return false;
  }

  raycastCollision(origin, direction, far = Infinity) {
    const raycaster = new THREE.Raycaster(origin, direction, 0, far);
    raycaster.firstHitOnly = true;
    const intersections = raycaster.intersectObjects(this.colliders, false);
    return intersections[0] ?? null;
  }

  _createGltfLoader() {
    const loader = new GLTFLoader();
    loader.setMeshoptDecoder(MeshoptDecoder);
    return loader;
  }

  _prepareVisualScene(root) {
    const anisotropy = Math.min(
      8,
      this.sceneManager.renderer.capabilities.getMaxAnisotropy(),
    );
    root.traverse((object) => {
      if (!object.isMesh) return;
      object.castShadow = false;
      object.receiveShadow = false;
      object.frustumCulled = true;
      const materials = Array.isArray(object.material)
        ? object.material
        : [object.material];
      for (const material of materials) {
        if (!material) continue;
        material.side = THREE.DoubleSide;
        if ('roughness' in material) material.roughness = 1;
        if ('metalness' in material) material.metalness = 0;
        if (material.map) {
          material.map.colorSpace = THREE.SRGBColorSpace;
          material.map.anisotropy = anisotropy;
          material.map.needsUpdate = true;
        }
        material.needsUpdate = true;
      }
    });
  }

  async _warmVisualScene(root) {
    const renderer = this.sceneManager.renderer;
    const textures = new Set();
    root.traverse((object) => {
      const materials = Array.isArray(object.material) ? object.material : [object.material];
      for (const material of materials) {
        for (const value of Object.values(material || {})) {
          if (value?.isTexture) textures.add(value);
        }
      }
    });
    // Move first-use uploads out of the visible render frame, with a bounded
    // main-thread slice. Texture pixels/materials/geometry remain unchanged.
    let sliceStart = performance.now();
    for (const texture of textures) {
      renderer.initTexture(texture);
      if (performance.now() - sliceStart >= 4) {
        await yieldToMainThread();
        sliceStart = performance.now();
      }
    }
    await renderer.compileAsync(root, this.sceneManager.camera, this.scene);
  }

  async _loadZoning(root) {
    if (!this.manifest.zoning?.uri) return null;
    this.zoningData = await fetchJson(`${root}/${this.manifest.zoning.uri}`);
    this._clearGroup(this.zoningGroup, true);

    const classMap = new Map(
      this.zoningData.classes.map((item) => [item.id, item]),
    );
    const fillMaterials = new Map();
    const lineMaterials = new Map();
    const overlayAltitude = (this.manifest.collision?.bounds?.maximum?.[1] ?? 50) + 4;
    for (const item of this.zoningData.classes) {
      fillMaterials.set(
        item.id,
        new THREE.MeshBasicMaterial({
          color: item.color,
          transparent: true,
          opacity: 0.26,
          depthTest: true,
          depthWrite: false,
          side: THREE.DoubleSide,
          toneMapped: false,
        }),
      );
      lineMaterials.set(
        item.id,
        new THREE.LineBasicMaterial({
          color: item.color,
          transparent: true,
          opacity: 0.92,
          depthTest: true,
          depthWrite: false,
          toneMapped: false,
        }),
      );
    }

    for (const feature of this.zoningData.features) {
      const category = classMap.get(feature.class_id);
      if (!category) continue;
      const polygons = feature.geometry.type === 'Polygon'
        ? [feature.geometry.coordinates]
        : feature.geometry.coordinates;
      for (const polygon of polygons) {
        if (!polygon[0]?.length) continue;
        const shape = this._shapeFromZonePolygon(polygon);
        const mesh = new THREE.Mesh(
          new THREE.ShapeGeometry(shape),
          fillMaterials.get(feature.class_id),
        );
        mesh.name = `Zone_${feature.class_id}_${feature.id}`;
        mesh.rotation.x = -Math.PI / 2;
        mesh.position.y = overlayAltitude;
        mesh.renderOrder = 40;
        mesh.userData.zone = {
          id: feature.id,
          classId: feature.class_id,
          sourceCode: feature.source_code,
          label: category.label,
          flightCost: category.flight_cost,
          policy: category.policy,
        };
        this.zoningGroup.add(mesh);

        for (const ring of polygon) {
          if (ring.length < 3) continue;
          const line = new THREE.LineLoop(
            new THREE.BufferGeometry().setFromPoints(
              ring.slice(0, -1).map(
                ([x, z]) => new THREE.Vector3(x, overlayAltitude + 0.1, z),
              ),
            ),
            lineMaterials.get(feature.class_id),
          );
          line.renderOrder = 41;
          this.zoningGroup.add(line);
        }
      }
    }
    this._populateZoneLegend();
    return this.zoningData;
  }

  _shapeFromZonePolygon(polygon) {
    const shape = new THREE.Shape();
    polygon[0].forEach(([x, z], index) => {
      if (index === 0) shape.moveTo(x, -z);
      else shape.lineTo(x, -z);
    });
    for (const ring of polygon.slice(1)) {
      const hole = new THREE.Path();
      ring.forEach(([x, z], index) => {
        if (index === 0) hole.moveTo(x, -z);
        else hole.lineTo(x, -z);
      });
      shape.holes.push(hole);
    }
    return shape;
  }

  _populateZoneLegend() {
    const legend = document.getElementById('zoning-legend');
    if (!legend || !this.zoningData) return;
    legend.replaceChildren();
    for (const category of this.zoningData.classes) {
      const item = document.createElement('div');
      const swatch = document.createElement('i');
      swatch.style.background = category.color;
      const label = document.createElement('span');
      label.textContent = `${category.label} ${category.feature_count}`;
      item.title = `${category.policy}；规划代价 ${category.flight_cost}`;
      item.append(swatch, label);
      legend.append(item);
    }
  }

  getFunctionalZone(x, z) {
    if (!this.zoningData) return null;
    for (const feature of this.zoningData.features) {
      const polygons = feature.geometry.type === 'Polygon'
        ? [feature.geometry.coordinates]
        : feature.geometry.coordinates;
      for (const polygon of polygons) {
        if (
          polygon[0]
          && this._pointInRing(x, z, polygon[0])
          && !polygon.slice(1).some((ring) => this._pointInRing(x, z, ring))
        ) {
          const category = this.zoningData.classes.find(
            (item) => item.id === feature.class_id,
          );
          return { ...feature, category };
        }
      }
    }
    return null;
  }

  _pointInRing(x, z, ring) {
    let inside = false;
    for (let i = 0, j = ring.length - 1; i < ring.length; j = i, i += 1) {
      const [xi, zi] = ring[i];
      const [xj, zj] = ring[j];
      if (
        ((zi > z) !== (zj > z))
        && (x < ((xj - xi) * (z - zi)) / (zj - zi) + xi)
      ) {
        inside = !inside;
      }
    }
    return inside;
  }

  async _loadCollisionDiagnostics(root, loader) {
    const collision = this.manifest.collision;
    const gltf = await loader.loadAsync(`${root}/${collision.uri}`);
    this._clearGroup(this.collisionDebugGroup);
    this.colliders = [];

    gltf.scene.traverse((object) => {
      if (!object.isMesh) return;
      object.updateMatrixWorld(true);
      object.geometry.computeBoundsTree({ targetLeafSize: 16 });
      this.colliders.push(object);
      object.material = new THREE.MeshBasicMaterial({
        color: 0xffa44d,
        wireframe: true,
        transparent: true,
        opacity: 0.16,
        depthTest: false,
        depthWrite: false,
        side: THREE.DoubleSide,
        toneMapped: false,
      });
      object.renderOrder = 30;
    });
    this.collisionDebugGroup.add(gltf.scene);

    this.collisionDiagnosticsLoaded = true;
    this._updateCollisionStats(collision);
    return collision;
  }

  async _setEsdfAltitude(altitude) {
    if (!this.manifest) return;
    this.esdfAltitude = Number(altitude);
    const slice = this.manifest.collision.esdf_slices.find(
      (item) => Number(item.altitude_m) === this.esdfAltitude,
    );
    if (!slice) return;

    const root = `/data/helsinki_mesh/${LICENSED_ASSET_SCENE}`;
    const texture = await new THREE.TextureLoader().loadAsync(
      `${root}/${slice.image}`,
    );
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.minFilter = THREE.LinearFilter;
    texture.magFilter = THREE.LinearFilter;

    this._clearGroup(this.esdfDebugGroup, true);
    const size = this.manifest.operation_size_m;
    const plane = new THREE.Mesh(
      new THREE.PlaneGeometry(size, size),
      new THREE.MeshBasicMaterial({
        map: texture,
        transparent: true,
        opacity: 0.58,
        depthTest: false,
        depthWrite: false,
        side: THREE.DoubleSide,
        toneMapped: false,
      }),
    );
    plane.name = `ESDF_${this.esdfAltitude}m`;
    plane.rotation.x = -Math.PI / 2;
    plane.position.y = this.esdfAltitude;
    plane.renderOrder = 31;
    this.esdfDebugGroup.add(plane);

    const half = size / 2;
    const border = new THREE.LineLoop(
      new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(-half, this.esdfAltitude + 0.15, -half),
        new THREE.Vector3(half, this.esdfAltitude + 0.15, -half),
        new THREE.Vector3(half, this.esdfAltitude + 0.15, half),
        new THREE.Vector3(-half, this.esdfAltitude + 0.15, half),
      ]),
      new THREE.LineBasicMaterial({
        color: 0x5cd8ff,
        transparent: true,
        opacity: 0.9,
        depthTest: false,
        depthWrite: false,
        toneMapped: false,
      }),
    );
    border.renderOrder = 32;
    this.esdfDebugGroup.add(border);
    this.esdfSliceLoaded = true;
  }

  _updateCollisionStats(collision) {
    const stats = document.getElementById('collision-layer-stats');
    if (!stats || !collision) return;
    const watertight = collision.watertight
      ? '闭合网格'
      : '三角网格 BVH';
    stats.textContent = `${collision.triangles.toLocaleString()} 三角面 · ${watertight} · ESDF ${collision.heightmap.resolution_m} m`;
  }

  async _ensureCollisionDiagnostics() {
    if (this.collisionDiagnosticsLoaded || !this.assetRoot || !this.loader) {
      return;
    }
    await yieldToMainThread();
    await this._loadCollisionDiagnostics(this.assetRoot, this.loader);
  }

  async _ensureEsdfSlice() {
    if (this.esdfSliceLoaded) return;
    await yieldToMainThread();
    await this._setEsdfAltitude(this.esdfAltitude);
  }

  _populateEsdfOptions() {
    const select = document.getElementById('esdf-altitude-select');
    if (!select) return;
    select.replaceChildren();
    for (const slice of this.manifest.collision.esdf_slices) {
      const option = document.createElement('option');
      option.value = String(slice.altitude_m);
      option.textContent = `${slice.altitude_m} m`;
      option.selected = Number(slice.altitude_m) === this.esdfAltitude;
      select.append(option);
    }
  }

  _bindDiagnosticsControls() {
    const zoningButton = document.getElementById('toggle-zoning-layer');
    const collisionButton = document.getElementById('toggle-collision-layer');
    const esdfButton = document.getElementById('toggle-esdf-layer');
    const altitudeSelect = document.getElementById('esdf-altitude-select');

    zoningButton?.addEventListener('click', () => {
      this.zoningGroup.visible = !this.zoningGroup.visible;
      if (this.zoningGroup.visible) {
        window.dispatchEvent(new KeyboardEvent('keydown', { key: '3' }));
      }
      zoningButton.classList.toggle('is-active', this.zoningGroup.visible);
      zoningButton.setAttribute(
        'aria-pressed',
        String(this.zoningGroup.visible),
      );
      this.sceneManager.invalidate();
    });
    collisionButton?.addEventListener('click', async () => {
      const shouldShow = !this.collisionDebugGroup.visible;
      if (shouldShow) await this._ensureCollisionDiagnostics();
      this.collisionDebugGroup.visible = shouldShow;
      collisionButton.classList.toggle(
        'is-active',
        this.collisionDebugGroup.visible,
      );
      collisionButton.setAttribute(
        'aria-pressed',
        String(this.collisionDebugGroup.visible),
      );
      this.sceneManager.invalidate();
    });
    esdfButton?.addEventListener('click', async () => {
      const shouldShow = !this.esdfDebugGroup.visible;
      if (shouldShow) await this._ensureEsdfSlice();
      this.esdfDebugGroup.visible = shouldShow;
      esdfButton.classList.toggle(
        'is-active',
        this.esdfDebugGroup.visible,
      );
      esdfButton.setAttribute(
        'aria-pressed',
        String(this.esdfDebugGroup.visible),
      );
      this.sceneManager.invalidate();
    });
    altitudeSelect?.addEventListener('change', async (event) => {
      this.esdfAltitude = Number(event.target.value);
      this.esdfSliceLoaded = false;
      if (this.esdfDebugGroup.visible) await this._ensureEsdfSlice();
      this.sceneManager.invalidate();
    });
  }

  _configureCamera(cameraConfig) {
    const camera = this.sceneManager.camera;
    const target = new THREE.Vector3().fromArray(cameraConfig.target);
    camera.up.set(0, 1, 0);
    camera.position.fromArray(cameraConfig.position);
    camera.near = 0.25;
    camera.far = 3000;
    camera.fov = cameraConfig.fov_degrees;
    camera.updateProjectionMatrix();
    camera.lookAt(target);

    this.sceneManager.controls.target.copy(target);
    this.sceneManager.controls.minDistance = 8;
    this.sceneManager.controls.maxDistance = 1200;
    this.sceneManager.controls.update();
  }

  _createOperationalVolume(size) {
    this._clearGroup(this.operationalGroup, true);
    const half = size / 2;
    const floorY = 0.6;
    const ceilingY = 120;
    const primary = new THREE.LineBasicMaterial({
      color: 0x7ce7ff,
      transparent: true,
      opacity: 0.62,
    });
    const subtle = new THREE.LineBasicMaterial({
      color: 0xb8efff,
      transparent: true,
      opacity: 0.16,
    });

    const floorPoints = [
      new THREE.Vector3(-half, floorY, -half),
      new THREE.Vector3(half, floorY, -half),
      new THREE.Vector3(half, floorY, half),
      new THREE.Vector3(-half, floorY, half),
    ];
    this.operationalGroup.add(
      new THREE.LineLoop(
        new THREE.BufferGeometry().setFromPoints(floorPoints),
        primary,
      ),
    );
    this.operationalGroup.add(
      new THREE.LineLoop(
        new THREE.BufferGeometry().setFromPoints(
          floorPoints.map((point) => point.clone().setY(ceilingY)),
        ),
        subtle,
      ),
    );
    const verticalGeometry = new THREE.BufferGeometry().setFromPoints(
      floorPoints.flatMap((point) => [
        point,
        point.clone().setY(ceilingY),
      ]),
    );
    this.operationalGroup.add(new THREE.LineSegments(verticalGeometry, subtle));
  }

  _clearGroup(group, dispose = false) {
    if (dispose) {
      this._disposeObject(group);
    }
    group.clear();
  }

  _disposeObject(root) {
    root.traverse((object) => {
      if (object.geometry) object.geometry.dispose();
      const materials = Array.isArray(object.material)
        ? object.material
        : [object.material];
      for (const material of materials) {
        if (!material) continue;
        if (material.map) material.map.dispose();
        material.dispose();
      }
    });
  }
}

export { OPERATION_SIZE_METERS };
