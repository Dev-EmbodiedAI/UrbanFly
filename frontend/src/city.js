/**
 * 城市渲染器
 * ==========
 * 使用 city_layout.json 构建分区、道路、建筑和街区细节。
 */

import * as THREE from 'three';

const STYLE_COLORS = {
  modern_glass: 0x9bb7d1,
  classic_stone: 0xcabbab,
  brick: 0x9b715b,
  postmodern: 0xb7ab95,
  art_deco: 0xc3b285,
  skyscraper: 0x8ca5bf,
  highrise: 0xcfc5b8,
  midrise: 0xd8cec0,
  lowrise: 0xe7ded2,
};

const DISTRICT_COLORS = {
  cbd: 0x19344c,
  mixed: 0x4c483a,
  residential: 0x314e40,
  industrial: 0x58463b,
  park: 0x244c35,
  plaza: 0x6d675b,
};

const ROAD_COLORS = {
  arterial: 0x161d26,
  collector: 0x222a33,
  local: 0x2d3640,
};

export class CityRenderer {
  constructor(scene) {
    this.scene = scene;
    this.cityGroup = new THREE.Group();
    this.cityGroup.name = 'GeneratedCity';
    this.scene.add(this.cityGroup);

    this.atlasTexture = null;
    this.zoneTextures = new Map();
    this.windowTextures = new Map();
    this.lightBudget = 18;
    this.blockBuildings = new Map();
  }

  async loadDefault() {
    try {
      const resp = await fetch('/data/scene/city_layout.json');
      const layout = await resp.json();
      this._buildCity(layout);
      console.log(
        `[City] Rendered ${layout.buildings?.length || 0} buildings, ${layout.blocks?.length || 0} blocks`
      );
    } catch (e) {
      console.warn('[City] No layout, generating placeholder', e.message);
      this._generatePlaceholder();
    }
  }

  _loadTexture(url) {
    return new Promise((resolve, reject) => {
      new THREE.TextureLoader().load(url, resolve, undefined, reject);
    });
  }

  _buildCity(layout) {
    if (!layout) return;
    this.blockBuildings = this._indexBuildingsByBlock(layout.buildings || []);
    this._renderBlocks(layout.blocks || []);
    this._renderRoads(layout.roads || []);
    this._renderBuildings(layout.buildings || []);
  }

  _renderBlocks(blocks) {
    const surfaceMatCache = new Map();
    const lineMatCache = new Map();

    for (const block of blocks) {
      const polygon = Array.isArray(block.polygon) ? block.polygon : [];
      if (polygon.length < 3) continue;

      const district = block.district || 'mixed';
      const shape = new THREE.Shape();
      shape.moveTo(polygon[0][0], polygon[0][1]);
      for (let i = 1; i < polygon.length; i++) {
        shape.lineTo(polygon[i][0], polygon[i][1]);
      }
      shape.closePath();

      const extrude = new THREE.ExtrudeGeometry(shape, {
        depth: 0.18,
        bevelEnabled: false,
      });
      extrude.rotateX(-Math.PI / 2);

      if (!surfaceMatCache.has(district)) {
        surfaceMatCache.set(
          district,
          new THREE.MeshStandardMaterial({
            map: this._getZoneTexture(district),
            color: DISTRICT_COLORS[district] || 0x5c6570,
            transparent: true,
            opacity: district === 'park' ? 0.94 : 0.78,
            roughness: district === 'plaza' ? 0.9 : 0.96,
            metalness: district === 'cbd' ? 0.08 : 0.03,
          })
        );
      }

      const surface = new THREE.Mesh(extrude, surfaceMatCache.get(district));
      surface.position.y = district === 'park' ? -0.03 : -0.05;
      surface.receiveShadow = true;
      this.cityGroup.add(surface);

      if (!lineMatCache.has(district)) {
        lineMatCache.set(
          district,
          new THREE.LineBasicMaterial({
            color: this._mixColors(DISTRICT_COLORS[district] || 0x6688aa, 0xaed9ff, 0.28),
            transparent: true,
            opacity: 0.28,
          })
        );
      }
      const outline = new THREE.LineLoop(this._polygonLineGeometry(polygon, 0.07), lineMatCache.get(district));
      this.cityGroup.add(outline);

      if (district === 'park') {
        this._addParkDetails(block, polygon);
      } else if (district === 'plaza') {
        this._addPlazaDetails(block, polygon);
      } else {
        this._addDistrictBlockAssets(block, polygon);
      }
    }
  }

  _renderRoads(roads) {
    for (const road of roads) {
      const [x1, z1] = road.start;
      const [x2, z2] = road.end;
      const dx = x2 - x1;
      const dz = z2 - z1;
      const length = Math.hypot(dx, dz);
      if (length < 1) continue;

      const roadClass = road.class || 'local';
      const width = Math.max(road.width || 6, roadClass === 'arterial' ? 18 : roadClass === 'collector' ? 10 : 6);
      const angle = Math.atan2(dz, dx);

      const roadDeck = new THREE.Mesh(
        new THREE.BoxGeometry(length, 0.16, width),
        new THREE.MeshStandardMaterial({
          color: ROAD_COLORS[roadClass] || 0x30343a,
          roughness: 0.98,
          metalness: 0.02,
        })
      );
      roadDeck.position.set((x1 + x2) / 2, -0.09, (z1 + z2) / 2);
      roadDeck.rotation.y = angle;
      roadDeck.receiveShadow = true;
      this.cityGroup.add(roadDeck);

      const shoulder = new THREE.Mesh(
        new THREE.BoxGeometry(length, 0.04, width + 0.9),
        new THREE.MeshBasicMaterial({
          color: 0x51606f,
          transparent: true,
          opacity: 0.1,
        })
      );
      shoulder.position.set((x1 + x2) / 2, -0.005, (z1 + z2) / 2);
      shoulder.rotation.y = angle;
      this.cityGroup.add(shoulder);

      const stripes = this._createRoadMarkings(length, width, roadClass);
      stripes.position.set((x1 + x2) / 2, 0.01, (z1 + z2) / 2);
      stripes.rotation.y = angle;
      this.cityGroup.add(stripes);

      if (roadClass !== 'local' && length > 70) {
        this._addStreetLights(x1, z1, x2, z2, width, roadClass);
      }
    }
  }

  _renderBuildings(buildings) {
    const geometryCache = new Map();
    const materialCache = new Map();

    for (const b of buildings) {
      const { x, z, w, d, h, style, district } = b;
      if (!w || !d || !h) continue;

      const geoKey = `${w.toFixed(1)},${h.toFixed(1)},${d.toFixed(1)}`;
      let geo = geometryCache.get(geoKey);
      if (!geo) {
        geo = new THREE.BoxGeometry(w, h, d);
        geometryCache.set(geoKey, geo);
      }

      const matKey = `${style || 'midrise'}:${district || 'mixed'}`;
      let mat = materialCache.get(matKey);
      if (!mat) {
        mat = this._createBuildingMaterial(style, district);
        materialCache.set(matKey, mat);
      }

      const body = new THREE.Mesh(geo, mat);
      body.position.set(x, h / 2, z);
      body.castShadow = true;
      body.receiveShadow = true;
      this.cityGroup.add(body);

      const crown = new THREE.Mesh(
        new THREE.BoxGeometry(w * 0.96, Math.max(0.8, Math.min(2.4, h * 0.03)), d * 0.96),
        new THREE.MeshStandardMaterial({
          color: this._mixColors(STYLE_COLORS[style] || 0xd6cdc1, 0xf4fbff, 0.3),
          roughness: 0.46,
          metalness: 0.18,
          emissive: district === 'cbd' ? 0x1a3958 : 0x000000,
          emissiveIntensity: district === 'cbd' ? 0.35 : 0.0,
        })
      );
      crown.position.set(x, h + crown.geometry.parameters.height / 2, z);
      crown.castShadow = true;
      this.cityGroup.add(crown);

      const edge = new THREE.LineSegments(
        new THREE.EdgesGeometry(geo, 30),
        new THREE.LineBasicMaterial({
          color: this._mixColors(STYLE_COLORS[style] || 0xd6cdc1, 0xe7f3ff, 0.22),
          transparent: true,
          opacity: 0.12,
        })
      );
      edge.position.copy(body.position);
      this.cityGroup.add(edge);

      if (b.has_podium && h > 18) {
        const podiumHeight = Math.max(4, h * 0.16);
        const podium = new THREE.Mesh(
          new THREE.BoxGeometry(w * 1.14, podiumHeight, d * 1.14),
          this._createBuildingMaterial(style, district, true)
        );
        podium.position.set(x, podiumHeight / 2, z);
        podium.castShadow = true;
        podium.receiveShadow = true;
        this.cityGroup.add(podium);
      }

      if (district === 'cbd' && h > 46) {
        this._addCbdSetbacks(x, z, w, d, h);
        this._addCbdCrown(x, z, w, d, h);
        this._addRoofBeacon(x, z, h, Math.max(w, d) * 0.22);
      } else if (district === 'industrial' && h > 18) {
        this._addIndustrialSawtoothRoof(x, z, w, d, h);
        this._addIndustrialRooftop(x, z, w, d, h);
      } else if (district === 'residential' && h < 28) {
        this._addResidentialBalconies(x, z, w, d, h);
        this._addResidentialRoof(x, z, w, d, h, style);
      } else if (district === 'mixed' && h > 16) {
        this._addMixedAnnex(x, z, w, d, h);
        this._addMixedTerrace(x, z, w, d, h);
      }
    }
  }

  _indexBuildingsByBlock(buildings) {
    const grouped = new Map();
    for (const building of buildings) {
      const blockId = building.block_id;
      if (!grouped.has(blockId)) grouped.set(blockId, []);
      grouped.get(blockId).push(building);
    }
    return grouped;
  }

  _addDistrictBlockAssets(block, polygon) {
    const district = block.district || 'mixed';
    const anchors = this._getBlockAnchors(polygon);
    if (!anchors) return;

    if (district === 'industrial') {
      this._addIndustrialYard(block, anchors);
    } else if (district === 'residential') {
      this._addResidentialCourtyard(block, anchors);
    } else if (district === 'mixed') {
      this._addMixedBlockAmenities(block, anchors);
    } else if (district === 'cbd') {
      this._addCbdForecourt(block, anchors);
    }
  }

  _addParkDetails(block, polygon) {
    const center = this._polygonCentroid(polygon);
    const radius = Math.max(6, Math.min(20, Math.sqrt(block.area || 120) * 0.16));

    const lawn = new THREE.Mesh(
      new THREE.CircleGeometry(radius, 28),
      new THREE.MeshStandardMaterial({
        color: 0x3b8056,
        roughness: 0.97,
        metalness: 0.0,
      })
    );
    lawn.rotation.x = -Math.PI / 2;
    lawn.position.set(center.x, 0.01, center.z);
    this.cityGroup.add(lawn);

    const treeCount = Math.max(3, Math.min(8, Math.round(radius / 2.8)));
    for (let i = 0; i < treeCount; i++) {
      const angle = (i / treeCount) * Math.PI * 2 + Math.random() * 0.5;
      const dist = radius * (0.25 + Math.random() * 0.48);
      this._addTree(
        center.x + Math.cos(angle) * dist,
        center.z + Math.sin(angle) * dist,
        0.45 + Math.random() * 0.35
      );
    }
  }

  _addPlazaDetails(block, polygon) {
    const center = this._polygonCentroid(polygon);
    const footprint = Math.max(8, Math.min(24, Math.sqrt(block.area || 180) * 0.18));

    const pad = new THREE.Mesh(
      new THREE.CircleGeometry(footprint, 32),
      new THREE.MeshStandardMaterial({
        color: 0x8c8472,
        roughness: 0.92,
        metalness: 0.03,
      })
    );
    pad.rotation.x = -Math.PI / 2;
    pad.position.set(center.x, 0.02, center.z);
    this.cityGroup.add(pad);

    const ring = new THREE.Mesh(
      new THREE.RingGeometry(footprint * 0.42, footprint * 0.74, 32),
      new THREE.MeshBasicMaterial({
        color: 0xc6d8ea,
        transparent: true,
        opacity: 0.14,
        side: THREE.DoubleSide,
      })
    );
    ring.rotation.x = -Math.PI / 2;
    ring.position.set(center.x, 0.03, center.z);
    this.cityGroup.add(ring);

    const marker = new THREE.Mesh(
      new THREE.CylinderGeometry(3.4, 4.6, 1.2, 20),
      new THREE.MeshStandardMaterial({
        color: 0xa89a80,
        roughness: 0.76,
        metalness: 0.08,
      })
    );
    marker.position.set(center.x, 0.6, center.z);
    marker.receiveShadow = true;
    this.cityGroup.add(marker);

    const anchors = this._getBlockAnchors(polygon);
    if (anchors) {
      for (const anchor of anchors.edgeCenters.slice(0, 2)) {
        this._addBench(anchor.x, anchor.z, anchor.angle);
      }
      for (const anchor of anchors.innerCorners.slice(0, 2)) {
        this._addLightPylon(anchor.x, anchor.z, 4.8, 0x93d7ff, false);
      }
    }
  }

  _addTree(x, z, scale = 1) {
    const trunk = new THREE.Mesh(
      new THREE.CylinderGeometry(0.16 * scale, 0.22 * scale, 1.4 * scale, 8),
      new THREE.MeshStandardMaterial({
        color: 0x5d4332,
        roughness: 0.98,
      })
    );
    trunk.position.set(x, 0.7 * scale, z);
    trunk.castShadow = true;
    trunk.receiveShadow = true;
    this.cityGroup.add(trunk);

    const crown = new THREE.Mesh(
      new THREE.SphereGeometry(0.85 * scale, 10, 10),
      new THREE.MeshStandardMaterial({
        color: 0x4e8a61,
        roughness: 0.95,
      })
    );
    crown.position.set(x, 1.6 * scale, z);
    crown.castShadow = true;
    this.cityGroup.add(crown);
  }

  _addDistrictGroundPad(x, z, w, d, color, opacity = 0.9) {
    const pad = new THREE.Mesh(
      new THREE.PlaneGeometry(w, d),
      new THREE.MeshStandardMaterial({
        color,
        transparent: true,
        opacity,
        roughness: 0.95,
        metalness: 0.02,
      })
    );
    pad.rotation.x = -Math.PI / 2;
    pad.position.set(x, 0.01, z);
    this.cityGroup.add(pad);
    return pad;
  }

  _addIndustrialYard(block, anchors) {
    const areaScale = Math.min(1.2, Math.max(0.7, Math.sqrt((block.area || 8000) / 12000)));
    this._addDistrictGroundPad(
      anchors.center.x,
      anchors.center.z,
      anchors.innerWidth * 0.42 * areaScale,
      anchors.innerDepth * 0.32 * areaScale,
      0x5d5951,
      0.88
    );

    const containers = anchors.innerCorners.slice(0, 3);
    for (const anchor of containers) {
      this._addContainerStack(anchor.x, anchor.z, 1 + Math.floor(Math.random() * 2));
    }

    const siloAnchor = anchors.edgeCenters[0];
    this._addSiloCluster(siloAnchor.x, siloAnchor.z, 3);

    const loading = anchors.edgeCenters[1];
    this._addLoadingBays(loading.x, loading.z, loading.angle, 4);
  }

  _addResidentialCourtyard(block, anchors) {
    const courtW = anchors.innerWidth * 0.34;
    const courtD = anchors.innerDepth * 0.24;
    this._addDistrictGroundPad(anchors.center.x, anchors.center.z, courtW, courtD, 0x6c715d, 0.82);

    const lawn = new THREE.Mesh(
      new THREE.PlaneGeometry(courtW * 0.74, courtD * 0.56),
      new THREE.MeshStandardMaterial({
        color: 0x4d815d,
        roughness: 0.98,
      })
    );
    lawn.rotation.x = -Math.PI / 2;
    lawn.position.set(anchors.center.x, 0.02, anchors.center.z);
    this.cityGroup.add(lawn);

    for (const anchor of anchors.innerCorners) {
      this._addTree(anchor.x, anchor.z, 0.44 + Math.random() * 0.15);
    }

    for (const anchor of anchors.edgeCenters.slice(0, 2)) {
      this._addParkedCar(anchor.x, anchor.z, anchor.angle);
    }
  }

  _addMixedBlockAmenities(block, anchors) {
    const kioskAnchor = anchors.edgeCenters[0];
    this._addCanopy(kioskAnchor.x, kioskAnchor.z, kioskAnchor.angle, 5.4, 3.4, 0x3a6d95);
    this._addBench(kioskAnchor.x + Math.cos(kioskAnchor.angle) * 2.4, kioskAnchor.z + Math.sin(kioskAnchor.angle) * 2.4, kioskAnchor.angle);

    const plazaAnchor = anchors.center;
    this._addDistrictGroundPad(plazaAnchor.x, plazaAnchor.z, anchors.innerWidth * 0.26, anchors.innerDepth * 0.18, 0x726d63, 0.78);

    for (const anchor of anchors.innerCorners.slice(0, 2)) {
      this._addTree(anchor.x, anchor.z, 0.48);
    }
  }

  _addCbdForecourt(block, anchors) {
    const forecourt = this._addDistrictGroundPad(
      anchors.center.x,
      anchors.center.z,
      anchors.innerWidth * 0.3,
      anchors.innerDepth * 0.24,
      0x586777,
      0.84
    );
    forecourt.material.emissive = new THREE.Color(0x0d2030);
    forecourt.material.emissiveIntensity = 0.26;

    const ring = new THREE.Mesh(
      new THREE.RingGeometry(Math.min(anchors.innerWidth, anchors.innerDepth) * 0.06, Math.min(anchors.innerWidth, anchors.innerDepth) * 0.12, 24),
      new THREE.MeshBasicMaterial({
        color: 0x7dd0ff,
        transparent: true,
        opacity: 0.2,
        side: THREE.DoubleSide,
      })
    );
    ring.rotation.x = -Math.PI / 2;
    ring.position.set(anchors.center.x, 0.02, anchors.center.z);
    this.cityGroup.add(ring);

    for (const anchor of anchors.innerCorners.slice(0, 3)) {
      this._addLightPylon(anchor.x, anchor.z, 6.2, 0x7cc9ff, true);
    }

    const landmark = anchors.edgeCenters[0];
    this._addSignTotem(landmark.x, landmark.z, landmark.angle);
  }

  _addRoofBeacon(x, z, h, radius) {
    const base = new THREE.Mesh(
      new THREE.CylinderGeometry(radius * 0.55, radius * 0.65, 1.4, 16),
      new THREE.MeshStandardMaterial({
        color: 0x93a5b8,
        roughness: 0.58,
        metalness: 0.4,
      })
    );
    base.position.set(x, h + 0.7, z);
    base.castShadow = true;
    this.cityGroup.add(base);

    const beacon = new THREE.Mesh(
      new THREE.SphereGeometry(radius * 0.16, 12, 12),
      new THREE.MeshStandardMaterial({
        color: 0x8fd7ff,
        emissive: 0x4ec8ff,
        emissiveIntensity: 1.25,
        roughness: 0.25,
        metalness: 0.1,
      })
    );
    beacon.position.set(x, h + 1.7, z);
    this.cityGroup.add(beacon);
  }

  _addRoofAntenna(x, z, h) {
    const mast = new THREE.Mesh(
      new THREE.CylinderGeometry(0.08, 0.12, 6 + Math.random() * 8, 8),
      new THREE.MeshStandardMaterial({
        color: 0x848d94,
        metalness: 0.6,
        roughness: 0.38,
      })
    );
    mast.position.set(x + (Math.random() - 0.5) * 2.5, h + mast.geometry.parameters.height / 2, z + (Math.random() - 0.5) * 2.5);
    mast.castShadow = true;
    this.cityGroup.add(mast);
  }

  _addCbdSetbacks(x, z, w, d, h) {
    const upperH = Math.max(7, h * 0.18);
    const midH = Math.max(5, h * 0.12);
    const upper = new THREE.Mesh(
      new THREE.BoxGeometry(w * 0.74, upperH, d * 0.74),
      new THREE.MeshPhysicalMaterial({
        color: 0xa9bfd7,
        emissive: 0x5ea9ff,
        emissiveIntensity: 0.22,
        roughness: 0.32,
        metalness: 0.28,
      })
    );
    upper.position.set(x, h - upperH / 2, z);
    upper.castShadow = true;
    this.cityGroup.add(upper);

    const mid = new THREE.Mesh(
      new THREE.BoxGeometry(w * 0.86, midH, d * 0.86),
      new THREE.MeshStandardMaterial({
        color: 0xb7c9d8,
        roughness: 0.4,
        metalness: 0.18,
      })
    );
    mid.position.set(x, h - upperH - midH / 2, z);
    mid.castShadow = true;
    this.cityGroup.add(mid);
  }

  _addCbdCrown(x, z, w, d, h) {
    const frame = new THREE.Mesh(
      new THREE.BoxGeometry(w * 0.82, Math.max(1.6, h * 0.04), d * 0.82),
      new THREE.MeshPhysicalMaterial({
        color: 0xa9bfd7,
        emissive: 0x4ca5ff,
        emissiveIntensity: 0.28,
        roughness: 0.28,
        metalness: 0.36,
        clearcoat: 0.42,
      })
    );
    frame.position.set(x, h + frame.geometry.parameters.height * 0.9, z);
    frame.castShadow = true;
    this.cityGroup.add(frame);

    const spine = new THREE.Mesh(
      new THREE.BoxGeometry(w * 0.14, Math.max(5, h * 0.1), d * 0.14),
      new THREE.MeshStandardMaterial({
        color: 0xc9d7e6,
        emissive: 0x61b7ff,
        emissiveIntensity: 0.36,
        roughness: 0.34,
        metalness: 0.25,
      })
    );
    spine.position.set(x, h + spine.geometry.parameters.height / 2 + frame.geometry.parameters.height, z);
    spine.castShadow = true;
    this.cityGroup.add(spine);
  }

  _addIndustrialRooftop(x, z, w, d, h) {
    const unitCount = Math.max(1, Math.min(4, Math.round((w + d) / 18)));
    for (let i = 0; i < unitCount; i++) {
      const boxW = Math.max(1.8, w * (0.14 + Math.random() * 0.12));
      const boxD = Math.max(1.8, d * (0.12 + Math.random() * 0.1));
      const boxH = 1.6 + Math.random() * 2.4;
      const unit = new THREE.Mesh(
        new THREE.BoxGeometry(boxW, boxH, boxD),
        new THREE.MeshStandardMaterial({
          color: 0x7f868d,
          roughness: 0.68,
          metalness: 0.24,
        })
      );
      unit.position.set(
        x + (Math.random() - 0.5) * Math.max(2, w * 0.45),
        h + boxH / 2 + 0.3,
        z + (Math.random() - 0.5) * Math.max(2, d * 0.45)
      );
      unit.castShadow = true;
      unit.receiveShadow = true;
      this.cityGroup.add(unit);
    }

    if (Math.random() > 0.32) {
      this._addRoofAntenna(x, z, h);
    }
  }

  _addIndustrialSawtoothRoof(x, z, w, d, h) {
    if (w * d < 900 || h > 40) return;
    const ridgeCount = Math.max(2, Math.min(5, Math.round(w / 10)));
    for (let i = 0; i < ridgeCount; i++) {
      const ridge = new THREE.Mesh(
        new THREE.BoxGeometry(w * 0.12, 1.2 + Math.random() * 1.1, d * 0.84),
        new THREE.MeshStandardMaterial({
          color: 0xb1967b,
          roughness: 0.82,
          metalness: 0.06,
        })
      );
      ridge.position.set(
        x - w * 0.32 + (i / Math.max(1, ridgeCount - 1)) * w * 0.64,
        h + ridge.geometry.parameters.height / 2,
        z
      );
      ridge.castShadow = true;
      this.cityGroup.add(ridge);
    }
  }

  _addResidentialRoof(x, z, w, d, h, style) {
    const cap = new THREE.Mesh(
      new THREE.BoxGeometry(w * 1.03, 1.1, d * 1.03),
      new THREE.MeshStandardMaterial({
        color: style === 'brick' ? 0x7d4235 : 0x5d6975,
        roughness: 0.88,
        metalness: 0.06,
      })
    );
    cap.position.set(x, h + 0.55, z);
    cap.castShadow = true;
    cap.receiveShadow = true;
    this.cityGroup.add(cap);

    const tank = new THREE.Mesh(
      new THREE.CylinderGeometry(0.8, 0.9, 1.8, 12),
      new THREE.MeshStandardMaterial({
        color: 0x90979f,
        roughness: 0.56,
        metalness: 0.3,
      })
    );
    tank.position.set(x + w * 0.18, h + 1.45, z - d * 0.18);
    tank.castShadow = true;
    this.cityGroup.add(tank);
  }

  _addResidentialBalconies(x, z, w, d, h) {
    if (w < 12 || d < 12 || h < 12) return;
    const levels = Math.max(2, Math.min(5, Math.floor(h / 5.5)));
    for (let i = 1; i <= levels; i++) {
      const y = (h / (levels + 1)) * i;
      const front = new THREE.Mesh(
        new THREE.BoxGeometry(w * 0.82, 0.14, 0.6),
        new THREE.MeshStandardMaterial({
          color: 0xc2c7cc,
          roughness: 0.78,
          metalness: 0.08,
        })
      );
      front.position.set(x, y, z + d / 2 + 0.24);
      this.cityGroup.add(front);

      const back = front.clone();
      back.position.z = z - d / 2 - 0.24;
      this.cityGroup.add(back);
    }
  }

  _addMixedTerrace(x, z, w, d, h) {
    const terrace = new THREE.Mesh(
      new THREE.BoxGeometry(w * 0.68, 0.8, d * 0.68),
      new THREE.MeshStandardMaterial({
        color: 0x6d7782,
        roughness: 0.74,
        metalness: 0.12,
      })
    );
    terrace.position.set(x, h + 0.42, z);
    terrace.castShadow = true;
    this.cityGroup.add(terrace);
  }

  _addMixedAnnex(x, z, w, d, h) {
    const annex = new THREE.Mesh(
      new THREE.BoxGeometry(w * 0.24, Math.max(4, h * 0.28), d * 0.42),
      new THREE.MeshStandardMaterial({
        color: 0x8e938f,
        roughness: 0.68,
        metalness: 0.1,
      })
    );
    annex.position.set(x + w * 0.34, annex.geometry.parameters.height / 2, z);
    annex.castShadow = true;
    annex.receiveShadow = true;
    this.cityGroup.add(annex);
  }

  _addStreetLights(x1, z1, x2, z2, width, roadClass) {
    const dx = x2 - x1;
    const dz = z2 - z1;
    const length = Math.hypot(dx, dz);
    if (length < 1) return;

    const dirX = dx / length;
    const dirZ = dz / length;
    const perpX = -dirZ;
    const perpZ = dirX;
    const spacing = roadClass === 'arterial' ? 54 : 68;
    const count = Math.min(6, Math.max(2, Math.floor(length / spacing)));
    const sideOffset = width * 0.58;

    for (let i = 0; i < count; i++) {
      const t = (i + 0.5) / count;
      const baseX = x1 + dx * t;
      const baseZ = z1 + dz * t;
      const leftFirst = i % 2 === 0;
      const side = leftFirst ? 1 : -1;

      this._placeStreetLight(
        baseX + perpX * sideOffset * side,
        baseZ + perpZ * sideOffset * side,
        roadClass === 'arterial'
      );
    }
  }

  _placeStreetLight(x, z, withPointLight = false) {
    const pole = new THREE.Mesh(
      new THREE.CylinderGeometry(0.08, 0.12, 7.5, 8),
      new THREE.MeshStandardMaterial({
        color: 0x9aa4ad,
        roughness: 0.52,
        metalness: 0.46,
      })
    );
    pole.position.set(x, 3.75, z);
    pole.castShadow = true;
    this.cityGroup.add(pole);

    const arm = new THREE.Mesh(
      new THREE.BoxGeometry(1.8, 0.08, 0.08),
      new THREE.MeshStandardMaterial({
        color: 0x9aa4ad,
        roughness: 0.48,
        metalness: 0.42,
      })
    );
    arm.position.set(x + 0.75, 7.15, z);
    arm.castShadow = true;
    this.cityGroup.add(arm);

    const lamp = new THREE.Mesh(
      new THREE.SphereGeometry(0.18, 10, 10),
      new THREE.MeshStandardMaterial({
        color: 0xffddb2,
        emissive: 0xffc06c,
        emissiveIntensity: 1.45,
        roughness: 0.16,
      })
    );
    lamp.position.set(x + 1.58, 7.02, z);
    this.cityGroup.add(lamp);

    if (withPointLight && this.lightBudget > 0) {
      const light = new THREE.PointLight(0xffbf78, 1.2, 56, 2.1);
      light.position.set(x + 1.3, 6.9, z);
      this.cityGroup.add(light);
      this.lightBudget -= 1;
    }
  }

  _addContainerStack(x, z, levels = 1) {
    const palette = [0x874848, 0x476995, 0x8a6f3c, 0x707a83];
    for (let i = 0; i < levels; i++) {
      const container = new THREE.Mesh(
        new THREE.BoxGeometry(5.2, 2.3, 2.4),
        new THREE.MeshStandardMaterial({
          color: palette[(i + Math.floor(Math.random() * palette.length)) % palette.length],
          roughness: 0.84,
          metalness: 0.08,
        })
      );
      container.position.set(
        x + (Math.random() - 0.5) * 3.6,
        1.15 + i * 2.35,
        z + (Math.random() - 0.5) * 2.4
      );
      container.castShadow = true;
      container.receiveShadow = true;
      this.cityGroup.add(container);
    }
  }

  _addSiloCluster(x, z, count = 3) {
    for (let i = 0; i < count; i++) {
      const silo = new THREE.Mesh(
        new THREE.CylinderGeometry(1.4, 1.55, 8 + Math.random() * 4, 16),
        new THREE.MeshStandardMaterial({
          color: 0xa3aaaf,
          roughness: 0.52,
          metalness: 0.2,
        })
      );
      silo.position.set(x + (i - (count - 1) / 2) * 3.2, silo.geometry.parameters.height / 2, z);
      silo.castShadow = true;
      this.cityGroup.add(silo);
    }
  }

  _addLoadingBays(x, z, angle, count = 4) {
    for (let i = 0; i < count; i++) {
      const bay = new THREE.Mesh(
        new THREE.PlaneGeometry(2.8, 1.2),
        new THREE.MeshBasicMaterial({
          color: 0xd6dde3,
          transparent: true,
          opacity: 0.34,
          side: THREE.DoubleSide,
        })
      );
      bay.rotation.x = -Math.PI / 2;
      bay.rotation.z = angle;
      bay.position.set(
        x + Math.cos(angle + Math.PI / 2) * (i - (count - 1) / 2) * 3.4,
        0.02,
        z + Math.sin(angle + Math.PI / 2) * (i - (count - 1) / 2) * 3.4
      );
      this.cityGroup.add(bay);
    }
  }

  _addParkedCar(x, z, angle = 0) {
    const car = new THREE.Mesh(
      new THREE.BoxGeometry(2.4, 0.7, 1.2),
      new THREE.MeshStandardMaterial({
        color: Math.random() > 0.5 ? 0x8aa1b5 : 0x5c6b76,
        roughness: 0.54,
        metalness: 0.14,
      })
    );
    car.position.set(x, 0.38, z);
    car.rotation.y = angle;
    car.castShadow = true;
    this.cityGroup.add(car);
  }

  _addBench(x, z, angle = 0) {
    const bench = new THREE.Group();
    const seat = new THREE.Mesh(
      new THREE.BoxGeometry(1.8, 0.12, 0.42),
      new THREE.MeshStandardMaterial({ color: 0x8e6b4b, roughness: 0.86 })
    );
    seat.position.y = 0.5;
    bench.add(seat);

    const legs = [
      [-0.7, 0.22, 0],
      [0.7, 0.22, 0],
    ];
    for (const [lx, ly, lz] of legs) {
      const leg = new THREE.Mesh(
        new THREE.BoxGeometry(0.1, 0.44, 0.1),
        new THREE.MeshStandardMaterial({ color: 0x555c63, roughness: 0.7 })
      );
      leg.position.set(lx, ly, lz);
      bench.add(leg);
    }

    bench.position.set(x, 0, z);
    bench.rotation.y = angle;
    this.cityGroup.add(bench);
  }

  _addCanopy(x, z, angle, w, d, color) {
    const postOffsets = [
      [-w / 2 + 0.18, 1.25, -d / 2 + 0.18],
      [w / 2 - 0.18, 1.25, -d / 2 + 0.18],
      [-w / 2 + 0.18, 1.25, d / 2 - 0.18],
      [w / 2 - 0.18, 1.25, d / 2 - 0.18],
    ];
    const canopy = new THREE.Group();
    const roof = new THREE.Mesh(
      new THREE.BoxGeometry(w, 0.18, d),
      new THREE.MeshStandardMaterial({
        color,
        emissive: 0x12304a,
        emissiveIntensity: 0.18,
        roughness: 0.54,
        metalness: 0.12,
      })
    );
    roof.position.y = 2.5;
    canopy.add(roof);

    for (const [px, py, pz] of postOffsets) {
      const post = new THREE.Mesh(
        new THREE.CylinderGeometry(0.07, 0.07, 2.5, 8),
        new THREE.MeshStandardMaterial({ color: 0x9099a2, roughness: 0.6, metalness: 0.22 })
      );
      post.position.set(px, py, pz);
      canopy.add(post);
    }

    canopy.position.set(x, 0, z);
    canopy.rotation.y = angle;
    this.cityGroup.add(canopy);
  }

  _addLightPylon(x, z, height, color, withLight) {
    const pylon = new THREE.Mesh(
      new THREE.BoxGeometry(0.42, height, 0.42),
      new THREE.MeshStandardMaterial({
        color: 0x9ca6b1,
        emissive: color,
        emissiveIntensity: 0.22,
        roughness: 0.34,
        metalness: 0.28,
      })
    );
    pylon.position.set(x, height / 2, z);
    pylon.castShadow = true;
    this.cityGroup.add(pylon);

    const beacon = new THREE.Mesh(
      new THREE.SphereGeometry(0.16, 10, 10),
      new THREE.MeshStandardMaterial({
        color,
        emissive: color,
        emissiveIntensity: 1.2,
        roughness: 0.14,
      })
    );
    beacon.position.set(x, height + 0.25, z);
    this.cityGroup.add(beacon);

    if (withLight && this.lightBudget > 0) {
      const light = new THREE.PointLight(color, 0.9, 44, 2.0);
      light.position.set(x, height + 0.2, z);
      this.cityGroup.add(light);
      this.lightBudget -= 1;
    }
  }

  _addSignTotem(x, z, angle) {
    const totem = new THREE.Group();
    const body = new THREE.Mesh(
      new THREE.BoxGeometry(1.1, 4.8, 0.38),
      new THREE.MeshPhysicalMaterial({
        color: 0x6d85a0,
        emissive: 0x4ea5ff,
        emissiveIntensity: 0.26,
        roughness: 0.34,
        metalness: 0.22,
      })
    );
    body.position.y = 2.4;
    totem.add(body);

    const panel = new THREE.Mesh(
      new THREE.BoxGeometry(0.72, 2.8, 0.08),
      new THREE.MeshBasicMaterial({
        color: 0xb4e6ff,
        transparent: true,
        opacity: 0.46,
      })
    );
    panel.position.set(0, 2.5, 0.24);
    totem.add(panel);

    totem.position.set(x, 0, z);
    totem.rotation.y = angle;
    this.cityGroup.add(totem);
  }

  _createRoadMarkings(length, width, roadClass) {
    const group = new THREE.Group();
    const dashLength = roadClass === 'arterial' ? 10 : roadClass === 'collector' ? 7 : 5;
    const laneCount = width >= 16 ? 2 : 1;
    const spacing = laneCount > 1 ? width / 4 : 0;

    for (let lane = 0; lane < laneCount; lane++) {
      const zOffset = laneCount > 1 ? (lane === 0 ? -spacing : spacing) : 0;
      const stripeCount = Math.max(3, Math.floor(length / (dashLength * 1.7)));
      for (let i = 0; i < stripeCount; i++) {
        const dash = new THREE.Mesh(
          new THREE.PlaneGeometry(dashLength, 0.36),
          new THREE.MeshBasicMaterial({
            color: 0xe7edf2,
            transparent: true,
            opacity: 0.3,
            side: THREE.DoubleSide,
            depthWrite: false,
          })
        );
        dash.rotation.x = -Math.PI / 2;
        dash.position.set(
          -length / 2 + (i + 0.55) * (length / stripeCount),
          0,
          zOffset
        );
        group.add(dash);
      }
    }

    const edgeStrip = new THREE.Mesh(
      new THREE.PlaneGeometry(length, 0.18),
      new THREE.MeshBasicMaterial({
        color: 0x8aa3b6,
        transparent: true,
        opacity: 0.18,
        side: THREE.DoubleSide,
        depthWrite: false,
      })
    );
    edgeStrip.rotation.x = -Math.PI / 2;
    edgeStrip.position.set(0, 0, width / 2 - 0.22);
    group.add(edgeStrip);

    const edgeStrip2 = edgeStrip.clone();
    edgeStrip2.position.z = -width / 2 + 0.22;
    group.add(edgeStrip2);

    return group;
  }

  _createBuildingMaterial(style, district, darkened = false) {
    const districtColor = DISTRICT_COLORS[district] || 0x666666;
    const baseColor = STYLE_COLORS[style] || 0xd8d0c4;
    const facadeColor = this._mixColors(baseColor, districtColor, darkened ? 0.34 : 0.16);
    const windowMap = this._getWindowTexture(style, district);
    const emissiveColor = district === 'cbd'
      ? 0x7dbdff
      : style === 'modern_glass'
        ? 0x79ccff
        : district === 'industrial'
          ? 0xffd6a0
          : 0xffd39b;
    const emissiveIntensity = district === 'cbd'
      ? 0.62
      : style === 'modern_glass'
        ? 0.42
        : district === 'industrial'
          ? 0.26
          : 0.2;

    return new THREE.MeshPhysicalMaterial({
      color: facadeColor,
      map: windowMap,
      emissiveMap: windowMap,
      roughness: style === 'modern_glass' ? 0.34 : 0.72,
      metalness: style === 'modern_glass' ? 0.36 : 0.08,
      clearcoat: style === 'modern_glass' ? 0.38 : 0.08,
      clearcoatRoughness: 0.74,
      reflectivity: 0.28,
      emissive: emissiveColor,
      emissiveIntensity: darkened ? emissiveIntensity * 0.72 : emissiveIntensity,
    });
  }

  _getZoneTexture(district) {
    if (this.zoneTextures.has(district)) return this.zoneTextures.get(district);

    const canvas = document.createElement('canvas');
    canvas.width = 256;
    canvas.height = 256;
    const ctx = canvas.getContext('2d');

    ctx.fillStyle = district === 'park' ? '#355d43' : district === 'plaza' ? '#70695e' : '#273646';
    ctx.fillRect(0, 0, 256, 256);

    if (district === 'park') {
      ctx.strokeStyle = 'rgba(166, 236, 192, 0.18)';
      for (let i = 0; i < 20; i++) {
        ctx.beginPath();
        ctx.arc(Math.random() * 256, Math.random() * 256, 8 + Math.random() * 18, 0, Math.PI * 2);
        ctx.stroke();
      }
    } else if (district === 'plaza') {
      ctx.strokeStyle = 'rgba(230, 241, 250, 0.16)';
      ctx.lineWidth = 2;
      for (let i = 0; i < 256; i += 32) {
        ctx.beginPath();
        ctx.moveTo(i, 0);
        ctx.lineTo(i, 256);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(0, i);
        ctx.lineTo(256, i);
        ctx.stroke();
      }
    } else {
      ctx.strokeStyle = 'rgba(144, 204, 255, 0.08)';
      ctx.lineWidth = 1;
      for (let i = 0; i < 256; i += 24) {
        ctx.beginPath();
        ctx.moveTo(i, 0);
        ctx.lineTo(i, 256);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(0, i);
        ctx.lineTo(256, i);
        ctx.stroke();
      }
    }

    const texture = new THREE.CanvasTexture(canvas);
    texture.wrapS = THREE.RepeatWrapping;
    texture.wrapT = THREE.RepeatWrapping;
    texture.repeat.set(2, 2);
    texture.colorSpace = THREE.SRGBColorSpace;
    this.zoneTextures.set(district, texture);
    return texture;
  }

  _getWindowTexture(style, district) {
    const key = `${style || 'midrise'}:${district || 'mixed'}`;
    if (this.windowTextures.has(key)) return this.windowTextures.get(key);

    const canvas = document.createElement('canvas');
    canvas.width = 256;
    canvas.height = 512;
    const ctx = canvas.getContext('2d');

    const base = this._mixColors(STYLE_COLORS[style] || 0xd0c7bc, DISTRICT_COLORS[district] || 0x44515d, 0.18);
    ctx.fillStyle = `#${base.getHexString()}`;
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const cols = style === 'modern_glass' || district === 'cbd' ? 6 : district === 'industrial' ? 8 : 7;
    const rows = district === 'cbd' ? 24 : district === 'industrial' ? 16 : 18;
    const padX = 10;
    const padY = 12;
    const cellW = (canvas.width - padX * 2) / cols;
    const cellH = (canvas.height - padY * 2) / rows;

    for (let y = 0; y < rows; y++) {
      for (let x = 0; x < cols; x++) {
        const lit = Math.random() > (district === 'cbd' ? 0.08 : district === 'industrial' ? 0.2 : 0.34);
        const alpha = lit ? 0.24 + Math.random() * 0.22 : 0.04 + Math.random() * 0.04;
        const color = style === 'modern_glass' || district === 'cbd'
          ? `rgba(150, 214, 255, ${alpha})`
          : district === 'industrial'
            ? `rgba(255, 207, 140, ${alpha})`
            : `rgba(255, 226, 184, ${alpha})`;
        ctx.fillStyle = color;
        ctx.fillRect(
          padX + x * cellW + cellW * 0.18,
          padY + y * cellH + cellH * 0.15,
          cellW * 0.64,
          cellH * 0.56
        );
      }
    }

    ctx.fillStyle = district === 'cbd' ? 'rgba(255,255,255,0.06)' : 'rgba(255,255,255,0.04)';
    for (let i = 0; i < 18; i++) {
      ctx.fillRect(0, (i / 18) * canvas.height, canvas.width, 1);
    }

    const texture = new THREE.CanvasTexture(canvas);
    texture.wrapS = THREE.RepeatWrapping;
    texture.wrapT = THREE.RepeatWrapping;
    texture.repeat.set(1, 1);
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.anisotropy = 8;
    this.windowTextures.set(key, texture);
    return texture;
  }

  _polygonLineGeometry(polygon, y) {
    const points = polygon.map(([x, z]) => new THREE.Vector3(x, y, z));
    return new THREE.BufferGeometry().setFromPoints(points);
  }

  _getBlockAnchors(polygon) {
    if (!polygon || polygon.length < 4) return null;
    const center = this._polygonCentroid(polygon);
    const edgeCenters = [];
    const innerCorners = [];
    const bounds = { minX: Infinity, maxX: -Infinity, minZ: Infinity, maxZ: -Infinity };

    for (let i = 0; i < polygon.length; i++) {
      const a = polygon[i];
      const b = polygon[(i + 1) % polygon.length];
      bounds.minX = Math.min(bounds.minX, a[0]);
      bounds.maxX = Math.max(bounds.maxX, a[0]);
      bounds.minZ = Math.min(bounds.minZ, a[1]);
      bounds.maxZ = Math.max(bounds.maxZ, a[1]);

      const midX = (a[0] + b[0]) / 2;
      const midZ = (a[1] + b[1]) / 2;
      edgeCenters.push({
        x: center.x + (midX - center.x) * 0.74,
        z: center.z + (midZ - center.z) * 0.74,
        angle: Math.atan2(b[1] - a[1], b[0] - a[0]),
      });

      innerCorners.push({
        x: center.x + (a[0] - center.x) * 0.7,
        z: center.z + (a[1] - center.z) * 0.7,
      });
    }

    return {
      center,
      edgeCenters,
      innerCorners,
      innerWidth: Math.max(8, (bounds.maxX - bounds.minX) * 0.72),
      innerDepth: Math.max(8, (bounds.maxZ - bounds.minZ) * 0.72),
    };
  }

  _mixColors(a, b, t = 0.15) {
    const c1 = new THREE.Color(a);
    const c2 = new THREE.Color(b);
    c1.lerp(c2, t);
    return c1;
  }

  _polygonCentroid(polygon) {
    let sx = 0;
    let sz = 0;
    for (const [x, z] of polygon) {
      sx += x;
      sz += z;
    }
    return { x: sx / polygon.length, z: sz / polygon.length };
  }

  _generatePlaceholder() {
    const colors = [0xcfd8e2, 0xc3bfb5, 0xe2d7c8, 0x8ca9c6, 0xa1b1a2];

    for (let r = 900; r >= 260; r -= 180) {
      const ring = new THREE.Mesh(
        new THREE.RingGeometry(r - 60, r, 64),
        new THREE.MeshBasicMaterial({
          color: r > 700 ? 0x0f2438 : r > 500 ? 0x22415b : 0x2e4d40,
          transparent: true,
          opacity: 0.12,
          side: THREE.DoubleSide,
        })
      );
      ring.rotation.x = -Math.PI / 2;
      ring.position.y = -0.05;
      this.cityGroup.add(ring);
    }

    const gridSize = 18;
    const spacing = 52;
    for (let ix = -gridSize / 2; ix < gridSize / 2; ix++) {
      for (let iz = -gridSize / 2; iz < gridSize / 2; iz++) {
        if (Math.random() > 0.86) continue;
        const x = ix * spacing + (Math.random() - 0.5) * 14;
        const z = iz * spacing + (Math.random() - 0.5) * 14;
        const h = 10 + Math.random() * 92;
        const w = 10 + Math.random() * 32;
        const d = 10 + Math.random() * 28;
        const color = colors[Math.floor(Math.random() * colors.length)];

        const body = new THREE.Mesh(
          new THREE.BoxGeometry(w, h, d),
          new THREE.MeshPhysicalMaterial({
            color,
            roughness: 0.62,
            metalness: 0.12,
            clearcoat: 0.08,
          })
        );
        body.position.set(x, h / 2, z);
        body.castShadow = true;
        body.receiveShadow = true;
        this.cityGroup.add(body);
      }
    }
    console.log('[City] Generated placeholder city');
  }

  clear() {
    while (this.cityGroup.children.length) {
      const child = this.cityGroup.children[0];
      if (child.geometry) child.geometry.dispose();
      if (Array.isArray(child.material)) child.material.forEach(m => m.dispose());
      else if (child.material) child.material.dispose();
      this.cityGroup.remove(child);
    }
  }
}
