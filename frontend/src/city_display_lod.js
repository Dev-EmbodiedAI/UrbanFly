import * as THREE from 'three';

// Common actors/lights stay on layer 0. These two bits are city-specific.
export const SENSOR_CITY_LAYER = 1;
export const DISPLAY_CITY_LAYER = 2;

function renderables(root) {
  const result = [];
  root?.traverse((object) => { if (object.isMesh) result.push(object); });
  return result;
}

export class CityDisplayLod {
  constructor({ nearDistance = 200, hysteresis = 0.2 } = {}) {
    this.nearDistance = nearDistance;
    this.farDistance = nearDistance * (1 + hysteresis);
    this.tiles = new Map();
    this.enabled = true;
  }

  addOverview(name, root) {
    const tile = this.tiles.get(name) || { high: [], overview: [], selectedHigh: false };
    tile.overview = renderables(root);
    tile.appliedHigh = null;
    for (const mesh of tile.overview) mesh.layers.set(DISPLAY_CITY_LAYER);
    this.tiles.set(name, tile);
  }

  addHigh(name, root, bounds) {
    const tile = this.tiles.get(name) || { high: [], overview: [], selectedHigh: false };
    tile.high = renderables(root);
    tile.appliedHigh = null;
    tile.bounds = new THREE.Box3(
      new THREE.Vector3().fromArray(bounds.minimum),
      new THREE.Vector3().fromArray(bounds.maximum),
    );
    // This bit is never toggled by presentation selection, even off-screen.
    for (const mesh of tile.high) mesh.layers.set(SENSOR_CITY_LAYER);
    this.tiles.set(name, tile);
  }

  update(cameraPosition, { fullDetail = false } = {}) {
    for (const tile of this.tiles.values()) {
      const distance = tile.bounds?.distanceToPoint(cameraPosition) ?? Infinity;
      const high = tile.high.length > 0 && (
        fullDetail || !this.enabled || tile.overview.length === 0
        || distance <= (tile.selectedHigh ? this.farDistance : this.nearDistance)
      );
      tile.selectedHigh = high;
      if (tile.appliedHigh === high) continue;
      tile.appliedHigh = high;
      for (const mesh of tile.high) {
        if (high) mesh.layers.enable(DISPLAY_CITY_LAYER);
        else mesh.layers.disable(DISPLAY_CITY_LAYER);
      }
      for (const mesh of tile.overview) {
        if (high) mesh.layers.disable(DISPLAY_CITY_LAYER);
        else mesh.layers.enable(DISPLAY_CITY_LAYER);
      }
    }
  }

  statistics() {
    const tiles = [...this.tiles.values()];
    return {
      display_lod_enabled: this.enabled,
      display_high_tiles: tiles.filter((tile) => tile.selectedHigh).length,
      display_overview_tiles: tiles.filter((tile) => !tile.selectedHigh && tile.overview.length).length,
      sensor_high_tiles: tiles.filter((tile) => tile.high.length).length,
    };
  }
}
