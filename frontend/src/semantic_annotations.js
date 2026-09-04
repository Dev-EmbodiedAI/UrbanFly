import * as THREE from 'three';

export const BUSINESS_NODE_TYPES = Object.freeze({
  supply: { label: '供货点', limit: 100, color: 0xf4b942 },
  delivery: { label: '送货点', limit: 100, color: 0x39a9db },
  resupply: { label: '补给点', limit: 20, color: 0x48b779 },
  drone_origin: { label: '无人机原始位点', limit: 20, color: 0xe76f51 },
});

const NODE_SCHEMA = 'urbanfly-business-nodes-v1';

function finitePosition(position) {
  return Array.isArray(position) && position.length === 3
    && position.every((value) => Number.isFinite(Number(value)))
    ? position.map(Number) : null;
}

export class BusinessNodeStore {
  constructor(nodes = []) {
    this.nodes = new Map();
    this.history = [];
    this.selectedId = null;
    this.replace(nodes);
  }

  replace(nodes) {
    if (!Array.isArray(nodes)) throw new Error('节点列表必须是数组');
    const next = new Map();
    const counts = Object.fromEntries(Object.keys(BUSINESS_NODE_TYPES).map((type) => [type, 0]));
    for (const raw of nodes) {
      const node = this._normalize(raw);
      if (next.has(node.id)) throw new Error(`节点编号重复：${node.id}`);
      counts[node.type] += 1;
      if (counts[node.type] > BUSINESS_NODE_TYPES[node.type].limit) {
        throw new Error(`${BUSINESS_NODE_TYPES[node.type].label}已超过 ${BUSINESS_NODE_TYPES[node.type].limit} 个上限`);
      }
      next.set(node.id, node);
    }
    this.nodes = next;
    this.selectedId = this.nodes.has(this.selectedId) ? this.selectedId : null;
    this.history = [];
    return this.list();
  }

  add(type, position) {
    this._assertType(type);
    const coordinates = finitePosition(position);
    if (!coordinates) throw new Error('点击位置不是有效的世界坐标');
    if (this.count(type) >= BUSINESS_NODE_TYPES[type].limit) {
      throw new Error(`${BUSINESS_NODE_TYPES[type].label}已达到 ${BUSINESS_NODE_TYPES[type].limit} 个上限`);
    }
    const node = {
      id: this._nextId(type),
      type,
      position: coordinates,
      qa_status: 'UNCHECKED',
    };
    this.nodes.set(node.id, node);
    this.selectedId = node.id;
    this.history.push({ action: 'add', node: { ...node, position: [...node.position] } });
    return node;
  }

  update(id, patch) {
    const current = this.nodes.get(id);
    if (!current) throw new Error(`未找到节点：${id}`);
    const next = { ...current };
    if (patch.type !== undefined) {
      this._assertType(patch.type);
      next.type = patch.type;
    }
    if (patch.position !== undefined) {
      next.position = finitePosition(patch.position);
      if (!next.position) throw new Error('节点位置不是有效的世界坐标');
    }
    if (patch.qa_status !== undefined) next.qa_status = String(patch.qa_status);
    if (patch.type !== undefined && patch.type !== current.type
      && this.count(patch.type) >= BUSINESS_NODE_TYPES[patch.type].limit) {
      throw new Error(`${BUSINESS_NODE_TYPES[patch.type].label}已达到上限`);
    }
    if (patch.type !== undefined && patch.type !== current.type) {
      next.id = this._nextId(patch.type);
      this.nodes.delete(id);
    }
    this.nodes.set(next.id, next);
    this.selectedId = next.id;
    this.history.push({ action: 'update', before: { ...current, position: [...current.position] }, after: { ...next, position: [...next.position] } });
    return next;
  }

  remove(id) {
    const node = this.nodes.get(id);
    if (!node) return null;
    this.nodes.delete(id);
    this.selectedId = null;
    this.history.push({ action: 'remove', node: { ...node, position: [...node.position] } });
    return node;
  }

  undo() {
    const operation = this.history.pop();
    if (!operation) return null;
    if (operation.action === 'add') this.nodes.delete(operation.node.id);
    if (operation.action === 'remove') this.nodes.set(operation.node.id, operation.node);
    if (operation.action === 'update') {
      this.nodes.delete(operation.after.id);
      this.nodes.set(operation.before.id, operation.before);
    }
    this.selectedId = null;
    return operation;
  }

  select(id) {
    this.selectedId = this.nodes.has(id) ? id : null;
    return this.selectedId ? this.nodes.get(this.selectedId) : null;
  }

  count(type) {
    return [...this.nodes.values()].filter((node) => node.type === type).length;
  }

  list() {
    return [...this.nodes.values()];
  }

  document() {
    return {
      schema: NODE_SCHEMA,
      scene: 'HelsinkiCentral1km',
      coordinate_frame: 'world_enu',
      axis_order: ['east', 'up', 'north'],
      nodes: this.list().map((node) => ({ ...node, position: [...node.position] })),
    };
  }

  _assertType(type) {
    if (!Object.hasOwn(BUSINESS_NODE_TYPES, type)) throw new Error(`不支持的节点类别：${type}`);
  }

  _normalize(raw) {
    if (!raw || typeof raw !== 'object') throw new Error('节点数据格式错误');
    this._assertType(raw.type);
    const position = finitePosition(raw.position);
    if (!raw.id || !position) throw new Error('节点编号或坐标无效');
    return {
      id: String(raw.id), type: raw.type, position, qa_status: String(raw.qa_status || 'UNCHECKED'),
    };
  }

  _nextId(type) {
    const prefix = BUSINESS_NODE_TYPES[type].label;
    let number = 1;
    while (this.nodes.has(`${prefix}_${String(number).padStart(3, '0')}`)) number += 1;
    return `${prefix}_${String(number).padStart(3, '0')}`;
  }
}

function makeLabelTexture(text, color) {
  const canvas = document.createElement('canvas');
  canvas.width = 512;
  canvas.height = 112;
  const context = canvas.getContext('2d');
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.fillStyle = '#07131a';
  context.globalAlpha = 0.88;
  context.roundRect(4, 4, canvas.width - 8, canvas.height - 8, 16);
  context.fill();
  context.globalAlpha = 1;
  context.fillStyle = `#${new THREE.Color(color).getHexString()}`;
  context.font = 'bold 42px Microsoft YaHei, sans-serif';
  context.textBaseline = 'middle';
  context.fillText(text, 22, canvas.height / 2);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.needsUpdate = true;
  return texture;
}

export class BusinessNodeRenderer {
  constructor(scene) {
    this.group = new THREE.Group();
    this.group.name = 'BusinessSemanticNodes';
    this.group.userData.excludeFromSensors = true;
    scene.add(this.group);
    this.objects = new Map();
  }

  sync(nodes) {
    const wanted = new Map(nodes.map((node) => [node.id, node]));
    for (const [id] of this.objects) if (!wanted.has(id)) this._removeObject(id);
    for (const node of nodes) {
      const existing = this.objects.get(node.id);
      if (existing) this._updateObject(existing, node);
      else this._addObject(node);
    }
  }

  setSelected(id) {
    for (const [nodeId, entry] of this.objects) {
      entry.selection.visible = nodeId === id;
    }
  }

  _addObject(node) {
    const config = BUSINESS_NODE_TYPES[node.type];
    const root = new THREE.Group();
    root.name = `BusinessNode_${node.id}`;
    root.userData.nodeId = node.id;
    root.position.fromArray(node.position);
    const material = new THREE.MeshBasicMaterial({ color: config.color, toneMapped: false });
    let geometry = new THREE.SphereGeometry(1.15, 16, 10);
    if (node.type === 'supply') geometry = new THREE.BoxGeometry(1.8, 1.8, 1.8);
    if (node.type === 'delivery') geometry = new THREE.ConeGeometry(1.35, 2.6, 6);
    if (node.type === 'resupply') geometry = new THREE.CylinderGeometry(1.25, 1.25, 2.2, 12);
    if (node.type === 'drone_origin') geometry = new THREE.OctahedronGeometry(1.5);
    const marker = new THREE.Mesh(geometry, material);
    marker.position.y = 1.35;
    marker.userData.nodeId = node.id;
    root.add(marker);
    const ring = new THREE.Mesh(
      new THREE.RingGeometry(1.9, 2.2, 24),
      new THREE.MeshBasicMaterial({ color: config.color, transparent: true, opacity: 0.85, side: THREE.DoubleSide, depthWrite: false, toneMapped: false }),
    );
    ring.rotation.x = -Math.PI / 2;
    ring.position.y = 0.08;
    root.add(ring);
    const label = new THREE.Sprite(new THREE.SpriteMaterial({ map: makeLabelTexture(node.id, config.color), transparent: true, depthTest: false, toneMapped: false }));
    label.scale.set(8, 1.75, 1);
    label.position.set(0, 4.1, 0);
    root.add(label);
    const selection = new THREE.Mesh(
      new THREE.RingGeometry(2.5, 2.8, 32),
      new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.9, side: THREE.DoubleSide, depthWrite: false, toneMapped: false }),
    );
    selection.rotation.x = -Math.PI / 2;
    selection.position.y = 0.12;
    selection.visible = false;
    root.add(selection);
    this.group.add(root);
    this.objects.set(node.id, { root, marker, selection, label });
  }

  _updateObject(entry, node) {
    entry.root.position.fromArray(node.position);
    entry.root.userData.nodeId = node.id;
  }

  _removeObject(id) {
    const entry = this.objects.get(id);
    if (!entry) return;
    entry.root.traverse((child) => {
      child.geometry?.dispose();
      if (child.material?.map) child.material.map.dispose();
      child.material?.dispose();
    });
    this.group.remove(entry.root);
    this.objects.delete(id);
  }
}

export class BusinessNodeAnnotationController {
  constructor({ sceneManager, digitalTwin }) {
    this.sceneManager = sceneManager;
    this.digitalTwin = digitalTwin;
    this.store = new BusinessNodeStore();
    this.renderer = new BusinessNodeRenderer(sceneManager.scene);
    this.mode = null;
    this.pointerDown = null;
    this.messageEl = document.getElementById('annotation-message');
    this.typeSelect = document.getElementById('annotation-type');
    this._bindUi();
    this._bindCanvas();
    this._render();
    this.load().catch(() => this._setMessage('暂无已保存标注，当前从空白标注集开始'));
  }

  _bindUi() {
    document.getElementById('annotation-enable')?.addEventListener('click', () => {
      this._setMode(this.mode === 'add' ? null : 'add');
    });
    document.getElementById('annotation-reposition')?.addEventListener('click', () => {
      if (!this.store.selectedId) return this._setMessage('请先选择一个节点');
      this._setMode(this.mode === 'reposition' ? null : 'reposition');
    });
    document.getElementById('annotation-delete')?.addEventListener('click', () => {
      if (!this.store.selectedId) return this._setMessage('请先选择一个节点');
      this.store.remove(this.store.selectedId);
      this.mode = null;
      this._render();
    });
    document.getElementById('annotation-undo')?.addEventListener('click', () => {
      if (!this.store.undo()) return this._setMessage('没有可撤销的操作');
      this._render();
    });
    document.getElementById('annotation-save')?.addEventListener('click', () => this.save());
    document.getElementById('annotation-load')?.addEventListener('click', () => this.load());
    document.getElementById('annotation-export')?.addEventListener('click', () => this.exportJson());
    document.getElementById('annotation-import')?.addEventListener('change', (event) => {
      const [file] = event.target.files || [];
      if (file) this.importJson(file);
      event.target.value = '';
    });
    this.typeSelect?.addEventListener('change', (event) => {
      if (!this.store.selectedId) return this._render();
      try {
        this.store.update(this.store.selectedId, { type: event.target.value });
        this._render();
      } catch (error) { this._setMessage(error.message); }
    });
    document.getElementById('annotation-list')?.addEventListener('click', (event) => {
      const row = event.target.closest('[data-node-id]');
      if (!row) return;
      this.store.select(row.dataset.nodeId);
      this.mode = null;
      this._render();
    });
  }

  _bindCanvas() {
    const canvas = this.sceneManager.canvas;
    canvas.addEventListener('pointerdown', (event) => {
      this.pointerDown = { x: event.clientX, y: event.clientY };
    });
    canvas.addEventListener('pointerup', (event) => {
      if (!this.mode || !this.pointerDown) return;
      const distance = Math.hypot(event.clientX - this.pointerDown.x, event.clientY - this.pointerDown.y);
      this.pointerDown = null;
      if (distance > 5) return;
      this._place(event.clientX, event.clientY);
    });
  }

  _setMode(mode) {
    this.mode = mode;
    this._render();
  }

  _place(clientX, clientY) {
    const rect = this.sceneManager.canvas.getBoundingClientRect();
    const ndc = new THREE.Vector2(
      ((clientX - rect.left) / rect.width) * 2 - 1,
      -((clientY - rect.top) / rect.height) * 2 + 1,
    );
    const ray = new THREE.Raycaster();
    ray.setFromCamera(ndc, this.sceneManager.camera);
    const hit = this.digitalTwin.raycastAnnotationSurface(ray.ray.origin, ray.ray.direction);
    if (!hit?.point) return this._setMessage('未命中城市 Mesh，请点击可见建筑或地表');
    try {
      if (this.mode === 'reposition') {
        this.store.update(this.store.selectedId, { position: hit.point.toArray() });
      } else {
        this.store.add(this.typeSelect?.value || 'supply', hit.point.toArray());
      }
      this.mode = null;
      this._render();
    } catch (error) { this._setMessage(error.message); }
  }

  async save() {
    try {
      const response = await fetch('/api/semantic-nodes', {
        method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(this.store.document()),
      });
      if (!response.ok) throw new Error(`保存失败（${response.status}）`);
      await response.json();
      this._setMessage('已保存到后端统一标注文件');
    } catch (error) {
      this._setMessage(`后端不可用：${error.message}；可使用导出 JSON 保留结果`);
    }
  }

  async load() {
    const response = await fetch('/api/semantic-nodes');
    if (!response.ok) throw new Error(`加载失败（${response.status}）`);
    const document = await response.json();
    if (document.schema !== NODE_SCHEMA) throw new Error('标注文件版本不匹配');
    this.store.replace(document.nodes || []);
    this._render();
    this._setMessage(`已加载 ${this.store.nodes.size} 个城市业务节点`);
  }

  exportJson() {
    const blob = new Blob([`${JSON.stringify(this.store.document(), null, 2)}\n`], { type: 'application/json' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = 'helsinki_business_nodes.json';
    link.click();
    URL.revokeObjectURL(link.href);
    this._setMessage('已导出城市业务节点 JSON');
  }

  async importJson(file) {
    try {
      const document = JSON.parse(await file.text());
      if (document.schema !== NODE_SCHEMA) throw new Error('标注文件版本不匹配');
      this.store.replace(document.nodes || []);
      this._render();
      this._setMessage('已导入标注，点击保存写入后端');
    } catch (error) { this._setMessage(`导入失败：${error.message}`); }
  }

  _render() {
    const list = document.getElementById('annotation-list');
    if (!list) return;
    this.renderer.sync(this.store.list());
    this.renderer.setSelected(this.store.selectedId);
    for (const [type, config] of Object.entries(BUSINESS_NODE_TYPES)) {
      const count = document.getElementById(`annotation-count-${type}`);
      if (count) count.textContent = `${this.store.count(type)} / ${config.limit}`;
    }
    const selected = this.store.selectedId ? this.store.nodes.get(this.store.selectedId) : null;
    if (this.typeSelect) this.typeSelect.value = selected?.type || this.typeSelect.value || 'supply';
    list.replaceChildren();
    for (const node of this.store.list()) {
      const row = document.createElement('button');
      row.type = 'button';
      row.dataset.nodeId = node.id;
      row.className = `annotation-row${node.id === this.store.selectedId ? ' selected' : ''}`;
      row.innerHTML = `<span class="annotation-swatch" style="background:#${new THREE.Color(BUSINESS_NODE_TYPES[node.type].color).getHexString()}"></span><strong>${node.id}</strong><small>${BUSINESS_NODE_TYPES[node.type].label} · ${node.position.map((value) => value.toFixed(2)).join(', ')}</small>`;
      list.appendChild(row);
    }
    const enable = document.getElementById('annotation-enable');
    const reposition = document.getElementById('annotation-reposition');
    if (enable) {
      enable.textContent = this.mode === 'add' ? '退出点击标注' : '点击场景新增';
      enable.classList.toggle('active', this.mode === 'add');
    }
    if (reposition) {
      reposition.textContent = this.mode === 'reposition' ? '取消重新定位' : '重新定位所选';
      reposition.disabled = !selected;
      reposition.classList.toggle('active', this.mode === 'reposition');
    }
    document.getElementById('annotation-delete').disabled = !selected;
    document.getElementById('annotation-undo').disabled = this.store.history.length === 0;
    document.getElementById('annotation-total').textContent = `${this.store.nodes.size} / 240`;
    document.getElementById('annotation-mode-status').textContent = this.mode === 'add'
      ? '点击 Mesh 放置新节点' : this.mode === 'reposition' ? '点击 Mesh 更新所选节点位置' : '浏览模式';
  }

  _setMessage(message) {
    if (this.messageEl) this.messageEl.textContent = message;
  }
}
