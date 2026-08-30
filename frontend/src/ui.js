/**
 * UrbanFly UI controller.
 * Keeps the interface deliberately compact so the real-city view remains primary.
 */
export class UIManager {
  constructor(state) {
    this.state = state;
    this.onSelectScenario = null;
    this.onSelectAlgorithm = null;
    this.onControl = null;

    this.scenarioSelect = document.getElementById('scenario-select');
    this.algorithmSelect = document.getElementById('algorithm-select');
    this.speedSelect = document.getElementById('speed-select');
    this.simTimeEl = document.getElementById('sim-time');
    this.connectionEl = document.getElementById('connection-status');
    this.twinStatusEl = document.getElementById('twin-status');
    this.loadingOverlay = document.getElementById('twin-loading');

    this._bindControls();
    this._bindTabs();
  }

  _bindControls() {
    this.scenarioSelect?.addEventListener('change', (event) => {
      this.onSelectScenario?.(event.target.value);
    });
    this.algorithmSelect?.addEventListener('change', (event) => {
      this.onSelectAlgorithm?.(event.target.value);
    });
    document.getElementById('btn-play')?.addEventListener('click', () => this.onControl?.('play'));
    document.getElementById('btn-pause')?.addEventListener('click', () => this.onControl?.('pause'));
    document.getElementById('btn-stop')?.addEventListener('click', () => this.onControl?.('stop'));
    this.speedSelect?.addEventListener('change', (event) => {
      this.onControl?.('set_speed', Number(event.target.value));
    });
  }

  _bindTabs() {
    document.querySelectorAll('.tab-btn').forEach((button) => {
      button.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn').forEach((item) => item.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach((item) => item.classList.remove('active'));
        button.classList.add('active');
        document.getElementById(`tab-${button.dataset.tab}`)?.classList.add('active');
      });
    });
  }

  populateScenarioList(scenarios) {
    if (!this.scenarioSelect) return;
    this.scenarioSelect.innerHTML = '<option value="">选择任务场景</option>';
    for (const scenario of scenarios) {
      const option = document.createElement('option');
      option.value = scenario.name;
      option.textContent = `${scenario.name} · ${scenario.num_drones || 30} 架 / ${scenario.num_tasks || 100} 单`;
      this.scenarioSelect.appendChild(option);
    }
  }

  populateAlgorithmList(algorithms) {
    if (!this.algorithmSelect) return;
    this.algorithmSelect.innerHTML = '';
    for (const algorithm of algorithms) {
      const option = document.createElement('option');
      option.value = algorithm.id;
      option.textContent = algorithm.name;
      this.algorithmSelect.appendChild(option);
    }
  }

  updateTime(timeSeconds) {
    if (!this.simTimeEl) return;
    const minutes = Math.floor(timeSeconds / 60);
    const seconds = Math.floor(timeSeconds % 60);
    this.simTimeEl.textContent = `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
  }

  setConnectionState(connected) {
    if (!this.connectionEl) return;
    this.connectionEl.classList.toggle('is-live', connected);
    const label = this.connectionEl.querySelector('span:last-child');
    if (label) label.textContent = connected ? '仿真后端已连接' : '本地演示模式';
  }

  setTwinStatus(status) {
    if (this.twinStatusEl) {
      this.twinStatusEl.textContent = status.phase === 'ready' ? '三角网格实景在线' : '正在载入实景';
      this.twinStatusEl.classList.toggle('is-ready', status.phase === 'ready');
    }
    if (!this.loadingOverlay) return;
    const title = this.loadingOverlay.querySelector('[data-loading-title]');
    const detail = this.loadingOverlay.querySelector('[data-loading-detail]');
    if (title) title.textContent = status.title;
    if (detail) detail.textContent = status.detail;
    if (status.phase === 'ready') {
      window.setTimeout(() => this.loadingOverlay.classList.add('is-hidden'), 650);
    } else if (status.phase === 'fallback') {
      this.loadingOverlay.classList.add('is-fallback');
      window.setTimeout(() => this.loadingOverlay.classList.add('is-hidden'), 1800);
    }
  }

  updateDroneList(drones = []) {
    const count = document.getElementById('drone-count');
    if (count) count.textContent = `${drones.length} 架在线`;
    const list = document.getElementById('drone-list');
    if (!list) return;

    const stateLabels = {
      idle: '空闲',
      takeoff: '起飞',
      en_route: '执行航段',
      picking_up: '取件',
      delivering: '递送',
      hovering: '悬停',
      returning: '返航',
      charging: '充电',
      landed: '降落',
      emergency: '紧急状态',
    };
    const colors = { heavy: '#ff8f70', standard: '#54c7ff', light: '#72e0ae' };
    list.innerHTML = drones.slice(0, 6).map((drone) => {
      const color = colors[drone.drone_type] || '#9ab0bf';
      const battery = Math.round((drone.battery || 0) * 100);
      const batteryColor = battery > 50 ? '#6ee4a6' : battery > 20 ? '#ffc86b' : '#ff7d6f';
      return `
        <article class="drone-item" style="--drone-color:${color}">
          <div class="drone-header">
            <strong>${drone.id}</strong>
            <span class="drone-type" style="color:${color}">${drone.drone_type}</span>
          </div>
          <div class="drone-info">
            <span>电量 <strong style="color:${batteryColor}">${battery}%</strong></span>
            <span>${(drone.speed || 0).toFixed(1)} m/s</span>
            ${drone.payload > 0.1 ? `<span>${drone.payload.toFixed(1)} kg</span>` : ''}
          </div>
          <div class="drone-state">${stateLabels[drone.state] || drone.state || '未知'} · ${drone.current_task || '待命'}</div>
          ${drone.dynamics_model ? `
            <div class="drone-dynamics">
              R ${Number(drone.roll || 0).toFixed(1)}° ·
              P ${Number(drone.pitch || 0).toFixed(1)}° ·
              ${Number(drone.power_w || 0).toFixed(0)} W
            </div>
          ` : ''}
          ${drone.world_model?.enabled ? `
            <div class="drone-world-model">
              ${drone.world_model.status === 'external_learned_policy'
                || drone.world_model.status === 'policy_timeout_hover'
                ? `${drone.world_model.backend} · #${drone.world_model.policy_step_id} · `
                  + `${drone.world_model.safety_intervened ? 'SHIELD' : 'RAW'} · `
                  + `${Number(drone.world_model.inference_latency_ms || 0).toFixed(0)} ms`
                : `WM H${Number(drone.world_model.horizon_s || 0).toFixed(1)}s · `
                  + `${Number(drone.world_model.safe_candidate_count || 0)}/`
                  + `${Number(drone.world_model.candidate_count || 0)} safe · `
                  + `${Number(drone.world_model.belief_cell_count || 0)} cells · `
                  + `C ${drone.world_model.minimum_predicted_clearance_m == null
                    ? '—'
                    : `${Number(drone.world_model.minimum_predicted_clearance_m).toFixed(1)}m`}`
              }
            </div>
          ` : ''}
        </article>
      `;
    }).join('');
  }

  updateTasks(tasks = []) {
    if (!document.getElementById('tab-tasks')?.classList.contains('active')) {
      return;
    }
    const summary = document.getElementById('task-summary');
    const list = document.getElementById('task-list');
    if (!list) return;
    const completed = tasks.filter((task) => task.status === 'completed').length;
    const inProgress = tasks.filter((task) => (
      ['assigned', 'en_route_pickup', 'en_route_delivery'].includes(task.status)
    )).length;
    const pending = tasks.filter((task) => task.status === 'pending').length;
    if (summary) {
      summary.innerHTML = `
        <div><strong>${inProgress}</strong><span>执行中</span></div>
        <div><strong>${pending}</strong><span>待分配</span></div>
        <div><strong>${completed}</strong><span>已完成</span></div>
      `;
    }

    const priorityLabels = {
      0: '紧急医疗',
      1: '医疗物资',
      2: '生鲜配送',
      3: '常规配送',
      4: '城市巡检',
    };
    const statusLabels = {
      pending: '待分配',
      assigned: '已分配',
      en_route_pickup: '前往取件',
      en_route_delivery: '配送中',
      completed: '已完成',
      failed: '失败',
    };
    list.innerHTML = tasks.slice(-8).reverse().map((task) => `
      <article class="task-item">
        <div>
          <span class="task-priority p${task.priority ?? 3}"></span>
          <strong>${task.id}</strong>
          <span>${priorityLabels[task.priority] || '配送任务'}</span>
        </div>
        <em>${statusLabels[task.status] || task.status}</em>
        <p>${task.business_tag || '城市低空配送'} · ${task.assigned_to || '待指派'}</p>
      </article>
    `).join('');
  }

  updateWorldModel(drones = []) {
    if (!document.getElementById('tab-world-model')?.classList.contains('active')) {
      return;
    }
    const drone = drones.find((item) => item.world_model?.enabled);
    const canvas = document.getElementById('world-model-canvas');
    const status = document.getElementById('world-model-status');
    const metrics = document.getElementById('world-model-metrics');
    if (!canvas || !status || !metrics) return;
    const context = canvas.getContext('2d');
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.fillStyle = 'rgba(3, 9, 14, 0.96)';
    context.fillRect(0, 0, canvas.width, canvas.height);

    if (!drone?.world_model) {
      status.textContent = '等待世界模型';
      metrics.replaceChildren();
      context.fillStyle = 'rgba(159, 193, 225, 0.56)';
      context.font = '11px "Segoe UI", sans-serif';
      context.fillText('选择 single_uav_world_model 场景', 54, 126);
      return;
    }

    const worldModel = drone.world_model;
    if (!worldModel.selected_trajectory_world_m?.length) {
      status.textContent = `${worldModel.backend} · #${worldModel.policy_step_id ?? '—'}`;
      const centerX = canvas.width / 2;
      const centerY = canvas.height / 2;
      context.strokeStyle = 'rgba(98, 214, 255, 0.35)';
      context.beginPath();
      context.moveTo(20, centerY);
      context.lineTo(canvas.width - 20, centerY);
      context.moveTo(centerX, 20);
      context.lineTo(centerX, canvas.height - 20);
      context.stroke();
      const raw = worldModel.raw_action_physical_body_flu || [0, 0, 0, 0];
      const executed = worldModel.command_world_mps || [0, 0, 0];
      const scale = 20;
      context.strokeStyle = '#ffcf66';
      context.lineWidth = 3;
      context.beginPath();
      context.moveTo(centerX, centerY);
      context.lineTo(centerX + Number(raw[1] || 0) * scale, centerY - Number(raw[0] || 0) * scale);
      context.stroke();
      context.strokeStyle = worldModel.safety_intervened ? '#ff6b6b' : '#6ee4a6';
      context.lineWidth = 4;
      context.beginPath();
      context.moveTo(centerX, centerY);
      context.lineTo(
        centerX + Number(executed[2] || 0) * scale,
        centerY - Number(executed[0] || 0) * scale,
      );
      context.stroke();
      context.fillStyle = 'rgba(219, 238, 252, 0.76)';
      context.font = '11px "Segoe UI", sans-serif';
      context.fillText('黄色：模型原始动作', 16, canvas.height - 28);
      context.fillText(
        worldModel.safety_intervened ? '红色：安全层执行动作' : '绿色：实际执行动作',
        16,
        canvas.height - 12,
      );
      metrics.innerHTML = `
        <div><dt>策略状态</dt><dd>${worldModel.status || 'unknown'}</dd></div>
        <div><dt>原始动作</dt><dd>${raw.map((value) => Number(value).toFixed(2)).join(', ')}</dd></div>
        <div><dt>执行速度</dt><dd>${executed.map((value) => Number(value).toFixed(2)).join(', ')} m/s</dd></div>
        <div><dt>安全干预</dt><dd>${worldModel.safety_intervened
          ? (worldModel.safety_intervention_reasons || []).join(', ')
          : '无'}</dd></div>
        <div><dt>累计干预</dt><dd>${worldModel.safety_intervention_count || 0}</dd></div>
        <div><dt>推理延迟</dt><dd>${Number(worldModel.inference_latency_ms || 0).toFixed(1)} ms</dd></div>
        <div><dt>预测风险</dt><dd>${Number(worldModel.predicted_risk || 0).toFixed(3)}</dd></div>
      `;
      return;
    }
    status.textContent = `#${worldModel.decision_sequence} · ${drone.id}`;
    const origin = drone.pos;
    const candidates = worldModel.top_candidates || [];
    const allTrajectories = [
      ...candidates.map((candidate) => candidate.trajectory_world_m || []),
      worldModel.selected_trajectory_world_m,
    ];
    const allPoints = allTrajectories
      .flat()
      .map((point) => [point[0] - origin[0], point[2] - origin[2]]);
    allPoints.push([0, 0]);
    const xs = allPoints.map((point) => point[0]);
    const zs = allPoints.map((point) => point[1]);
    const minX = Math.min(...xs, -4);
    const maxX = Math.max(...xs, 4);
    const minZ = Math.min(...zs, -4);
    const maxZ = Math.max(...zs, 4);
    const padding = 18;
    const scale = Math.min(
      (canvas.width - padding * 2) / Math.max(maxX - minX, 1),
      (canvas.height - padding * 2) / Math.max(maxZ - minZ, 1),
    );
    const project = (point) => [
      padding + (point[0] - origin[0] - minX) * scale,
      canvas.height - padding - (point[2] - origin[2] - minZ) * scale,
    ];

    context.strokeStyle = 'rgba(104, 151, 185, 0.12)';
    context.lineWidth = 1;
    for (let x = 20; x < canvas.width; x += 40) {
      context.beginPath();
      context.moveTo(x, 0);
      context.lineTo(x, canvas.height);
      context.stroke();
    }
    for (let y = 20; y < canvas.height; y += 40) {
      context.beginPath();
      context.moveTo(0, y);
      context.lineTo(canvas.width, y);
      context.stroke();
    }

    const drawTrajectory = (trajectory, color, width) => {
      if (!trajectory?.length) return;
      context.strokeStyle = color;
      context.lineWidth = width;
      context.beginPath();
      const start = project(origin);
      context.moveTo(start[0], start[1]);
      for (const point of trajectory) {
        const projected = project(point);
        context.lineTo(projected[0], projected[1]);
      }
      context.stroke();
    };
    for (const candidate of candidates.slice(1)) {
      drawTrajectory(
        candidate.trajectory_world_m,
        candidate.predicted_collision
          ? 'rgba(255, 107, 107, 0.48)'
          : 'rgba(157, 140, 255, 0.55)',
        1.25,
      );
    }
    drawTrajectory(
      worldModel.selected_trajectory_world_m,
      'rgba(255, 209, 102, 0.98)',
      3,
    );
    const start = project(origin);
    context.fillStyle = '#62d6ff';
    context.beginPath();
    context.arc(start[0], start[1], 4, 0, Math.PI * 2);
    context.fill();

    const safeCount = candidates.filter((candidate) => !candidate.predicted_collision).length;
    metrics.innerHTML = `
      <div><dt>候选轨迹</dt><dd>${safeCount} / ${worldModel.candidate_count || candidates.length} 低风险</dd></div>
      <div><dt>选中候选</dt><dd>#${worldModel.selected_index ?? '—'} · ${worldModel.selection_method || 'unknown'}</dd></div>
      <div><dt>规划延迟</dt><dd>${Number(worldModel.planner_latency_ms || worldModel.inference_latency_ms || 0).toFixed(1)} ms</dd></div>
      <div><dt>预测风险</dt><dd>${Number(worldModel.predicted_risk || 0).toFixed(3)}</dd></div>
      <div><dt>当前指令</dt><dd>${(worldModel.command_world_mps || [])
        .map((value) => Number(value).toFixed(1)).join(', ')} m/s</dd></div>
      <div><dt>透明安全层</dt><dd>${worldModel.safety_intervened
        ? (worldModel.safety_intervention_reasons || []).join(', ')
        : '未干预'}</dd></div>
    `;
  }

  updateCommGraph(commGraph, topologyStats) {
    if (!document.getElementById('tab-comm')?.classList.contains('active')) {
      return;
    }
    const canvas = document.getElementById('comm-canvas');
    const stats = document.getElementById('comm-stats');
    if (!canvas || !commGraph) return;
    const context = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    const centerX = width / 2;
    const centerY = height / 2;
    const radius = Math.min(width, height) * 0.34;
    const count = Math.min(commGraph.length, 30);
    const nodes = Array.from({ length: count }, (_, index) => {
      const angle = index / count * Math.PI * 2 - Math.PI / 2;
      return { x: centerX + Math.cos(angle) * radius, y: centerY + Math.sin(angle) * radius };
    });

    context.clearRect(0, 0, width, height);
    context.strokeStyle = 'rgba(84,199,255,.25)';
    for (let i = 0; i < count; i += 1) {
      for (let j = i + 1; j < count; j += 1) {
        if (!commGraph[i]?.[j]) continue;
        context.beginPath();
        context.moveTo(nodes[i].x, nodes[i].y);
        context.lineTo(nodes[j].x, nodes[j].y);
        context.stroke();
      }
    }
    context.fillStyle = '#72e0ae';
    for (const node of nodes) {
      context.beginPath();
      context.arc(node.x, node.y, 3.2, 0, Math.PI * 2);
      context.fill();
    }
    if (stats && topologyStats) {
      stats.textContent = `连通率 ${Math.round((topologyStats.connectivity_ratio || 0) * 100)}% · ${topologyStats.num_components || 1} 个网络分量`;
    }
  }

  updateStats(stats) {
    if (!document.getElementById('tab-stats')?.classList.contains('active')) {
      return;
    }
    const panel = document.getElementById('stats-panel');
    if (!panel || !stats) return;
    const rows = [
      ['任务完成率', `${Math.round((stats.on_time_rate || 0) * 100)}%`],
      ['飞行总里程', `${((stats.total_distance || 0) / 1000).toFixed(2)} km`],
      ['机队平均电量', `${Math.round((stats.avg_battery || 0) * 100)}%`],
      ['航迹重规划', `${stats.path_replans || 0} 次`],
      ['碰撞预警', `${stats.collision_warnings || 0} 次`],
      ['通信中断', `${stats.comm_disconnections || 0} 次`],
    ];
    panel.innerHTML = rows.map(([label, value]) => `
      <div class="stat-row"><span>${label}</span><strong>${value}</strong></div>
    `).join('');
  }

  updateAllocStats(allocStats) {
    if (!document.getElementById('tab-stats')?.classList.contains('active')) {
      return;
    }
    const panel = document.getElementById('stats-panel');
    if (!panel || !allocStats) return;
    const existing = panel.querySelector('.alloc-info');
    existing?.remove();
    const info = document.createElement('p');
    info.className = 'alloc-info';
    info.textContent = `${allocStats.algorithm} · ${allocStats.iterations} 次迭代 · ${allocStats.last_runtime_ms?.toFixed(1)} ms`;
    panel.appendChild(info);
  }

  updateSemanticAgent(snapshot) {
    const tab = document.getElementById('tab-semantic-agent');
    const status = document.getElementById('semantic-agent-status');
    const authority = document.getElementById('semantic-agent-authority');
    if (!tab || !status || !authority) return;
    const enabled = Boolean(snapshot?.enabled);
    status.textContent = enabled ? (snapshot.provider || 'provider unknown') : '未启用';
    authority.textContent = enabled
      ? String(snapshot.control_authority || 'semantic_only').toUpperCase()
      : 'NO CONTROL';
    authority.classList.toggle('is-ready', enabled);

    if (!tab.classList.contains('active')) return;
    const facts = document.getElementById('semantic-agent-facts');
    const eventCount = document.getElementById('semantic-event-count');
    const eventList = document.getElementById('semantic-event-list');
    const planCount = document.getElementById('semantic-plan-count');
    const assignmentList = document.getElementById('semantic-assignment-list');
    const events = snapshot?.active_events || [];
    const plan = snapshot?.last_plan || {};
    const assignments = plan.assignments || {};
    if (facts) {
      facts.innerHTML = `
        <div><dt>输入时间线</dt><dd>${snapshot?.timeline_consumed || 0} / ${snapshot?.timeline_total || 0}</dd></div>
        <div><dt>已应用规划</dt><dd>${snapshot?.applied_plan_count || 0}</dd></div>
        <div><dt>路径校验失败</dt><dd>${snapshot?.path_validation_failure_count || 0}</dd></div>
      `;
    }
    if (eventCount) eventCount.textContent = `${events.length} ACTIVE`;
    if (planCount) planCount.textContent = `${snapshot?.applied_plan_count || 0} PLAN`;
    if (eventList) {
      eventList.replaceChildren(...events.map((event) => {
        const item = document.createElement('article');
        item.className = `semantic-event semantic-event-${event.event_type || 'unknown'}`;
        const title = document.createElement('strong');
        title.textContent = String(event.event_type || 'unknown').replaceAll('_', ' ');
        const detail = document.createElement('span');
        detail.textContent = `${event.event_id} · ${(Number(event.confidence || 0) * 100).toFixed(0)}% · R${Number(event.radius_m || 0).toFixed(0)}m`;
        const evidence = document.createElement('p');
        evidence.textContent = event.evidence || '无证据摘要';
        item.append(title, detail, evidence);
        return item;
      }));
      if (!events.length) eventList.textContent = enabled ? '当前无活动语义事件' : '请选择 qwen_semantic_fleet 场景';
    }
    if (assignmentList) {
      const entries = Object.entries(assignments);
      assignmentList.replaceChildren(...entries.map(([droneId, taskIds]) => {
        const row = document.createElement('div');
        const drone = document.createElement('strong');
        drone.textContent = droneId;
        const tasks = document.createElement('span');
        tasks.textContent = taskIds?.length ? taskIds.join(', ') : '待命';
        row.append(drone, tasks);
        return row;
      }));
      if (!entries.length) assignmentList.textContent = enabled ? '等待首个确定性规划' : '—';
    }
  }

  addEvent(event) {
    const stream = document.getElementById('event-stream');
    if (!stream) return;
    const item = document.createElement('span');
    item.className = 'event-item';
    item.textContent = `${this._formatEventTime(event.time)}  ${event.message || event.type}`;
    stream.prepend(item);
    while (stream.children.length > 24) stream.lastElementChild.remove();
  }

  clearEvents() {
    const stream = document.getElementById('event-stream');
    if (stream) stream.replaceChildren();
  }

  showNotification(message) {
    const notification = document.createElement('div');
    notification.className = 'notification';
    notification.textContent = message;
    document.body.appendChild(notification);
    window.setTimeout(() => {
      notification.classList.add('is-leaving');
      window.setTimeout(() => notification.remove(), 360);
    }, 2600);
  }

  _formatEventTime(seconds = 0) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
  }
}
