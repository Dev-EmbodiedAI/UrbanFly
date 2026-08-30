import { presentationHidden } from './host_lifecycle.js';

const formatMs = (value) => (
  value !== null && value !== undefined && Number.isFinite(Number(value))
    ? `${Number(value).toFixed(1)} ms` : '—'
);

const formatBytes = (value) => {
  const bytes = Number(value || 0);
  if (bytes < 1024) return `${bytes.toFixed(0)} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
};

export class RuntimeHealth {
  constructor({ endpoint, network, sensors, presentation, sceneReady = () => false }) {
    this.endpoint = endpoint;
    this.network = network;
    this.sensors = sensors;
    this.presentation = presentation;
    this.sceneReady = sceneReady;
    this.surface = new URLSearchParams(window.location.search).get('surface') === 'desktop'
      ? 'desktop' : 'browser';
    this.refreshActive = false;
    this.abortController = null;
    this.onVisibilityChange = () => this.render();
    this.backend = null;
    this.backendError = null;
    this.lastHealthAt = null;
    this.longTasks = 0;
    this.longTaskMaxMs = 0;
    this.longTaskEntries = [];
    this.timer = null;
    this.elements = {
      root: document.getElementById('runtime-health'),
      backend: document.getElementById('health-backend'),
      sim: document.getElementById('health-sim'),
      sensor: document.getElementById('health-sensor'),
      network: document.getElementById('health-network'),
      flow: document.getElementById('health-flow'),
      page: document.getElementById('health-page'),
    };
    this._observeLongTasks();
  }

  start() {
    if (this.timer) return;
    this.refresh();
    this.timer = setInterval(() => this.refresh(), 2000);
    document.addEventListener('visibilitychange', this.onVisibilityChange);
  }

  stop() {
    clearInterval(this.timer);
    this.timer = null;
    this.abortController?.abort();
    this.observer?.disconnect();
    document.removeEventListener('visibilitychange', this.onVisibilityChange);
  }

  async refresh() {
    if (this.refreshActive) return;
    this.refreshActive = true;
    this.abortController = new AbortController();
    const timeout = setTimeout(() => this.abortController?.abort(), 1500);
    try {
      const response = await fetch(this.endpoint, {
        cache: 'no-store', signal: this.abortController.signal,
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      this.backend = await response.json();
      this.backendError = null;
      this.lastHealthAt = performance.now();
    } catch (error) {
      this.backendError = String(error);
    } finally {
      clearTimeout(timeout);
      this.abortController = null;
      this.refreshActive = false;
    }
    // Bounded, low-rate diagnostics only; never used as control/sensor state.
    if (this.network.connected) this.network.send('runtime_client_status', {
      surface: this.surface,
      scene_ready: this.sceneReady(),
      hidden: presentationHidden(),
      presentation: this.presentation?.statistics() || {},
      sensors: this.sensors.statistics(),
    });
    this.render();
  }

  render() {
    if (!this.elements.root) return;
    const network = this.network.statistics();
    const sensors = this.sensors.statistics();
    const metrics = this.backend?.metrics || {};
    const counters = metrics.counters || {};
    const windows = metrics.windows || {};
    const fresh = this.lastHealthAt !== null
      && performance.now() - this.lastHealthAt < 5000;
    const online = fresh && !this.backendError;
    const recentCutoff = performance.now() - 30000;
    this.longTaskEntries = this.longTaskEntries.filter(
      (entry) => entry.at >= recentCutoff,
    );
    const recentLongTaskMax = this.longTaskEntries.reduce(
      (maximum, entry) => Math.max(maximum, entry.duration),
      0,
    );

    this._set(
      this.elements.backend,
      online ? 'CORE ONLINE' : 'CORE OFFLINE',
      online ? 'ok' : 'error',
      this.backendError || `uptime ${Number(metrics.uptime_s || 0).toFixed(0)} s`,
    );

    const simP95 = windows.sim_step_ms?.p95;
    this._set(
      this.elements.sim,
      `SIM ${formatMs(simP95)}`,
      Number(simP95) > 50 ? 'warn' : online ? 'ok' : 'muted',
      `step p95 / ${windows.sim_step_ms?.samples || 0} samples`,
    );

    const captureMs = sensors.latest_capture_ms;
    const sensorLevel = sensors.bridge_dropped_frames > 0 ? 'warn' : 'ok';
    this._set(
      this.elements.sensor,
      `RGB-D ${formatMs(captureMs)}`,
      sensors.bridge_enabled ? sensorLevel : 'muted',
      `${sensors.bridge_frames} frames / ${sensors.bridge_dropped_frames} dropped / worker ${formatMs(sensors.latest_bridge_encode_ms)}`,
    );

    this._set(
      this.elements.network,
      `WS ${formatMs(network.rtt_ms)} · ${formatBytes(network.buffered_amount)}`,
      !network.connected ? 'error' : network.buffered_amount > 1024 ** 2 ? 'warn' : 'ok',
      `RTT / buffered · reconnects ${network.reconnects}`,
    );

    const coalesced = Number(counters.sim_state_coalesced_total || 0);
    const pipelineP95 = windows.sensor_capture_to_packet_ms?.p95;
    this._set(
      this.elements.flow,
      `PIPE ${formatMs(pipelineP95)} · C${coalesced}`,
      Number(pipelineP95) > 150 ? 'warn' : 'ok',
      `capture→packet p95 · coalesced states ${coalesced}`,
    );

    const presentation = this.presentation?.statistics();
    this._set(
      this.elements.page,
      presentationHidden() ? 'VIEW HIDDEN' : presentation?.idle ? 'VIEW IDLE'
        : `VIEW ${presentation?.fps.toFixed(0) ?? '—'} FPS`,
      presentationHidden() ? 'muted' : recentLongTaskMax > 200 ? 'warn' : 'ok',
      `render CPU p95 ${formatMs(presentation?.render_submission_p95_ms)} / target ${presentation?.target_fps ?? '—'} FPS · long tasks 30 s: ${this.longTaskEntries.length} / max ${recentLongTaskMax.toFixed(0)} ms`,
    );
    this.elements.root.dataset.level = online && network.connected ? 'ok' : 'error';
  }

  _set(element, text, level, title) {
    if (!element) return;
    element.textContent = text;
    element.dataset.level = level;
    element.title = title;
  }

  _observeLongTasks() {
    if (!('PerformanceObserver' in window)) return;
    try {
      this.observer = new PerformanceObserver((list) => {
        for (const entry of list.getEntries()) {
          this.longTasks += 1;
          this.longTaskMaxMs = Math.max(this.longTaskMaxMs, entry.duration);
          this.longTaskEntries.push({ at: performance.now(), duration: entry.duration });
        }
      });
      this.observer.observe({ type: 'longtask', buffered: true });
    } catch (error) {
      console.debug('[RuntimeHealth] Long-task observer unavailable:', error);
    }
  }
}
