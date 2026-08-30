export class RuntimeRecorder {
  constructor(canvas, network, state, cameraManager) {
    this.canvas = canvas;
    this.network = network;
    this.state = state;
    this.cameraManager = cameraManager;
    this.recorder = null;
    this.chunks = [];
    this.request = null;
    this.startedWallMs = 0;
    this.startedSimTime = 0;
    this.captureCanvas = document.createElement('canvas');
    this.captureCanvas.width = 1920;
    this.captureCanvas.height = 1080;
    this.captureContext = this.captureCanvas.getContext('2d');
    this.sensorRgbCanvas = document.createElement('canvas');
    this.sensorDepthCanvas = document.createElement('canvas');
    this.thirdPersonCanvas = document.createElement('canvas');
    this.overviewCanvas = document.createElement('canvas');
    this.sensorUnsubscribe = null;
    this.compositorFrame = null;
  }

  async handle(payload = {}) {
    if (payload.action === 'start') return this.start(payload);
    if (payload.action === 'stop') return this.stop(payload);
    throw new Error(`unknown recording action: ${payload.action}`);
  }

  start(payload) {
    if (this.recorder?.state === 'recording') throw new Error('runtime recording is already active');
    if (!this.captureCanvas.captureStream || typeof MediaRecorder === 'undefined') {
      throw new Error('this browser does not support canvas MediaRecorder');
    }
    const fps = Math.max(1, Math.min(60, Number(payload.fps || 30)));
    const mimeCandidates = [
      'video/webm;codecs=vp9',
      'video/webm;codecs=vp8',
      'video/webm',
    ];
    const mimeType = mimeCandidates.find((item) => MediaRecorder.isTypeSupported(item)) || '';
    this._startCompositor(payload);
    const stream = this.captureCanvas.captureStream(fps);
    this.recorder = new MediaRecorder(stream, {
      ...(mimeType ? { mimeType } : {}),
      videoBitsPerSecond: Number(payload.video_bitrate || 12_000_000),
    });
    this.chunks = [];
    this.request = { ...payload, fps, mime_type: mimeType || this.recorder.mimeType };
    this.startedWallMs = performance.now();
    this.startedSimTime = Number(this.state.simTime || 0);
    this.recorder.ondataavailable = (event) => {
      if (event.data?.size) this.chunks.push(event.data);
    };
    this.recorder.start(1000);
    if (payload.camera_mode === 'follow' && this.state.drones[0]) {
      this.cameraManager.followDistance = Math.max(10, Math.min(60, Number(payload.follow_distance_m || 22)));
      this.cameraManager.followHeight = Math.max(4, Math.min(30, Number(payload.follow_height_m || 9)));
      this.cameraManager.setMode('follow', this.state.drones[0].id);
    }
    this.network.send('runtime_recording_started', {
      recording_id: payload.recording_id,
      fps,
      mime_type: this.request.mime_type,
      sim_start_s: this.startedSimTime,
      source: 'continuous four-panel runtime compositor + MediaRecorder',
      layout: 'FPV / RGB / depth / candidates+risk',
    });
  }

  async stop(payload) {
    if (!this.recorder || this.recorder.state !== 'recording') {
      throw new Error('runtime recording is not active');
    }
    const stopped = new Promise((resolve) => {
      this.recorder.onstop = resolve;
    });
    this.recorder.stop();
    await stopped;
    this._stopCompositor();
    const wallDurationS = (performance.now() - this.startedWallMs) / 1000;
    const simEndS = Number(this.state.simTime || 0);
    const blob = new Blob(this.chunks, { type: this.request.mime_type || 'video/webm' });
    const recordingId = String(payload.recording_id || this.request.recording_id);
    const response = await fetch(`/api/runtime-recordings/${encodeURIComponent(recordingId)}`, {
      method: 'POST',
      headers: {
        'Content-Type': blob.type,
        'X-Target-Fps': String(this.request.fps),
        'X-Sim-Start-S': String(this.startedSimTime),
        'X-Sim-End-S': String(simEndS),
        'X-Wall-Duration-S': String(wallDurationS),
        'X-Source': 'continuous four-panel canvas.captureStream MediaRecorder',
      },
      body: blob,
    });
    if (!response.ok) throw new Error(`recording upload failed: ${response.status}`);
    const uploaded = await response.json();
    this.network.send('runtime_recording_complete', {
      ...uploaded,
      target_fps: this.request.fps,
      sim_start_s: this.startedSimTime,
      sim_end_s: simEndS,
      wall_duration_s: wallDurationS,
      encoded_bytes: blob.size,
      source: 'continuous_browser_runtime',
      screenshot_stitching: false,
      layout: 'FPV / RGB / depth / candidates+risk',
    });
    this.recorder = null;
    this.chunks = [];
    this.request = null;
  }

  _startCompositor(payload = {}) {
    this._stopCompositor();
    const sensors = window.urbanFlySensors;
    sensors?.setThirdPersonCapture?.({
      enabled: true,
      distance_m: Number(payload.follow_distance_m || 22),
      height_m: Number(payload.follow_height_m || 9),
    });
    if (sensors?.subscribe) {
      this.sensorUnsubscribe = sensors.subscribe((frame) => this._updateSensorPanels(frame));
    }
    const draw = () => {
      this._drawCompositeFrame();
      this.compositorFrame = requestAnimationFrame(draw);
    };
    draw();
  }

  _stopCompositor() {
    if (this.compositorFrame != null) cancelAnimationFrame(this.compositorFrame);
    this.compositorFrame = null;
    this.sensorUnsubscribe?.();
    this.sensorUnsubscribe = null;
    window.urbanFlySensors?.setThirdPersonCapture?.({ enabled: false });
  }

  _updateSensorPanels(frame) {
    const width = Number(frame?.width || 0);
    const height = Number(frame?.height || 0);
    if (!width || !height) return;
    this.sensorRgbCanvas.width = width;
    this.sensorRgbCanvas.height = height;
    this.sensorDepthCanvas.width = width;
    this.sensorDepthCanvas.height = height;
    const thirdWidth = Number(frame?.third_person_width || 0);
    const thirdHeight = Number(frame?.third_person_height || 0);
    if (thirdWidth && thirdHeight && frame.third_person_image_data_uint8) {
      this.thirdPersonCanvas.width = thirdWidth;
      this.thirdPersonCanvas.height = thirdHeight;
      const third = this.thirdPersonCanvas.getContext('2d');
      const thirdImage = third.createImageData(thirdWidth, thirdHeight);
      const source = frame.third_person_image_data_uint8;
      for (let y = 0; y < thirdHeight; y += 1) {
        const sourceY = thirdHeight - 1 - y;
        for (let x = 0; x < thirdWidth; x += 1) {
          const sourcePixel = sourceY * thirdWidth + x;
          const target = (y * thirdWidth + x) * 4;
          const sourceRgb = sourcePixel * 3;
          thirdImage.data[target] = source[sourceRgb] || 0;
          thirdImage.data[target + 1] = source[sourceRgb + 1] || 0;
          thirdImage.data[target + 2] = source[sourceRgb + 2] || 0;
          thirdImage.data[target + 3] = 255;
        }
      }
      third.putImageData(thirdImage, 0, 0);
    }
    const overviewWidth = Number(frame?.overview_width || 0);
    const overviewHeight = Number(frame?.overview_height || 0);
    if (overviewWidth && overviewHeight && frame.overview_image_data_uint8) {
      this.overviewCanvas.width = overviewWidth;
      this.overviewCanvas.height = overviewHeight;
      const overview = this.overviewCanvas.getContext('2d');
      const overviewImage = overview.createImageData(overviewWidth, overviewHeight);
      const source = frame.overview_image_data_uint8;
      for (let y = 0; y < overviewHeight; y += 1) {
        const sourceY = overviewHeight - 1 - y;
        for (let x = 0; x < overviewWidth; x += 1) {
          const sourcePixel = sourceY * overviewWidth + x;
          const target = (y * overviewWidth + x) * 4;
          const sourceRgb = sourcePixel * 3;
          overviewImage.data[target] = source[sourceRgb] || 0;
          overviewImage.data[target + 1] = source[sourceRgb + 1] || 0;
          overviewImage.data[target + 2] = source[sourceRgb + 2] || 0;
          overviewImage.data[target + 3] = 255;
        }
      }
      overview.putImageData(overviewImage, 0, 0);
      const position = frame.vehicle_pose?.position || [0, 0, 0];
      const markerX = ((Number(position[0]) + 1000) / 2000) * overviewWidth;
      const markerY = ((Number(position[2]) + 560) / 1120) * overviewHeight;
      overview.strokeStyle = '#ffffff';
      overview.lineWidth = 4;
      overview.fillStyle = '#29ff78';
      overview.beginPath();
      overview.arc(markerX, markerY, 9, 0, Math.PI * 2);
      overview.fill();
      overview.stroke();
    }
    const rgb = this.sensorRgbCanvas.getContext('2d');
    const depth = this.sensorDepthCanvas.getContext('2d');
    const rgbImage = rgb.createImageData(width, height);
    const depthImage = depth.createImageData(width, height);
    const rgbSource = frame.image_data_uint8 || [];
    const depthSource = frame.image_data_float || [];
    for (let y = 0; y < height; y += 1) {
      const sourceY = height - 1 - y;
      for (let x = 0; x < width; x += 1) {
        const sourcePixel = sourceY * width + x;
        const target = (y * width + x) * 4;
        const sourceRgb = sourcePixel * 3;
        rgbImage.data[target] = rgbSource[sourceRgb] || 0;
        rgbImage.data[target + 1] = rgbSource[sourceRgb + 1] || 0;
        rgbImage.data[target + 2] = rgbSource[sourceRgb + 2] || 0;
        rgbImage.data[target + 3] = 255;
        const range = Number(depthSource[sourcePixel] || 120);
        const normalized = Math.min(1, Math.log1p(Math.max(0, range)) / Math.log1p(120));
        const [r, g, b] = this._turbo(1 - normalized);
        depthImage.data[target] = r;
        depthImage.data[target + 1] = g;
        depthImage.data[target + 2] = b;
        depthImage.data[target + 3] = 255;
      }
    }
    rgb.putImageData(rgbImage, 0, 0);
    depth.putImageData(depthImage, 0, 0);
  }

  _drawCompositeFrame() {
    const context = this.captureContext;
    context.fillStyle = '#03070b';
    context.fillRect(0, 0, 1920, 1080);
    const longRange = this.request?.layout === 'long_range_world_model';
    this._drawPanel(
      longRange ? this.overviewCanvas : this.thirdPersonCanvas,
      0, 0, 960, 540,
      longRange ? '全地图实时轨迹 · 1 km 空域' : '近机第三视角 · 真实执行轨迹',
    );
    this._drawPanel(this.sensorRgbCanvas, 960, 0, 960, 540, '机载 RGB · 640×360 @ 10 Hz');
    this._drawPanel(
      longRange ? this.thirdPersonCanvas : this.sensorDepthCanvas,
      0, 540, 960, 540,
      longRange ? '近机第三视角 · 真实执行轨迹' : '深度 · 0–120 m',
    );
    this._drawPlannerPanel(960, 540, 960, 540);
  }

  _drawPanel(source, x, y, width, height, label) {
    const context = this.captureContext;
    if (source?.width && source?.height) {
      context.drawImage(source, x, y, width, height);
    }
    context.strokeStyle = 'rgba(98,214,255,0.55)';
    context.lineWidth = 2;
    context.strokeRect(x + 1, y + 1, width - 2, height - 2);
    context.fillStyle = 'rgba(2,8,14,0.78)';
    context.fillRect(x + 14, y + 14, Math.min(width - 28, 440), 40);
    context.fillStyle = '#e8f6ff';
    context.font = '600 22px "Segoe UI", sans-serif';
    context.fillText(label, x + 28, y + 42);
  }

  _drawPlannerPanel(x, y, width, height) {
    const context = this.captureContext;
    context.fillStyle = '#07111a';
    context.fillRect(x, y, width, height);
    const drone = this.state.drones.find((item) => item.world_model?.enabled);
    const model = drone?.world_model || {};
    const candidates = model.top_candidates || [];
    const origin = drone?.pos || [0, 0, 0];
    const points = candidates.flatMap((item) => item.trajectory_world_m || []);
    const local = points.map((point) => [Number(point[0]) - origin[0], Number(point[2]) - origin[2]]);
    local.push([0, 0]);
    const extent = Math.max(8, ...local.flatMap((point) => point.map((value) => Math.abs(value))));
    const centerX = x + width * 0.38;
    const centerY = y + height * 0.55;
    const scale = Math.min(width * 0.31, height * 0.38) / extent;
    const project = (point) => [
      centerX + (Number(point[0]) - origin[0]) * scale,
      centerY - (Number(point[2]) - origin[2]) * scale,
    ];
    context.strokeStyle = 'rgba(98,214,255,0.12)';
    context.lineWidth = 1;
    for (let grid = -4; grid <= 4; grid += 1) {
      context.beginPath();
      context.moveTo(centerX + grid * 45, y + 70);
      context.lineTo(centerX + grid * 45, y + height - 25);
      context.stroke();
      context.beginPath();
      context.moveTo(x + 20, centerY + grid * 45);
      context.lineTo(x + width * 0.72, centerY + grid * 45);
      context.stroke();
    }
    for (const candidate of candidates) {
      const selected = candidate.candidate_index === model.selected_index;
      context.strokeStyle = selected
        ? '#ffd166'
        : candidate.predicted_collision ? 'rgba(255,82,99,0.65)' : 'rgba(111,168,255,0.48)';
      context.lineWidth = selected ? 5 : 2;
      context.beginPath();
      context.moveTo(centerX, centerY);
      for (const point of candidate.trajectory_world_m || []) {
        const projected = project(point);
        context.lineTo(projected[0], projected[1]);
      }
      context.stroke();
    }
    context.fillStyle = '#62d6ff';
    context.beginPath();
    context.arc(centerX, centerY, 7, 0, Math.PI * 2);
    context.fill();
    const latent = Array.isArray(model.latent_state) ? model.latent_state : [];
    const heatX = x + width * 0.755;
    const heatY = y + 82;
    const columns = 16;
    const rows = Math.ceil(latent.length / columns);
    const cell = Math.min(11, Math.floor(190 / Math.max(columns, rows, 1)));
    if (latent.length) {
      const mean = latent.reduce((sum, value) => sum + Number(value), 0) / latent.length;
      const variance = latent.reduce((sum, value) => sum + (Number(value) - mean) ** 2, 0) / latent.length;
      const scaleValue = Math.max(Math.sqrt(variance), 1e-6);
      latent.forEach((value, index) => {
        const normalized = 0.5 + 0.5 * Math.tanh((Number(value) - mean) / (2 * scaleValue));
        const [red, green, blue] = this._turbo(normalized);
        context.fillStyle = `rgb(${red},${green},${blue})`;
        context.fillRect(heatX + (index % columns) * cell, heatY + Math.floor(index / columns) * cell, cell - 1, cell - 1);
      });
      context.fillStyle = '#b8d7e8';
      context.font = '13px "Segoe UI", sans-serif';
      context.fillText('192-D observation latent', heatX, heatY - 10);
      context.fillText(
        `||z|| ${Number(model.latent_norm || 0).toFixed(2)} · ||Δz|| ${Number(model.latent_delta_norm || 0).toFixed(3)}`,
        heatX,
        heatY + rows * cell + 20,
      );
    }
    const meterX = heatX;
    const meterY = heatY + rows * cell + 48;
    candidates.slice(0, 6).forEach((candidate, index) => {
      const risk = Math.max(0, Math.min(1, Number(candidate.collision_probability || 0)));
      context.fillStyle = 'rgba(255,255,255,0.10)';
      context.fillRect(meterX, meterY + index * 23, 170, 14);
      context.fillStyle = risk >= 0.5 ? '#ff5263' : '#6ee4a6';
      context.fillRect(meterX, meterY + index * 23, 170 * risk, 14);
      context.fillStyle = '#dbeefa';
      context.font = '12px "Segoe UI", sans-serif';
      context.fillText(`#${candidate.candidate_index}`, meterX - 32, meterY + 11 + index * 23);
    });
    context.fillStyle = '#e8f6ff';
    context.font = '600 22px "Segoe UI", sans-serif';
    context.fillText('World Model · 隐空间 / 候选轨迹 / 风险', x + 26, y + 42);
    context.font = '18px "Segoe UI", sans-serif';
    context.fillStyle = '#b8d7e8';
    context.fillText(
      `方法 ${model.selection_method || model.backend || 'waiting'} · #${model.selected_index ?? '—'} · ${Number(model.planner_latency_ms || model.inference_latency_ms || 0).toFixed(1)} ms`,
      x + 26,
      y + height - 18,
    );
  }

  _turbo(value) {
    const x = Math.max(0, Math.min(1, value));
    return [
      Math.max(0, Math.min(255, 255 * (1.5 - Math.abs(4 * x - 3)))),
      Math.max(0, Math.min(255, 255 * (1.5 - Math.abs(4 * x - 2)))),
      Math.max(0, Math.min(255, 255 * (1.5 - Math.abs(4 * x - 1)))),
    ];
  }
}
