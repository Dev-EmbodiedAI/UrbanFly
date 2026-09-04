import { SceneManager } from './scene.js';
import { CityRenderer } from './city.js';
import { DigitalTwinRenderer } from './digital_twin.js';
import { DroneManager } from './drone.js';
import { PathVisualizer } from './path.js';
import { CameraManager } from './camera.js';
import { NetworkClient } from './network.js';
import { UIManager } from './ui.js';
import { Minimap } from './minimap.js';
import { DemoFlight } from './demo.js';
import { DroneSensorSuite } from './drone_sensors.js';
import { DynamicActorRenderer } from './dynamic_actors.js';
import { SemanticFleetRenderer } from './semantic_fleet.js';
import { RuntimeRecorder } from './runtime_recorder.js';
import { RuntimeHealth } from './runtime_health.js';
import { bindHostLifecycle } from './host_lifecycle.js';
import { BusinessNodeAnnotationController } from './semantic_annotations.js';

const state = {
  drones: [],
  tasks: [],
  commGraph: null,
  stats: null,
  simTime: 0,
  simState: 'stopped',
  events: [],
  semanticAgent: null,
};

const UI_UPDATE_INTERVAL_MS = 250;
const MINIMAP_UPDATE_INTERVAL_MS = 100;
let lastTelemetryUiUpdate = 0;
let lastSensorConfigSignature = '';
let lastAppearanceSignature = '';

async function init() {
  bindHostLifecycle();
  const container = document.getElementById('canvas-container');
  const canvas = document.getElementById('render-canvas');
  const sceneManager = new SceneManager(container, canvas);
  sceneManager.init();

  const droneManager = new DroneManager(sceneManager.scene);
  const pathVisualizer = new PathVisualizer(sceneManager.scene);
  const cameraManager = new CameraManager(sceneManager.camera, sceneManager.controls);
  const minimap = new Minimap(sceneManager);
  const uiManager = new UIManager(state);
  // Demo is explicit; an idle/disconnected engine must not fabricate flights.
  const demoFlight = new URLSearchParams(window.location.search).get('demo') === '1'
    ? new DemoFlight(droneManager, pathVisualizer, state, uiManager) : null;
  const sensorSuite = new DroneSensorSuite(sceneManager, droneManager);
  const dynamicActorRenderer = new DynamicActorRenderer(sceneManager.scene);
  const semanticFleetRenderer = new SemanticFleetRenderer(sceneManager.scene);
  const digitalTwin = new DigitalTwinRenderer(
    sceneManager,
    (status) => uiManager.setTwinStatus(status),
  );
  const semanticAnnotations = new BusinessNodeAnnotationController({ sceneManager, digitalTwin });

  let backendActive = false;
  let lastUiUpdate = 0;
  let lastMinimapUpdate = 0;
  let sceneReady = false;

  uiManager.setConnectionState(false);
  uiManager.addEvent({ time: 0, message: '1 km × 1 km 运行空域已建立' });
  uiManager.addEvent({ time: 0, message: demoFlight ? '本地示范模式' : '等待真实引擎任务；未开始采集' });

  digitalTwin.load('CityCentral1km').then((metadata) => {
    sceneReady = true;
    document.getElementById('scene-source').textContent = '城市实景摄影测量网格';
    document.getElementById('scene-extent').textContent = `${metadata.operationSize} × ${metadata.operationSize} m`;
    minimap.setExtent(metadata.operationSize);
    uiManager.addEvent({ time: state.simTime, message: '城市 1 km 真实摄影测量网格与功能分区加载完成' });
  }).catch(async (error) => {
    console.error('[DigitalTwin] Falling back to procedural city:', error);
    sceneManager.enableProceduralGround();
    const cityRenderer = new CityRenderer(sceneManager.scene);
    await cityRenderer.loadDefault();
    uiManager.setTwinStatus({
      phase: 'fallback',
      title: '实景资产暂不可用',
      detail: '已切换到可计算城市代理层',
    });
    document.getElementById('scene-source').textContent = '城市代理模型';
  });

  const network = new NetworkClient(`ws://${window.location.hostname}:8765/ws`);
  const runtimeRecorder = new RuntimeRecorder(canvas, network, state, cameraManager);
  const runtimeHealth = new RuntimeHealth({
    endpoint: `${window.location.protocol}//${window.location.hostname}:8765/api/health`,
    network,
    sensors: window.urbanFlySensors,
    presentation: sceneManager,
    sceneReady: () => sceneReady,
  });
  const benchmarkButton = document.getElementById('display-benchmark');
  const benchmarkResult = document.getElementById('display-benchmark-result');
  benchmarkButton.addEventListener('click', () => {
    if (!sceneReady || sensorSuite.bridgeEnabled) {
      benchmarkResult.textContent = '请等待场景就绪，并先停止采集。';
      return;
    }
    benchmarkResult.removeAttribute('data-report');
    benchmarkResult.textContent = '预热 1 秒，测量 10 秒；不启动仿真。';
    benchmarkButton.disabled = true;
    document.getElementById('display-lod').disabled = true;
    document.getElementById('display-quality').disabled = true;
    sceneManager.benchmark.start(performance.now());
    sceneManager.invalidate();
  });
  sceneManager.onBenchmarkComplete = (report) => {
    benchmarkButton.disabled = false;
    document.getElementById('display-lod').disabled = false;
    document.getElementById('display-quality').disabled = false;
    benchmarkResult.dataset.report = JSON.stringify(report);
    benchmarkResult.textContent = report.cancelled ? '测速已取消（采集或隐藏窗口）'
      : `${report.fps.toFixed(1)} FPS · P95 ${report.frame_interval_p95_ms?.toFixed(1)} ms · ${report.draw_calls_median} 次绘制`;
  };
  window.urbanFlySensors.setBridgeSender(
    (packet) => network.sendBinary(packet),
  );
  window.urbanFlySensors.setBridgeCaptureStartedSender(
    (payload) => network.send('sensor_capture_started', payload),
  );
  network.on('connection', ({ connected }) => uiManager.setConnectionState(connected));
  network.on('sensor_bridge_control', (payload) => {
    window.urbanFlySensors.setBridgeEnabled(Boolean(payload.enabled));
    uiManager.addEvent({
      time: state.simTime,
      message: payload.enabled
        ? '世界模型 RGB-D 二进制桥已启用'
        : '世界模型 RGB-D 二进制桥已停用',
    });
  });
  network.on('runtime_recording_control', (payload) => {
    runtimeRecorder.handle(payload).catch((error) => {
      network.send('runtime_recording_failed', {
        recording_id: payload.recording_id,
        error: String(error),
      });
    });
  });
  network.on('scenario_list', (payload) => uiManager.populateScenarioList(payload));
  network.on('algorithm_list', (payload) => uiManager.populateAlgorithmList(payload));
  network.on('scenario_start', (payload) => {
    pathVisualizer.clearAll();
    uiManager.clearEvents();
    uiManager.showNotification(`任务场景已启动：${payload.name}`);
    resetState();
  });
  network.on('sim_state', (payload) => {
    if (!backendActive) {
      backendActive = true;
      demoFlight?.pause();
      uiManager.addEvent({ time: payload.t || 0, message: '已接管实时仿真状态流' });
    }
    updateFromSimState(
      payload,
      sceneManager,
      droneManager,
      pathVisualizer,
      minimap,
      uiManager,
      cameraManager,
      dynamicActorRenderer,
      semanticFleetRenderer,
      sensorSuite,
    );
  });
  network.on('event', (payload) => {
    state.events.push(payload);
    if (state.events.length > 200) state.events.shift();
    uiManager.addEvent(payload);
  });
  network.on('scenario_end', () => {
    state.simState = 'completed';
    uiManager.showNotification('仿真任务已完成');
  });
  network.on('algorithm_changed', (payload) => {
    uiManager.showNotification(`调度算法已切换：${payload.algorithm}`);
  });

  uiManager.onSelectScenario = (name) => {
    if (network.connected) network.send('select_scenario', { name });
  };
  uiManager.onSelectAlgorithm = (algorithm) => {
    if (network.connected) network.send('select_algorithm', { algorithm });
  };
  uiManager.onControl = (action, value) => {
    if (network.connected) {
      network.send('control', { action, value });
      return;
    }
    if (action === 'play') demoFlight?.start();
    if (action === 'pause') demoFlight?.pause();
    if (action === 'stop') demoFlight?.reset();
    if (action === 'set_speed') demoFlight?.setSpeed(value);
  };

  window.addEventListener('keydown', (event) => {
    if (event.key === ' ') {
      event.preventDefault();
      uiManager.onControl(state.simState === 'running' ? 'pause' : 'play');
    } else if (event.key === '1') {
      cameraManager.setMode('orbit');
    } else if (event.key === '2') {
      cameraManager.setMode('follow', state.drones[0]?.id);
    } else if (event.key === '3') {
      cameraManager.setMode('topdown');
    } else if (event.key.toLowerCase() === 'f' && state.drones.length > 0) {
      const current = state.drones.findIndex((drone) => drone.id === cameraManager.followTarget);
      cameraManager.setMode('follow', state.drones[(current + 1) % state.drones.length]?.id);
    }
    sceneManager.invalidate();
  });

  function animate() {
    requestAnimationFrame(animate);
    const now = performance.now();
    const busy = sensorSuite.bridgeEnabled;
    const engineState = runtimeHealth.backend?.simulator?.state ?? state.simState;
    if (!sceneManager.shouldRender(now, { busy, idle: engineState !== 'running' })) return;
    const dt = Math.min(0.1, sceneManager.clock.getDelta());

    if (!backendActive) demoFlight?.update(dt);
    cameraManager.update(state.drones, dt);
    sensorSuite.update(state.drones, dt, state.simTime);

    if (now - lastMinimapUpdate >= MINIMAP_UPDATE_INTERVAL_MS) {
      minimap.update(state.drones, state.commGraph);
      lastMinimapUpdate = now;
    }
    if (now - lastUiUpdate > 500) {
      uiManager.updateDroneList(state.drones);
      lastUiUpdate = now;
    }
    sceneManager.render({ busy });
  }

  animate();
  network.connect();
  runtimeHealth.start();
}

function resetState() {
  state.drones = [];
  state.tasks = [];
  state.commGraph = null;
  state.stats = null;
  state.simTime = 0;
  state.simState = 'running';
  state.events = [];
  state.semanticAgent = null;
}

function updateFromSimState(
  payload,
  sceneManager,
  droneManager,
  pathVisualizer,
  minimap,
  uiManager,
  cameraManager,
  dynamicActorRenderer,
  semanticFleetRenderer,
  sensorSuite,
) {
  sceneManager.invalidate();
  state.simTime = payload.t;
  state.simState = payload.state;
  state.stats = payload.stats;
  dynamicActorRenderer.update(payload.actors || []);
  state.semanticAgent = payload.semantic_agent || null;
  semanticFleetRenderer.update(state.semanticAgent);

  if (payload.sensor_config) {
    const signature = JSON.stringify(payload.sensor_config);
    if (signature !== lastSensorConfigSignature) {
      lastSensorConfigSignature = signature;
      sensorSuite.applyConfig(payload.sensor_config);
    }
  }
  if (payload.appearance_perturbation) {
    const signature = JSON.stringify(payload.appearance_perturbation);
    if (signature !== lastAppearanceSignature) {
      lastAppearanceSignature = signature;
      sceneManager.applyAppearance(payload.appearance_perturbation);
      sensorSuite.applyPerturbation(payload.appearance_perturbation);
    }
  }

  if (payload.drones) {
    state.drones = payload.drones;
    droneManager.update(payload.drones);
    // Capture the synchronized policy frame before path/UI visualization.
    // This is the real-time data plane and is not coupled to rAF rendering.
    sensorSuite.captureBridgeState(payload.drones, payload.t);
    for (const drone of payload.drones) {
      if (drone.path_remaining?.length > 0) {
        pathVisualizer.updatePath(
          drone.id,
          [drone.pos, ...drone.path_remaining],
          getDroneColor(drone.drone_type),
        );
      }
      pathVisualizer.updateWorldModel(
        drone.id,
        drone.pos,
        drone.world_model,
      );
      pathVisualizer.updateExecutedTrace(drone.id, drone.pos);
    }
    if (cameraManager.mode === 'follow' && !cameraManager.followTarget && payload.drones.length > 0) {
      cameraManager.setMode('follow', payload.drones[0].id);
    }
  }
  if (payload.tasks) {
    state.tasks = payload.tasks;
  }
  if (payload.comm_graph) {
    state.commGraph = payload.comm_graph;
  }
  const now = performance.now();
  if (now - lastTelemetryUiUpdate >= UI_UPDATE_INTERVAL_MS) {
    if (payload.tasks) uiManager.updateTasks(payload.tasks);
    if (payload.comm_graph) {
      uiManager.updateCommGraph(payload.comm_graph, payload.topology_stats);
    }
    if (payload.stats) uiManager.updateStats(payload.stats);
    uiManager.updateWorldModel(payload.drones);
    if (payload.alloc_stats) uiManager.updateAllocStats(payload.alloc_stats);
    uiManager.updateSemanticAgent(payload.semantic_agent);
    uiManager.updateTime(payload.t);
    lastTelemetryUiUpdate = now;
  }
}

function getDroneColor(type) {
  return {
    heavy: 0xff8f70,
    standard: 0x54c7ff,
    light: 0x72e0ae,
  }[type] || 0xffffff;
}

init();
