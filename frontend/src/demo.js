import * as THREE from 'three';

const ROUTE_DEFINITIONS = [
  {
    id: 'UF-07',
    drone_type: 'standard',
    color: 0x54c7ff,
    offset: 0,
    speed: 0.022,
    points: [
      [-185, 26, -145],
      [-148, 82, -112],
      [-70, 118, -92],
      [18, 132, -35],
      [86, 120, 46],
      [165, 28, 132],
    ],
  },
  {
    id: 'UF-12',
    drone_type: 'light',
    color: 0x72e0ae,
    offset: 0.31,
    speed: 0.027,
    points: [
      [-205, 34, 82],
      [-142, 88, 45],
      [-46, 112, 18],
      [48, 128, 72],
      [195, 44, 104],
    ],
  },
  {
    id: 'UF-03',
    drone_type: 'heavy',
    color: 0xff8f70,
    offset: 0.62,
    speed: 0.017,
    points: [
      [-176, 42, 178],
      [-108, 96, 126],
      [-12, 142, 94],
      [90, 126, 10],
      [184, 38, -118],
    ],
  },
];

export class DemoFlight {
  constructor(droneManager, pathVisualizer, state, uiManager) {
    this.droneManager = droneManager;
    this.pathVisualizer = pathVisualizer;
    this.state = state;
    this.uiManager = uiManager;
    this.running = true;
    this.elapsed = 0;
    this.speedMultiplier = 1;
    this.routes = ROUTE_DEFINITIONS.map((definition) => ({
      ...definition,
      curve: new THREE.CatmullRomCurve3(
        definition.points.map(([x, y, z]) => new THREE.Vector3(x, y, z)),
        false,
        'catmullrom',
        0.36,
      ),
    }));
    this._publishStaticData();
  }

  start() {
    this.running = true;
    this.state.simState = 'running';
  }

  pause() {
    this.running = false;
    this.state.simState = 'paused';
  }

  reset() {
    this.elapsed = 0;
    this.running = false;
    this.state.simState = 'stopped';
    this.update(0);
  }

  setSpeed(multiplier) {
    this.speedMultiplier = Math.max(0.25, Number(multiplier) || 1);
  }

  update(dt) {
    if (this.running) this.elapsed += dt * this.speedMultiplier;

    const drones = this.routes.map((route, index) => {
      const t = (route.offset + this.elapsed * route.speed) % 1;
      const easedT = t < 0.08
        ? THREE.MathUtils.smoothstep(t, 0, 0.08) * 0.08
        : t > 0.92
          ? 0.92 + THREE.MathUtils.smoothstep(t, 0.92, 1) * 0.08
          : t;
      const position = route.curve.getPointAt(easedT);
      const tangent = route.curve.getTangentAt(Math.min(0.999, easedT + 0.002));
      const yaw = THREE.MathUtils.radToDeg(Math.atan2(tangent.x, tangent.z));
      const battery = 0.94 - ((this.elapsed * 0.0027 + index * 0.08) % 0.42);

      return {
        id: route.id,
        drone_type: route.drone_type,
        pos: position.toArray(),
        yaw,
        speed: this.running ? 11.6 + index * 1.8 : 0,
        battery,
        payload: index === 0 ? 2.4 : index === 2 ? 6.8 : 0.8,
        state: this.running ? 'en_route' : 'hovering',
        current_task: `DEL-${String(index + 17).padStart(3, '0')}`,
        comm_neighbors: this.routes.filter((_, neighbor) => neighbor !== index).map((item) => item.id),
      };
    });

    this.state.drones = drones;
    this.state.simTime = this.elapsed;
    this.droneManager.update(drones);
    this.uiManager.updateTime(this.elapsed);

    if (Math.floor(this.elapsed * 2) % 4 === 0) {
      this.uiManager.updateDroneList(drones);
    }
  }

  _publishStaticData() {
    for (const route of this.routes) {
      this.pathVisualizer.updatePath(
        route.id,
        route.points,
        route.color,
      );
    }

    this.state.tasks = [
      {
        id: 'DEL-017',
        priority: 1,
        status: 'en_route_delivery',
        assigned_to: 'UF-07',
        business_tag: '医疗样本转运',
        pickup_district: 'S-01',
        delivery_district: 'D-04',
        cold_chain: true,
        fragile: true,
        risk_level: 0.31,
      },
      {
        id: 'DEL-018',
        priority: 3,
        status: 'assigned',
        assigned_to: 'UF-12',
        business_tag: '社区即时配送',
        pickup_district: 'S-03',
        delivery_district: 'D-02',
        cold_chain: false,
        fragile: false,
        risk_level: 0.14,
      },
      {
        id: 'DEL-019',
        priority: 2,
        status: 'en_route_delivery',
        assigned_to: 'UF-03',
        business_tag: '生鲜补给',
        pickup_district: 'S-02',
        delivery_district: 'D-05',
        cold_chain: true,
        fragile: false,
        risk_level: 0.26,
      },
    ];
    this.uiManager.updateTasks(this.state.tasks);
    this.uiManager.updateStats({
      total_tasks: 3,
      completed: 0,
      in_progress: 3,
      pending: 0,
      failed: 0,
      on_time_rate: 1,
      avg_task_risk: 0.24,
      total_distance: 812,
      total_energy: 64,
      avg_battery: 0.86,
      collision_warnings: 0,
      path_replans: 1,
      comm_disconnections: 0,
    });
  }
}
