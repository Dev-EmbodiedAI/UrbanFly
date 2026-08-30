# UrbanFly Digital-Twin Runtime Architecture

## Purpose

This document describes the real-time browser/backend boundary used by the
Helsinki digital twin. It covers runtime transport, synchronized RGB-D, UI
health, and failure isolation. It does not redefine the frozen navigation,
expert, controller, sampler, triangle-geometry, or Local Goal algorithms.

Updated operational entry: native `UrbanFly.Desktop.exe`; see
`DESKTOP_RUNTIME.md`. Latest measured results and remaining limitations are in
`PROJECT_STATE.md` section 28 (2026-08-28). The figures below labeled 2026-08-26
are historical, not the latest end-to-end qualification.

## Runtime topology

```text
backend simulator (20 Hz)
  |-- reliable ordered path --> policy/reset/action ACK/event clients
  `-- latest-value path ------> ordinary visualization clients
                                  |
                                  v
                         sim_state event handler
                           |-- scene pose/actors
                           |-- immediate RGB-D WebGL capture
                           `-- deferred path/UI visualization
                                  |
                                  v
                         RGB-D packet Web Worker
                    flip + RGB8 + depth-u16 + UFWM packet
                                  |
                                  v
                         WebSocket binary bridge
                                  |
                                  v
                   backend validation + policy relay
```

The data plane is event-driven. Synchronized policy capture is never scheduled
by `requestAnimationFrame`, so tab visibility and monitor refresh rate do not
govern collector progress. The presentation plane may still use rAF because it
is replaceable visual work.

## Backpressure and delivery semantics

- Control, scenario changes, reset, action ACK, events, and policy-client state
  are reliable and ordered.
- An ordinary visualization socket has one pending `sim_state` slot. A newer
  state replaces an unsent older state. A slow UI therefore cannot stall the
  simulator or accumulate an unbounded historical queue.
- Sensor packets remain fail-closed and schema-validated before policy relay.
- Browser WebSocket metrics expose RTT, current/high-water buffered bytes,
  reconnects, rejected writes, and traffic counts.

## Observability

`GET /api/health` returns schema `urbanfly-runtime-health-v1` with:

- simulator state, sim time, and loop status;
- connected visualization, policy, and lockstep-policy client counts;
- counters for state coalescing, packets, bytes, connections, disconnects, and
  send errors;
- bounded latency windows with mean, P50, P95, P99, and maximum for simulator
  steps, WebSocket sends, and capture-to-packet latency.

The bottom runtime-health strip renders CORE, SIM, RGB-D, WS, PIPE, and VIEW.
It includes a rolling 30-second browser long-task signal, so cold asset loading
does not leave the interface permanently marked degraded.

The latest frontend also reports bounded surface readiness, host visibility,
presentation FPS/CPU submission P95, GPU resource counts and bridge statistics.
Health requests are single-flight with a timeout. Stopped unchanged scenes
render on demand; minimized native windows forward visibility explicitly because
Page Visibility alone was insufficient in real WebView2 testing. Rendering
targets and measured achieved FPS are reported separately.

## Runtime invariants

1. Frozen navigation/control components are outside this architecture layer.
2. At most one synchronized bridge encoding may be in flight.
3. Duplicate `sim_state` timestamps do not generate duplicate RGB-D packets.
4. The browser sends `sensor_capture_started` only when it can begin a real
   capture; lockstep is not paused for a knowingly skipped busy frame.
5. RGB-D conversion and UFWM packet assembly run in a Web Worker. WebGL scene
   capture/readback remains on the rendering context owner.
6. Operator telemetry is bounded in memory and does not require a cloud
   service or an internet connection.
7. Common actors/lights use layer 0; original L21 city uses sensor layer 1;
   display-selected L21/L18 meshes use layer 2. Sensor camera mask is 0+1 and
   display camera mask is 0+2. Display LOD never removes layer 1 or changes
   high-detail geometry/materials/visibility. Near/far selection is per tile
   with hysteresis; missing overview falls back to high detail.
8. Display counters reset before presentation and accumulate all composer
   passes, then restore the renderer's counter policy. Sensor draws are not
   misreported as presentation draw calls.

2026-08-28 measured follow-up: section 29 of `PROJECT_STATE.md` records the
same-view LOD comparison, real reset/hover probes, retained-memory cost and
remaining shared-context bottleneck. These supersede the historical benchmark
below without implying a complete navigation qualification.

## Current validation (2026-08-26)

- Production Vite build: PASS.
- Runtime/backend focused tests: 8/8 PASS.
- Real task-053 reset to first synchronized RGB-D: 0.304 s with Worker packetizer.
- Twenty continuous lockstep actions: 20/20 PASS, zero timeout, mean/P95/max
  wall latency 0.254/0.319/0.324 s, exact 0.100 s mean simulation dt, and zero
  timestamp regression.
- RGB `90 x 160 x 3`, depth `90 x 160`, finite-depth ratio 100%, browser bridge
  21 frames and zero dropped frames.
- Health window after the probe: simulator-step P95 0.910 ms,
  capture-to-packet P95 126.673 ms, and WebSocket text-send P95 0.465 ms.

## Known boundaries

- The Three.js/WebGL renderer and render-target readback still own the main
  rendering thread. A full OffscreenCanvas renderer migration would require
  moving scene ownership, asset loading, camera controls, resize, picking, and
  input forwarding together; it is a separate regression milestone.
- The production main bundle is about 840 kB minified (227 kB gzip) and Vite
  emits its standard 500 kB chunk warning. The Helsinki photogrammetry asset
  stream remains much larger than the JavaScript bundle; code splitting should
  be measured against actual cold-start traces before changing load order.
- Runtime health is local and dependency-free. Its counter/histogram naming is
  ready for an OpenTelemetry exporter if remote fleet observability is later
  required.
- Use the backend-served production build at `http://127.0.0.1:8765/` for
  collection and long-running operation. The Vite development server is for
  code iteration; repeated hot reloads of the full photogrammetry scene can
  temporarily duplicate GPU resources and are not an operational topology.
