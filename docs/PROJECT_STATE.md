# UrbanFly Persistent Project State

Last audited: 2026-08-28, Asia/Shanghai  
Workspace root: `D:\AI\UrbanFly`

Latest operational handoff: **section 30**. Earlier sections are historical.
Collection was explicitly resumed toward 100 episodes; consult the section 30
job/progress files and live process state for the current count. The integrated
performance target is **not yet met**.

This file is the persistent project handoff. A fresh session must read `AGENTS.md`, then this file completely, then inspect the current code and outputs before proposing changes.

## 0. Repository and Handoff State

### Repository state

- `LIMITATION` — `D:\AI\UrbanFly` currently has no `.git` directory. A real `git status`, baseline diff, tracked/untracked classification, and commit reference are unavailable.
- `LIMITATION` — the modified/new-file inventory below is based on the completed work record, current files, and timestamps. It is not a Git-derived exhaustive diff.
- `PASS` — all key Helsinki assets, code modules, regression records, and audit outputs listed below exist and were read during this handoff.
- `PASS` — this handoff session changed only `AGENTS.md` and `docs/PROJECT_STATE.md`. It did not change business code, start a new experiment, train a model, or collect new data.

### Files changed during the completed Helsinki navigation phase

Existing files changed:

- `backend/config.py` — holds the active `HELSINKI_NAVIGATION` parameters. Current global planning resolution is 5.0 m; the source collision heightmap is 0.5 m.
- `backend/engine/helsinki_navigation.py` — integrates the frozen global route, smoothing, triangle validation/local repair, execution, and `PATH_GEOMETRY_INVALID` fail-closed behavior.
- `backend/engine/simulator.py` — contains the generic low-altitude vertical controller robustness changes.
- `scripts/verify_helsinki_low_altitude_expert.py` — records richer task, clearance, curvature, execution, and spatial metrics.

Files added:

- `backend/engine/triangle_geometry.py`
- `backend/engine/helsinki_urban_sampling.py`
- `backend/engine/local_goal.py`
- `scripts/audit_helsinki_triangle_geometry.py`
- `scripts/verify_helsinki_urban_core_regression.py`
- `scripts/replay_helsinki_controller_failures.py`
- `tests/test_triangle_geometry.py`
- `tests/test_local_goal_selector.py`

`FROZEN` — `backend/engine/planner.py` was not redesigned during the final triangle/controller/sampler phase. It is the existing global XYZ A* implementation and must remain frozen without new failure evidence or an explicit user request.

### Important pre-existing code discovered during handoff

- `PASS` — `uav_wm_navigation` already contains generic HDF5 v2 collection code, simulator adapters, World Model prototypes, MPPI, V-JEPA, Dreamer, and TD-MPC2-related code.
- `PASS` — a five-episode unattended mock collector smoke output exists at `uav_wm_navigation/outputs/collector_reality_smoke5_mock/collection_summary.json`.
- `LIMITATION` — that smoke used `MockCandidatePlanner` and mock episodes. It does not validate integration with the current frozen Helsinki expert, Helsinki triangle oracle, urban-core sampler, or Local Goal interface.
- `NOT TESTED` — a complete 500–1000 episode Helsinki Dataset v1 pilot has not been run.
- `PLANNED` — the next session should inspect and extend the existing collector where appropriate rather than assuming no collector exists or rewriting it blindly.

# 1. Project Goal

UrbanFly uses the real `HelsinkiCentral1km` photogrammetry scene as a reproducible, privileged urban low-altitude navigation environment. Its role is to provide real building geometry, constrained rooftop/ground transitions, street canyons, long routes, collision truth, and difficult local navigation situations for single-UAV research.

The research goal is not merely to find a global route. It is to establish a reliable navigation and data foundation for learned, action-conditioned local planning:

- `FROZEN` Global Planner: performs city-scale long-horizon XYZ route search.
- `FROZEN` Local Goal Selector: converts the global route into a 10–30 m receding-horizon target.
- `PLANNED` Action-Conditioned World Model: predicts local future latent/state/progress/risk under candidate action sequences.
- `PLANNED` MPPI: scores and selects local candidate action sequences using World Model predictions.
- `FROZEN` low-altitude controller: executes the chosen local motion command/route robustly.
- `FROZEN` triangle geometry: provides safety validation and evaluation truth.

The current system is a reliable infrastructure layer, not the final learned local-navigation algorithm. Its present purpose is to support a carefully audited World-Model Dataset v1 and later learned local planning without destabilizing the already qualified global navigation stack.

# 2. Current Architecture

## 2.1 Executed privileged-navigation path

```text
Helsinki 0.5 m highest-surface heightmap
    ↓ max-pool to current 5.0 m global planning grid
Frozen Global XYZ A*
    ↓
coarse global path
    ↓
line-of-sight shortcutting / validated smoothing
    ↓
triangle-level swept validation
    ↓
local repair using denser validated path candidates
    ├── valid → executable trajectory
    └── invalid → PATH_GEOMETRY_INVALID
    ↓
generic low-altitude 6-DOF controller
    ↓
execution-time triangle swept collision checks
    ↓
post-execution heightmap and triangle validation
```

The triangle mesh does not replace the 1 km² global A*. It is the precise local validation and safety layer.

## 2.2 Future learned local-navigation boundary

```text
Frozen Global Route
    ↓
Local Goal Selector
    ↓
local_goal_world / local_goal_body_flu
    ↓
PLANNED Action-Conditioned World Model + MPPI
    ↓
frozen low-level execution
    ↓
Triangle Geometry Safety Oracle
```

The World Model must not search the full 1 km city or require the complete global path as its main conditioning signal. It should use observation, local state, local goal, and candidate action sequence.

## 2.3 Coordinates and interfaces

The current navigation API convention is:

```text
World: [east x, up y, north z]
Body:  [forward, left, up] (FLU)
```

`LIMITATION` — `data/helsinki_mesh/HelsinkiCentral1km/manifest.json` describes its source-frame transform with `negative_z: north`, while `backend/engine/local_goal.py`, tests, and navigation plots label positive `z` as north. Internal planning and collision geometry are aligned and the regression is valid in that shared local frame, but the compass/sign convention has not been externally validated. Dataset v1 must resolve and document this sign convention before freezing exported coordinate metadata. Do not change frozen planner behavior merely to rename axes.

## 2.4 Key module responsibilities

- `backend/engine/planner.py` — occupancy-grid representation and frozen global XYZ A*.
- `backend/engine/helsinki_navigation.py` — Helsinki asset loading, global planning adapter, smoothing, fail-closed validation, triangle integration, local repair, and execution orchestration.
- `backend/engine/triangle_geometry.py` — R-tree-accelerated triangle surface and swept-sphere queries.
- `backend/engine/simulator.py` — generic 6-DOF execution and low-altitude controller.
- `backend/engine/helsinki_urban_sampling.py` — geometry-derived density score and spatial strata.
- `backend/engine/local_goal.py` — global-route projection and configurable arc-length lookahead target.

# 3. Frozen Components

The following modules are `FROZEN`. Reopening them requires a reproducible new failure, an asset/interface change that invalidates existing evidence, or an explicit user request.

## 3.1 Global XYZ A* and global planning logic — FROZEN

Why frozen:

- `PASS` — previous low-altitude qualification planned 200/200 tasks.
- `PASS` — candidate urban-core set planned 100/100 tasks.
- `PASS` — fresh unseen urban-core regression planned 100/100 tasks.
- `PASS` — all fresh tasks had straight-line routes blocked and disabled high-altitude fallback.

Allowed reasons to modify:

- a reproducible new `PLANNING_FAILED` case from the agreed spatial split;
- a confirmed global map-alignment or connectivity defect;
- an explicit user request to revisit global planning.

## 3.2 Smoothing and global-to-executable path generation — FROZEN

Why frozen:

- `PASS` — 100/100 fresh paths passed heightmap validation.
- `PASS` — 100/100 fresh paths passed triangle swept validation.
- `PASS` — invalid geometry is fail-closed through `PATH_GEOMETRY_INVALID`.

Allowed reasons to modify:

- a reproducible smoothed-path geometry failure that cannot be handled by the current repair path;
- a new trajectory continuity requirement with a regression test.

## 3.3 Triangle geometry query and validation — FROZEN

Why frozen:

- `PASS` — point, segment, and trajectory APIs exist and have focused tests.
- `PASS` — building, cavity, overhang-like, and thin-structure audits completed.
- `PASS` — 178,720 planned-path triangle samples and 879,399 executed-path triangle samples were evaluated in the final regression records.

Allowed reasons to modify:

- new mesh assets or a reproducible false positive/negative;
- a measured throughput blocker in Dataset v1;
- a requirement for a watertight or semantic obstacle representation.

## 3.4 Generic low-altitude controller — FROZEN

Why frozen:

- `PASS` — all four prior controller failures now succeed.
- `PASS` — fresh unseen regression has 0/100 controller failures and 0/100 ceiling violations.

Allowed reasons to modify:

- a new reproducible controller failure under the same statistics and unchanged ceiling;
- a new low-level command interface with dedicated regression coverage.

## 3.5 Urban-core sampler — FROZEN

Why frozen:

- `PASS` — candidate mix is exactly 75/15/10.
- `PASS` — geometry-derived density materially increased obstacle coverage relative to the previous 200 tasks.
- `PASS` — the candidate overview and quantitative checks exist.

Allowed reasons to modify:

- spatial-split design proves a coverage hole;
- Dataset v1 audit shows unacceptable sampling bias;
- scene geometry or map extent changes.

## 3.6 Local Goal interface — FROZEN

Why frozen:

- `PASS` — configurable 10–30 m lookahead and required outputs are implemented.
- `PASS` — 3/3 focused tests pass.

Allowed reasons to modify:

- Dataset v1 schema exposes a missing required signal;
- a new learned-policy coordinate contract is explicitly approved.

## 3.7 Prohibited modification patterns

- testcase-ID hacks;
- task-specific controller/planner branches;
- special handling for the old task indices 38, 79, 121, or 133;
- weakening statistics, ceilings, or collision criteria to pass a benchmark;
- redesigning a frozen module without new failure evidence;
- changing planner, controller, sampler, and World Model simultaneously.

# 4. Triangle Geometry Status

```text
mesh source:
D:\AI\UrbanFly\data\helsinki_mesh\HelsinkiCentral1km\collision\HelsinkiCentral1km_collision_L18.glb

triangles: 307,980
vertices: 301,037
watertight: false
acceleration structure:
trimesh triangles_tree / libspatialindex R-tree BVH/AABB index

is_collision: PASS
segment_collision: PASS
trajectory_collision: PASS
distance query: PASS
distance unit: meter
distance semantics: unsigned surface distance
```

Implementation: `backend/engine/triangle_geometry.py`

The simulator hot path uses the R-tree swept-sphere AABB broad phase and exact point-to-candidate-triangle distances. It does not scan all 307,980 triangles per query.

## 4.1 Geometry audit

Building:

```text
PASS
samples: 300
surface error <= 2 m: 96.7%
median heightmap-surface to triangle distance: 0.434 m
P95 heightmap-surface to triangle distance: 1.690 m
```

Bridge-like geometry:

```text
PASS
geometry cavity candidates: 91
upper structure: occupied
space below structure: free when triangle geometry supports the cavity
LIMITATION: no semantic bridge labels in the source asset
```

Overhang-like geometry:

```text
PASS
vertical cavity candidates: 126
upper structure surface and free space below are distinguishable
LIMITATION: no semantic overhang labels in the source asset
```

Thin structure:

```text
PASS
heightmap-miss candidates detected by triangle query: 20
```

Tree/vegetation:

```text
LIMITATION
TREE COLLISION COMPLETENESS NOT GUARANTEED
```

Critical geometry limitation:

```text
LIMITATION: collision mesh is non-watertight.
LIMITATION: arbitrary deep-inside structure classification is not guaranteed.
PASS: swept validation from a known free-space trajectory is reliable for the current navigation use case.
```

Audit evidence: `outputs/helsinki_triangle_geometry_audit.json`

# 5. Controller Status

The controller change is generic. There is no testcase-specific or task-ID-specific branch.

Generic robustness changes:

- vertical velocity saturation;
- vertical acceleration limiting;
- ceiling-aware and floor-aware braking;
- vertical damping;
- final output height safety constraint.

Regression status:

```text
before: 4 / 200 failures
failed task indices: 38, 79, 121, 133

targeted replay after fix: 0 / 4 failures
fresh unseen regression: 0 / 100 failures

vertical divergence: PASS — eliminated in the targeted replay and fresh regression
targeted ceiling violations: 0 / 4
unseen ceiling violations: 0 / 100
targeted collisions: 0 / 4
unseen collisions: 0 / 100

targeted tracking RMSE mean: 0.811 m
unseen-100 tracking RMSE mean: 0.783 m
```

Evidence: `outputs/helsinki_controller_failure_replay_after_fix/report.json`

# 6. Urban-Core Sampler Status

Implementation: `backend/engine/helsinki_urban_sampling.py`

Urban density score is a weighted combination of:

- building coverage within a 75 m neighborhood;
- obstacle/non-ground coverage within a 75 m neighborhood;
- normalized local surface height;
- street/free-space mixture.

Dense-core mask constraints:

- high geometry-derived density score;
- sufficient free-space corridor;
- water and pure open areas excluded using obstacle coverage;
- overly occupied cells excluded;
- tiny connected selections removed;
- map boundary distance constraint enforced.

```text
edge exclusion: 100 m
derivation: clipped 10% of effective 1 km map width, constrained to 80–120 m

candidate task mixture:
dense core: 75%
peripheral/mixed: 15%
cross-city/long-range: 10%
```

Candidate-100 statistics:

```text
mean local obstacle density: 0.498
old-200 mean: 0.343

median local obstacle density: 0.537
old-200 median: 0.320

mean distance to map boundary: 281.0 m
old-200 mean: 216.1 m

mean blocking obstacles: 1.13
mean path length: 354.7 m
P95 path length: 761.7 m
mean turn count: 3.39
inferred old dense-core task ratio: 1.5%
```

`PASS` — the new sampler resolved the previous task distribution's excessive concentration in open, coastal, and peripheral areas.

Overview: `outputs/helsinki_urban_core_regression/urban_core_100_tasks_overview.png`

# 7. Final 100-Task Regression

```text
regression random seed: 20260930
candidate random seed: 20260830
```

The regression seed differs from the candidate set and the old debug set.

Task composition:

```text
building_blocked: 20
street_canyon: 20
rooftop_to_ground: 20
ground_to_rooftop: 20
rooftop_to_rooftop: 20
total: 100

dense core: 75
peripheral/mixed: 15
cross-city: 10
```

Constraint checks:

```text
straight-line route blocked: 100 / 100
high-altitude fallback forbidden: 100 / 100
high-altitude escape used: 0 / 100
invalid path: 0 / 100
timeout: 0 / 100
```

Final metrics:

```text
Planning success: 100 / 100
Triangle validation success: 100 / 100
Execution success: 100 / 100
Collision: 0 / 100
Controller failure: 0 / 100
Ceiling violation: 0 / 100
Minimum planned triangle clearance: 4.036 m
Minimum executed triangle clearance: 4.036 m
Minimum heightmap clearance: 4.026 m
Tracking RMSE mean: 0.783 m
Tracking RMSE maximum: 0.857 m
```

Validation command:

```powershell
python scripts\verify_helsinki_urban_core_regression.py --regression-only
```

Evidence:

- `outputs/helsinki_urban_core_regression/unseen_100/summary.json`
- `outputs/helsinki_urban_core_regression/unseen_100/records.json`
- `outputs/helsinki_urban_core_regression/unseen_100/tasks.json`
- `outputs/helsinki_urban_core_regression/unseen_100/paths/` — 100 per-task NPZ files.

# 8. Local Navigation Interface

Implementation: `backend/engine/local_goal.py`

```text
Frozen Global Route
    ↓
Local Goal Selector
    ↓
PLANNED future Action-Conditioned World Model + MPPI
```

Inputs:

```text
current_position_world
current_velocity_world
yaw_degrees
global_path
lookahead_distance_m
```

Lookahead:

```text
allowed: 10–30 m
default: 20 m
selection: arc length along projected global route
```

Outputs:

```text
local_goal_world
local_goal_body_flu
remaining_global_path
route_progress_m
remaining_distance_m
```

```text
PASS: Local Goal tests 3 / 3
```

# 9. Current Verification Verdict

```text
READY FOR AUTOMATIC WORLD-MODEL DATA COLLECTION
```

Scope of this verdict:

- `PASS` — the frozen Helsinki navigation expert, geometry oracle, controller, sampler, and Local Goal interface are ready to support a small audited Dataset v1 pilot.
- `NOT TESTED` — this is not a claim that the legacy World Model models are ready for Helsinki training.
- `NOT TESTED` — this is not a claim that a 500–1000 episode Helsinki pilot has already been collected.

Test status:

```text
focused new/relevant tests: 11 / 11 PASS
full repository test run: 39 PASS, 1 FAIL
```

The one unrelated failure is:

```text
tests/test_citygs_collision_field.py::test_citygs_collision_artifacts_are_closed_and_metric
missing external asset:
data/citygs_collision/Residence/collision_geometry.json
```

`LIMITATION` — this failure is unrelated to the current Helsinki navigation work.

# 10. Current Research Decision

The current Helsinki infrastructure phase ends here.

The following remain `FROZEN`:

- Global XYZ A*;
- global planning logic;
- smoothing;
- triangle geometry query/validation;
- generic low-altitude controller;
- urban-core sampler;
- Local Goal interface.

The next phase must not immediately:

- expand or integrate V-JEPA;
- expand or integrate Dreamer;
- expand or integrate TD-MPC2;
- expand or integrate AeroVLA;
- start formal large-scale collection;
- modify planner, controller, and World Model simultaneously.

`PASS` — legacy/prototype code for several of these models exists in the repository.  
`NOT TESTED` — those prototypes are not validated as the current Helsinki learned-navigation solution.  
`PLANNED` — the next research target is World-Model Dataset v1, beginning with a small and audited 500–1000 episode pilot.

The pilot's purpose is to validate:

- dataset schema;
- data integrity and timestamp alignment;
- spatial coverage;
- action distribution;
- clearance distribution;
- future-state transition distribution;
- storage throughput;
- collection throughput.

# 11. Planned World-Model Dataset v1

Status: `PLANNED` / `NOT TESTED` on the frozen Helsinki stack.

Suggested rollout mixture:

```text
approximately 70% expert rollout
approximately 20% constrained action perturbation rollout
approximately 10% hard-case rollout
```

These are research targets, not completed collection statistics.

Hard cases should emphasize:

- bridge-like cavities;
- overhang-like cavities;
- thin structures;
- narrow corridors;
- ceiling/floor constrained regions;
- near-obstacle trajectories.

The final 100-task regression has a minimum triangle clearance of 4.036 m. This is strong safety evidence but may be too clean for an action-conditioned World Model. Dataset v1 must inspect and deliberately shape the clearance distribution without biasing all rollouts toward collisions.

## 11.1 Existing collector reality

- `PASS` — generic automatic HDF5 v2 collection code exists at `uav_wm_navigation/scripts/collect_data.py` and `uav_wm_navigation/src/uav_wm_navigation/data/collector.py`.
- `PASS` — it exposes `--num-episodes`, `--seed`, `--output`, `--max-steps`, and `--collection-mode` with `expert`, `perturbed_expert`, and `safe_exploration`.
- `PASS` — the existing mock smoke completed 5/5 episodes, reset 4 times, passed HDF5 readback, and reported valid `action_t → state_t+1` alignment.
- `LIMITATION` — the smoke teacher was `MockCandidatePlanner`, success rate was 0.2, and the scene was mock rather than Helsinki.
- `NOT TESTED` — the collector has not been requalified with the frozen Helsinki expert, Helsinki start/goal sampler, triangle clearance, Local Goal interface, and spatial split.
- `PLANNED` — reuse or extend the existing collector after a schema/integration audit. Do not rewrite it merely because the Helsinki Dataset v1 integration is incomplete.

Existing mock evidence: `uav_wm_navigation/outputs/collector_reality_smoke5_mock/collection_summary.json`

# 12. Proposed Dataset Schema

Status: `PLANNED`. The schema below is a Dataset v1 proposal and is not claimed as a completed Helsinki schema.

Timestep-level fields should include:

```text
observation_t
rgb_t                     if available
depth_t                   if available

position_world_t
velocity_world_t
yaw_t

local_goal_world_t
local_goal_body_flu_t
route_progress_m_t
remaining_distance_m_t

action_t

next_position_world
next_velocity_world

triangle_clearance_m
heightmap_clearance_m

collision_flag
ceiling_violation_flag

task_id
episode_id
timestamp
step_index
```

Required episode/scene metadata should include at least:

- scene identifier and asset version;
- coordinate-frame convention and units;
- task type and spatial stratum;
- start, goal, altitude constraints, and termination reason;
- random seed and rollout category;
- expert/controller/config version;
- observation calibration and timestamp semantics;
- dataset schema version.

The learned model should primarily condition on:

```text
observation
state
local goal
action
```

It should not depend directly on the complete Global A* path.

# 13. Dataset Split Strategy

Status: `PLANNED`.

A random task split alone is insufficient because all tasks originate from the same Helsinki 1 km region. A seed change does not demonstrate spatial generalization.

Dataset v1 must define geometry-based spatial regions:

```text
train regions
validation regions
held-out test regions
```

Requirements:

- no route leakage across region boundaries without an explicit crossing policy;
- held-out regions should contain representative building density and task types;
- image/depth appearance leakage must be considered, not only start/goal IDs;
- splits must be persisted in a manifest before model training.

`NOT TESTED` — no spatially held-out learned-model evaluation has been completed.

# 14. Planned Learned Navigation Architecture

Status: `PLANNED` / `NOT TESTED` in the current Helsinki pipeline.

```text
Frozen Global A*
    ↓
Global Route
    ↓
Local Goal
    ↓
Action-Conditioned World Model
    ↓
MPPI
    ↓
frozen low-level execution
    ↓
Triangle Geometry Safety Oracle
```

Existing World Model and MPPI prototype files in `uav_wm_navigation` are historical/prototype code. Their presence is not evidence that this architecture is integrated or qualified on the current Helsinki stack.

The first Helsinki version should not begin with Dreamer. It should test whether, given:

```text
observation_t
state_t
local_goal_t
candidate action sequence
```

the model can predict:

```text
future latent
future ego state
route progress
goal distance
collision risk
clearance / safety cost
```

Only after Dataset v1 and this prediction contract pass should more complex architectures be considered.

# 15. Initial MPPI Cost Design

Status: `PLANNED` proposal; not integrated or qualified on Helsinki.

```text
J =
    goal progress term
  + route deviation penalty
  + control smoothness penalty
  + collision risk penalty
  + low-clearance penalty
```

Responsibility split:

- `FROZEN` Global A*: long-horizon city-scale guidance.
- `PLANNED` World Model + MPPI: 10–30 m receding-horizon local decisions.
- `FROZEN` Triangle Geometry: safety and evaluation oracle.

# 16. Performance / Throughput Items To Measure

The following must be measured before scaling Dataset v1:

```text
episode wall time
simulation FPS
simulation real-time factor
triangle queries per second
dataset write throughput
average episode size
average trajectory length
```

Current evidence:

- `PASS` — the complete navigation/audit work was reported as 52 min 44 s.
- `PASS` — the final `--regression-only` report records 670.55 s for task generation plus the unseen-100 planning/validation/execution workflow.
- `LIMITATION` — neither number is single-episode collector throughput; both include work outside dataset writing.
- `NOT TESTED` — formal Helsinki collection FPS, HDF5 write throughput, average Dataset v1 episode size, and sustained collector throughput.

# 17. Important Files

## 17.1 Assets and configuration

- `data/helsinki_mesh/HelsinkiCentral1km/manifest.json` — Helsinki scene source, frame transform, visual tiles, collision mesh, heightmap, and diagnostic assets.
- `data/helsinki_mesh/HelsinkiCentral1km/collision/HelsinkiCentral1km_collision_L18.glb` — real triangle collision mesh used by the local oracle.
- `data/helsinki_mesh/HelsinkiCentral1km/diagnostics/heightmap_0p5m.npz` — 0.5 m highest-surface collision heightmap.
- `backend/config.py` — active `HELSINKI_NAVIGATION` parameters, including 5.0 m planning resolution and safety margins.

## 17.2 Navigation implementation

- `backend/engine/planner.py` — `FROZEN` occupancy-grid and global XYZ A* implementation.
- `backend/engine/collision.py` — heightmap collision-map implementation.
- `backend/engine/helsinki_navigation.py` — `FROZEN` Helsinki planning, smoothing, validation, local repair, and execution stack.
- `backend/engine/triangle_geometry.py` — `FROZEN` R-tree triangle query and swept validation API.
- `backend/engine/simulator.py` — `FROZEN` simulator and generic low-altitude controller.
- `backend/engine/helsinki_urban_sampling.py` — `FROZEN` urban density, dense-core mask, and spatial strata.
- `backend/engine/local_goal.py` — `FROZEN` global-route to local-goal interface.

## 17.3 Verification scripts

- `scripts/verify_helsinki_low_altitude_expert.py` — low-altitude task generation and per-task qualification metrics.
- `scripts/audit_helsinki_triangle_geometry.py` — building/cavity/overhang-like/thin/tree triangle audit.
- `scripts/verify_helsinki_urban_core_regression.py` — candidate overview and fresh unseen-100 regression.
- `scripts/replay_helsinki_controller_failures.py` — targeted replay of original failures 38, 79, 121, and 133 using the generic controller.

## 17.4 Focused tests

- `tests/test_triangle_geometry.py` — point distance, swept segment, and trajectory-free-path tests.
- `tests/test_local_goal_selector.py` — arc-length lookahead, yaw/body transform, and range validation tests.
- `tests/test_multirotor_dynamics.py` — existing low-level dynamics tests included in the 11-test focused run.

## 17.5 Key outputs

- `outputs/helsinki_triangle_geometry_audit.json` — triangle mesh metadata and geometry audit evidence.
- `outputs/helsinki_controller_failure_replay_after_fix/report.json` — 4/4 targeted replay evidence.
- `outputs/helsinki_urban_core_regression/report.json` — consolidated density, candidate, old-distribution, and unseen-regression report.
- `outputs/helsinki_urban_core_regression/urban_core_100_tasks_overview.png` — heightmap, density/core mask, boundary exclusion, and 100 candidate paths.
- `outputs/helsinki_urban_core_regression/candidate_tasks.json` — candidate-100 task definitions.
- `outputs/helsinki_urban_core_regression/candidate_records.json` — candidate-100 planning records.
- `outputs/helsinki_urban_core_regression/candidate_paths.npz` — candidate-100 planned paths.
- `outputs/helsinki_urban_core_regression/unseen_100/summary.json` — final regression summary.
- `outputs/helsinki_urban_core_regression/unseen_100/records.json` — all per-task regression metrics.
- `outputs/helsinki_urban_core_regression/unseen_100/tasks.json` — fresh task definitions.
- `outputs/helsinki_urban_core_regression/unseen_100/paths/` — 100 planned/executed trajectory files.

## 17.6 Existing collector and learned-model prototypes

- `uav_wm_navigation/scripts/collect_data.py` — generic unattended HDF5 v2 collector CLI.
- `uav_wm_navigation/src/uav_wm_navigation/data/collector.py` — per-episode collection and synchronized transition writer path.
- `uav_wm_navigation/outputs/collector_reality_smoke5_mock/collection_summary.json` — five-episode mock-only collector smoke evidence.
- `uav_wm_navigation/src/uav_wm_navigation/world_models/` — existing World Model prototypes; `NOT TESTED` as the approved Helsinki learned-local-navigation model.
- `uav_wm_navigation/src/uav_wm_navigation/planners/mppi.py` — existing MPPI prototype; `NOT TESTED` in the frozen Helsinki Local Goal pipeline.

# 18. Reproduction Commands

Do not rerun expensive commands automatically. Inspect existing outputs first.

Triangle geometry audit:

```powershell
python scripts\audit_helsinki_triangle_geometry.py
```

Targeted controller failure replay:

```powershell
python scripts\replay_helsinki_controller_failures.py
```

Candidate-100 generation and overview only:

```powershell
python scripts\verify_helsinki_urban_core_regression.py --overview-only
```

Fresh unseen-100 regression using saved candidates:

```powershell
python scripts\verify_helsinki_urban_core_regression.py --regression-only
```

Focused tests:

```powershell
python -m pytest tests\test_triangle_geometry.py tests\test_local_goal_selector.py tests\test_multirotor_dynamics.py -q
```

Last result: `11 PASS`.

Full repository tests:

```powershell
python -m pytest -q
```

Last result: `39 PASS`, `1 FAIL` due to the unrelated missing Residence CityGS asset.

Syntax check used for the final navigation additions:

```powershell
python -m py_compile backend\engine\triangle_geometry.py backend\engine\helsinki_navigation.py backend\engine\simulator.py backend\engine\helsinki_urban_sampling.py backend\engine\local_goal.py scripts\verify_helsinki_low_altitude_expert.py scripts\audit_helsinki_triangle_geometry.py scripts\verify_helsinki_urban_core_regression.py scripts\replay_helsinki_controller_failures.py
```

# 19. Known Limitations

## 19.1 Engineering limitations

1. `LIMITATION` — the collision mesh is non-watertight; arbitrary deep-inside occupancy classification is not guaranteed.
2. `LIMITATION` — `TREE COLLISION COMPLETENESS NOT GUARANTEED` because no semantic vegetation inventory exists.
3. `LIMITATION` — bridge-like and overhang-like cavities are geometrically detected but have no source semantic labels.
4. `LIMITATION` — one unrelated Residence CityGS test is missing `data/citygs_collision/Residence/collision_geometry.json`.
5. `LIMITATION` — no `.git` metadata exists at the workspace root, so repository cleanliness and exact diffs cannot be proven.
6. `LIMITATION` — source-manifest `negative_z: north` and navigation API `north z` labels require coordinate-sign resolution before Dataset v1 metadata is frozen.

## 19.2 Dataset limitations

1. `LIMITATION` — current successful trajectories may be too clean for action-conditioned World Model learning; minimum triangle clearance is 4.036 m.
2. `LIMITATION` — the existing five-episode unattended smoke is mock-only and uses `MockCandidatePlanner`.
3. `NOT TESTED` — existing collector integration with the frozen Helsinki expert, triangle clearance, urban-core sampler, and Local Goal interface.
4. `NOT TESTED` — 500–1000 episode Helsinki Dataset v1 pilot.
5. `NOT TESTED` — Helsinki collection throughput, storage throughput, and long-run reset reliability.

## 19.3 Research limitations

1. `LIMITATION` — unseen-100 uses a new random seed, not a spatially held-out learned-model test.
2. `NOT TESTED` — action-conditioned future prediction on Helsinki observations.
3. `NOT TESTED` — MPPI scoring with a learned Helsinki World Model.
4. `NOT TESTED` — spatial generalization to held-out Helsinki regions or another city.

# 20. Next Milestone

The next formal milestone is:

```text
World-Model Dataset v1 + 500–1000 Episode Pilot
```

Status: `PLANNED`. Do not implement it in the handoff session.

Recommended order:

1. Phase 1 — define and freeze Dataset v1 schema.
2. Phase 2 — audit and adapt the existing automatic collector to the frozen Helsinki stack; do not rewrite it without cause.
3. Phase 3 — implement persistent spatial train/validation/test partitions.
4. Phase 4 — implement and label rollout categories: expert, perturbed, hard-case.
5. Phase 5 — run a small 500–1000 episode pilot only after a new 5-episode Helsinki smoke passes.
6. Phase 6 — audit spatial coverage, actions, velocities, clearance, future 1 s/2 s/3 s transitions, data integrity, and throughput.
7. Phase 7 — only after the dataset audit passes, begin Action-Conditioned WAM v0.

# 21. Recommended First Actions for Next Codex Session

1. Read `AGENTS.md` completely.
2. Read `docs/PROJECT_STATE.md` completely.
3. Run `git status`; if `.git` is still absent, report that `LIMITATION` and do not invent a status.
4. Inspect the frozen module files listed in Section 17.
5. Inspect the latest JSON outputs before running any regression.
6. Confirm that the recorded evidence still exists and is readable.
7. Do not rerun the expensive 100-task regression without a concrete reason.
8. Do not modify the frozen Global A*, controller, triangle geometry, sampler, or Local Goal interface without new evidence or an explicit user request.
9. Focus on World-Model Dataset v1.
10. Before coding, present a concrete implementation plan for:
    - Dataset v1 schema;
    - reuse/adaptation of the existing collector;
    - Helsinki start/goal and Local Goal integration;
    - spatial split;
    - expert/perturbed/hard-case rollout categories;
    - five-episode Helsinki smoke validation;
    - 500–1000 episode pilot audit.
11. Explicitly resolve the `z`/north sign convention in metadata design without changing frozen planner behavior prematurely.
12. Distinguish existing mock/prototype code from Helsinki-qualified components.

# 22. Status Vocabulary

Use only these status terms in future handoffs:

- `PASS` — implemented/present and supported by the recorded evidence.
- `FAIL` — tested and did not satisfy the stated criterion.
- `LIMITATION` — known constraint, missing external dependency, or scope boundary.
- `NOT TESTED` — code or idea may exist, but the stated behavior has not been verified.
- `PLANNED` — agreed future work; not a completed result.
- `FROZEN` — qualified component that must not be changed without new evidence or explicit authorization.

Do not convert a recommendation, prototype, file name, class name, or existing mock result into a `PASS` claim for Helsinki. Actual code, tests, and outputs remain the final source of truth.

# 23. Dataset v1 Implementation Update (2026-08-19)

This section supersedes Sections 19–21 where they conflict. It records only
work and runtime evidence completed in the Dataset v1 session.

## 23.1 Coordinate convention and P0 fixes

- `PASS` — Dataset canonical world frame is ENU `[east, north, up]`.
- `PASS` — Dataset body/action frame is FLU `[forward, left, up]`.
- `PASS` — camera frame is RDF `[right, down, forward]`; quaternions are XYZW.
- `PASS` — Helsinki renderer/backend remains `[east, up, south]`, so manifest
  `negative_z: north` is correct. Explicit transforms are in
  `backend/engine/helsinki_frames.py`.
- `PASS` — P0 body-left sign errors in the external-action/local-goal bridge
  were fixed without redesigning the frozen planner, geometry, or controller.
- `PASS` — a newly observed P0 yaw semantic defect was fixed under explicit
  user authorization: positive FLU yaw rate now maps to negative backend yaw
  rate, and the backend persists/integrates the desired-yaw target across
  policy actions. Canonical yaw telemetry is no longer mislabeled backend yaw.
- `PASS` — the Dataset v1 validator now requires `next_state` fields and checks
  aggregate commanded-yaw versus observed-orientation direction and response.

## 23.2 Dataset v1 and spatial split

- `PASS` — factual HDF5 Dataset v1 writer and validator exist at
  `uav_wm_navigation/src/uav_wm_navigation/data/helsinki_dataset_v1.py`.
- `PASS` — deterministic train/validation/test spatial partition with guard
  bands exists at `backend/engine/helsinki_spatial_split.py`.
- `PASS` — split manifest and map:
  `outputs/helsinki_dataset_v1/spatial_split_v1.json` and
  `outputs/helsinki_dataset_v1/spatial_split_v1.png`.
- `PASS` — real browser bridge uses synchronized 160x90 raw RGB8 and uint16
  metric depth transport; HDF5 stores RGB uint8 and depth float32 meters.
- `PASS` — Dataset-only WebSocket lockstep pauses at RGB-D capture and resumes
  after the accepted external action. Actual sim `dt` is recorded.

Focused validation result:

```text
23 PASS, 1 deselected
frontend npm run build: PASS
```

## 23.3 First yaw-qualified real Helsinki episode

Two earlier real episodes are now `FAIL`, not qualified training data:

- `outputs/helsinki_dataset_v1/real_one_20260819_retry2/` — yaw-rate sign was
  reversed relative to canonical FLU; the current validator rejects it with
  `yaw_rate_orientation_consistency`.
- `outputs/helsinki_dataset_v1/real_one_yaw_fixed_20260819/` — sign was fixed,
  but desired yaw was reset on every action rather than integrated; the current
  validator also rejects it with `yaw_rate_orientation_consistency`.

`PASS` — after both P0 fixes, one complete real `building_blocked` episode was
automatically planned, executed, atomically written, and independently read
back:

```text
output: outputs/helsinki_dataset_v1/real_one_yaw_integrated_20260819/
file: HelsinkiCentral1km_real_smoke_000_building_blocked.h5
steps: 105
success: true
collision: false
path length: 143.669 m
flight time: 49.4 s
minimum triangle clearance: 5.248 m
mean/max speed: 2.897 / 3.569 m/s
mean/P95 dt: 0.470 / 0.600 s
RGB/depth/action/state counts: 105 / 105 / 105 / 105
action alignment: PASS
local-goal QA: PASS
HDF5 readback: PASS
MockCandidatePlanner used: false
yaw start/end/net: 63.435 / 110.121 / +46.686 deg ENU
commanded yaw integral: +117.094 deg
yaw command/response direction: PASS
actual body-forward mean: +2.716 m/s
backward fraction (all / second half): 0.0% / 0.0%
```

The independent validator returned `PASS` for timestamp monotonicity,
action-state temporal consistency, sensor-state alignment, RGB, metric depth,
finite values, quaternion norm, yaw-rate/orientation consistency, local-goal
progression/corridor membership, termination consistency, metadata
completeness, success consistency, and split leakage.

The real browser/backend yaw probe also returned `PASS`: 12 actions advanced
5.5 s of actual simulation time; commanded yaw integral was +247.5 deg,
observed ENU yaw change was +157.5 deg, response ratio was 0.636, and positive
executed-yaw fraction was 100%. The actual timestamps, not nominal action
duration, are the authoritative integration interval.

Focused yaw/Dataset regression result after the fixes:

```text
10 PASS
```

Visual audit artifacts generated directly from that HDF5 episode:

- `PASS` — `outputs/helsinki_dataset_v1/real_one_yaw_integrated_20260819/qa_overview.png`
  contains the Helsinki map, route/executed path, timelines, local goals,
  commanded/executed actions, and synchronized start/middle/end RGB-D.
- `PASS` — `outputs/helsinki_dataset_v1/real_one_yaw_integrated_20260819/dataset_rgbd_telemetry.mp4`
  is a 105-frame, 49.4 s RGB + metric-depth + telemetry replay reconstructed
  directly from the HDF5 arrays. OpenCV readback verified the first, middle,
  and final frames; resolution is 960x420.
- `LIMITATION` — this MP4 is a Dataset replay, not a browser UI screen recording.
- `PASS` — a standalone A-I handoff report suitable for a fresh GPT session is
  available at `docs/GPT_HANDOFF_DATASET_V1_20260819.md`.

## 23.4 Current limitations and next milestone

- `LIMITATION` — five consecutive real episodes have not yet completed; one
  diagnostic episode is not the required five-episode acceptance run.
- `LIMITATION` — measured transition frequency is variable (mean 0.470 s,
  P95 0.600 s in the yaw-qualified episode), although every actual `dt` is
  recorded and alignment passed.
- `NOT TESTED` — automatic reset reliability across five consecutive episodes.
- `NOT TESTED` — full five-episode visualization QA.
- `LIMITATION` — no `.git` directory exists, so clean/dirty status and exact
  repository diff provenance remain unavailable.

Next milestone: run the five deterministic real Helsinki task types in one
collector process, validate every HDF5 file, generate the QA visualization,
then decide readiness for 50-episode Dataset QA. Do not start 50 episodes
before the five-episode run passes.

# 24. Continuous Smoke5 Session Update (2026-08-22)

This section records the work actually completed in the new-session smoke5
attempt. The five-episode acceptance gate has not yet run.

## 24.1 Handoff and regression verification

- `PASS` — `AGENTS.md`, this file, and
  `docs/GPT_HANDOFF_DATASET_V1_20260819.md` were read completely.
- `LIMITATION` — `.git` is still absent; clean/dirty state and a Git diff are
  unavailable.
- `PASS` — the handoff's HDF5, JSON, PNG, MP4, Helsinki mesh/heightmap, and
  focused test files exist. The HDF5/PNG/MP4 SHA-256 values exactly match the
  recorded handoff values.
- `PASS` — independent readback of the existing yaw-qualified HDF5 returned
  every Dataset v1 integrity check true.
- `PASS` — the required focused regression command returned `10 passed in
  2.80s` before the new reset fix.

## 24.2 Cross-episode reset bug and fix

- `FAIL` (pre-fix audit) —
  `UrbanFlyWebSocketAdapter.reset()` cleared `_action_ack` but did not reset
  `_action_ack_step`. Because policy step ids restart at zero each episode,
  episode N+1 step 0 could satisfy its ACK wait using episode N's final ACK and
  read a stale `accepted_sim_time`.
- `PASS` (code fix) — reset now sets `_action_ack_step = -2`, clears the ACK,
  zeros velocity/acceleration history, and removes the cached canonical state.
- `PASS` — a focused regression test reproduces and checks this state reset.
- `PASS` — post-fix syntax check plus the adapter/coordinate/yaw/Dataset tests
  returned `13 passed in 1.86s`.
- `PASS` — the continuous collector now records commanded/executed counts,
  next-state counts, yaw start/end/net/integral, stale-action count, per-file
  integrity checks, process/connection identity, and explicit reset evidence
  for every episode boundary. Its acceptance result now includes yaw QA and
  reset-transition QA.

## 24.3 Deterministic task preparation and real runtime

- `PASS` — `--prepare-only` prepared all five required task types using the
  frozen bounded XYZ A*, Helsinki triangle validation, urban sampler, and
  train spatial split. Manifest:
  `outputs/helsinki_dataset_v1/real_smoke5_yaw_qualified_20260822/smoke_tasks.json`.
- `PASS` — backend startup reported the `2001 x 2001 @ 0.50 m` Helsinki height
  surface and the `200 x 19 x 200 @ 5.00 m` fail-closed global planner.
- `PASS` — the WebGL page visibly reached full Helsinki load and reported
  `307,980` triangle faces, BVH, `0.5 m` ESDF, and backend connected.
- `FAIL` — the runtime readiness probe timed out after 20 s waiting for the
  first post-reset synchronized RGB-D packet. Therefore
  `REAL_HELSINKI_RUNTIME_READY = false` and the five-episode collector was not
  started.
- `LIMITATION` — the in-app browser session disconnected after the visual
  check, and browser security policy blocked further local-page inspection.
  The user must keep `http://127.0.0.1:5173/` open in Chrome/browser before the
  readiness probe and smoke5 collection can continue.

Current verdict remains `NOT READY FOR 50-EPISODE DATASET QA`.

## 24.4 Completed continuous real smoke5 acceptance

This subsection supersedes the runtime-false and verdict statements in 24.3.
After the user kept the WebGL page open, a second readiness probe passed and
the formal acceptance run completed.

- `PASS` — `REAL_HELSINKI_RUNTIME_READY = true`: synchronized RGB was uint8
  `90 x 160 x 3`, metric depth was float32 `90 x 160`, state was ENU 6-DOF,
  the first action was new step 0 and non-stale in the readiness probe, and
  `state_t=0.3 <= action_t=0.3 < state_t+1=0.7` with actual `dt=0.4 s`.
- `FAIL` / isolated evidence — the first formal attempt was intentionally
  interrupted after two episodes because the newly added reset report
  incorrectly treated any within-episode timeout-hover label as cross-episode
  stale-action inheritance. Its files are preserved at
  `outputs/helsinki_dataset_v1/real_smoke5_yaw_qualified_20260822_failed_reset_gate_v1/`
  and are not part of acceptance.
- `PASS` — reset QA was corrected to use the synchronized new policy step id.
  A command can factually expire before a slow RGB-D capture and be labelled
  stale within the same transition; cross-episode inheritance is excluded by
  the reset ACK plus synchronized new `step_id=0` packet.
- `PASS` — the accepted run started again from episode 0 in one collector
  process and completed 5/5 episodes with exit code 0 and `all_pass=true`.
  Collector PID was `30692`; the same connection thread id
  `1795604516064` persisted through all five episodes.

Accepted output directory:

`outputs/helsinki_dataset_v1/real_smoke5_yaw_qualified_20260822/`

| Ep | Task | Steps | Success | Collision | Min clearance (m) | Mean/P95 dt (s) | Alignment | Local Goal | Yaw | HDF5 |
|---:|---|---:|---|---|---:|---:|---|---|---|---|
| 0 | building_blocked | 118 | true | false | 5.484 | 0.397 / 0.600 | PASS | PASS | PASS | PASS |
| 1 | street_canyon | 47 | true | false | 5.944 | 0.421 / 0.570 | PASS | PASS | PASS | PASS |
| 2 | rooftop_to_ground | 125 | true | false | 5.160 | 0.413 / 0.600 | PASS | PASS | PASS | PASS |
| 3 | ground_to_rooftop | 198 | true | false | 5.757 | 0.364 / 0.600 | PASS | PASS | PASS | PASS |
| 4 | rooftop_to_rooftop | 213 | true | false | 4.774 | 0.398 / 0.600 | PASS | PASS | PASS | PASS |

- `PASS` — 0->1, 1->2, 2->3, and 3->4 all passed state,
  controller, yaw, Local Goal, action-buffer/new-step, timestamp, writer-flush,
  goal-switch, and connection-persistence reset checks.
- `PASS` — independent post-run validation reopened all five HDF5 files and
  returned every Dataset v1 integrity check true. Total transitions: 701.
  There are no `.partial` files in the accepted output directory.
- `PASS` — final required focused tests returned `10 passed in 0.95s`.
- `PASS` — `qa_overview.png` and `dataset_rgbd_telemetry.mp4` were generated
  from the new street-canyon HDF5. The MP4 has 47 frames, 960 x 420,
  2.374 fps, and 19.8 s Dataset sim-time duration. First/third/two-third/final
  frames decoded successfully and were visually inspected.
- `PASS` — complete independent evidence is recorded in
  `outputs/helsinki_dataset_v1/real_smoke5_yaw_qualified_20260822/independent_qa_report.json`.
- `LIMITATION` — actual transition timing remains variable and slower than
  nominal 10 Hz. Across the five accepted files, 86/701 transitions (12.27%)
  carry explicit within-episode `stale_action` timeout-hover labels. These are
  factual labels with aligned executed actions, not cross-episode inheritance;
  their distribution must be audited in the 50-episode Dataset QA.
- `LIMITATION` — `.git` remains absent.

Final verdict: `READY FOR 50-EPISODE DATASET QA`. Do not start 50 episodes
without explicit user confirmation.

# 25. Helsinki Dataset 500-Route Run and Video Export (2026-08-25)

This section records only work actually completed. The 500-episode collection
is not complete.

- `PASS` — a reviewed 500-route manifest was prepared with exactly 100 routes
  for each of the five task types. Route coverage QA passed, with 131/144
  (90.97%) urban train cells covered and no planned triangle collisions. The
  manifest and coverage artifacts are in
  `outputs/helsinki_dataset_v1/real_500_dataset_v1_20260825/`.
- `PASS` — 35 real episodes (indices 000 through 034) were completed and
  closed as HDF5: 33 in the original collector run and 2 in a continuation
  from offset 33. All 35 completed episodes report success and no collision;
  there are zero `.partial` files across the two collection directories.
- `FAIL` / preserved evidence — collection is stopped at 35/500 after the
  local UrbanFly WebGL/backend response exceeded the fail-closed 20 s timeout
  while starting episode 035. The completed HDF5 files were preserved. The
  continuation failure evidence is at
  `outputs/helsinki_dataset_v1/real_500_dataset_v1_20260825_continuation_033_499/collection_failure.json`.
- `PASS` — all 35 completed HDF5 episodes were exported to individual RGB-D
  telemetry MP4 files in
  `outputs/helsinki_dataset_v1/real_500_dataset_v1_20260825_videos_000_034/`.
  Independent video readback found 35/35 readable files, 8,403 total frames,
  960 x 420 resolution, and exact per-episode MP4/HDF5 frame-count agreement.
  Total replay duration is 1,820.50 s (about 30 min 20.5 s); total size is
  177,437,668 bytes (0.165 GiB).
- `LIMITATION` — `.git` remains absent, so repository clean/dirty status and
  exact Git provenance remain unavailable.

Current verdict: `NOT READY` for Dataset v1 freeze because only 35/500 planned
episodes have completed. Next collection attempt must resume from episode 035;
it must not overwrite or relabel the 35 completed episodes.

# 26. User-Stopped 100-Episode Continuation (2026-08-26)

- `PASS` — local frontend/backend were restored with the Helsinki 0.5 m
  physical height surface, 5.0 m fail-closed planner, WebGL triangle BVH, and
  the browser simulation speed explicitly set to 1x.
- `PASS` — collection resumed from absolute episode 035 using the reviewed
  500-route manifest. The new output is
  `outputs/helsinki_dataset_v1/real_100_dataset_v1_20260826_continuation_035_099/`.
- `PASS` — 18 new episodes (035 through 052) completed successfully with zero
  collisions. Every completed record reports action alignment, Local Goal,
  yaw, HDF5 readback, and all Dataset integrity checks as `PASS`.
- `PASS` — the new-batch 5-episode and 10-episode checkpoint QA files passed.
  At the 10-episode checkpoint: 10/10 success, zero collisions, reset 9/9,
  minimum clearance 3.938 m, stale-action ratio 87.77%, maximum stale burst
  45, cross-episode stale inheritance zero, corrupted HDF5 zero, and partial
  files zero.
- `PASS` — an independent read-only combined audit at episodes 000 through 049
  returned 50/50 success, zero collisions, 10,074 transitions, minimum/median
  clearance 3.938/8.906 m, stale-action ratio 14.74%, maximum stale burst 53,
  and no HDF5 integrity failures.
- `FAIL` / preserved evidence — while starting episode 053, the UrbanFly
  response exceeded the fail-closed 20 s timeout. The collector stopped after
  18 new completed episodes; `collection_failure.json` preserves the failure
  and 17/17 within-process automatic reset records. No partial HDF5 remained.
- `PASS` — the user then explicitly requested that the task stop. No further
  continuation was launched. Collector process count is zero, local ports
  5173 and 8765 are closed, and the combined preserved total is 53 HDF5 files
  (episodes 000 through 052) with zero `.partial` files.
- `LIMITATION` — stale-action frequency in the 2026-08-26 continuation is much
  higher than in the earlier qualified 1x run. It was audited but not optimized;
  no frozen controller, planner, sampler, geometry, or Local Goal logic was
  modified.

Current status: `USER STOPPED AT 53/100`. If collection is resumed, restart
from absolute episode 053 using the reviewed manifest; do not overwrite or
relabel episodes 000 through 052.

# 27. Digital-Twin Runtime and UI Performance Architecture (2026-08-26)

This section records the backend/UI optimization requested after the user
stopped collection. Formal Dataset collection remains stopped at 53 completed
HDF5 files; the probes below did not write Dataset episodes.

## 27.1 Root cause and architecture change

- `FAIL` (pre-fix design) — synchronized RGB-D was initiated only from the
  frontend `requestAnimationFrame` loop. A background, hidden, or heavily
  rendering tab could therefore delay the policy data plane until the
  collector's fail-closed 20 s timeout.
- `PASS` — synchronized RGB-D is now initiated directly by each new
  `sim_state`, after sensor configuration and scene pose synchronization but
  before path and telemetry visualization. rAF now controls presentation only.
- `PASS` — RGB row conversion, metric-depth u16 quantization, and UFWM binary
  packet assembly moved to a dedicated Web Worker. The Three.js WebGL capture
  stays on its required rendering-context owner.
- `PASS` — ordinary UI clients now use one-slot latest-state coalescing. A slow
  visualization socket cannot block the 20 Hz simulator or accumulate an
  unbounded state backlog. Policy clients, control, reset, ACK, and event
  messages retain reliable ordered delivery.
- `PASS` — no Global Planner, Privileged Expert, triangle geometry, controller,
  sampler, or Local Goal core file was modified.

Implemented runtime files:

- `backend/server/runtime_metrics.py`
- `backend/server/server.py`
- `frontend/src/network.js`
- `frontend/src/drone_sensors.js`
- `frontend/src/sensor_packet_worker.js`
- `frontend/src/runtime_health.js`
- `frontend/src/index.js`
- `frontend/index.html`
- `frontend/src/style.css`
- `tests/test_server_runtime_metrics.py`
- `docs/RUNTIME_ARCHITECTURE.md`

## 27.2 Mature runtime visibility

- `PASS` — `GET /api/health` exposes simulator state, client roles, bounded
  counter/histogram windows, mean/P50/P95/P99/max latencies, state coalescing,
  packet/byte counts, and send failures.
- `PASS` — the interface has a compact CORE / SIM / RGB-D / WS / PIPE / VIEW
  health strip with RTT, buffered bytes, capture latency, packet loss,
  coalescing, and rolling 30 s browser long-task status.
- `PASS` — browser WebSocket handling now records RTT, buffered-amount
  high-water mark, traffic, rejected sends, and reconnects. Expected transport
  interruption during a backend restart is handled by the reconnect path.
- `PASS` — full Helsinki visual load reached `三角网格实景在线`; background semantic
  browser inspection found no new application error after the Worker build.
- `FAIL` / isolated development-only evidence — the Vite validation tab
  crashed after multiple hot reloads, repeated full-city GPU asset loads, two
  real probes, and a screenshot in one session. A clean production-build tab
  at `http://127.0.0.1:8765/` then cold-loaded to `三角网格实景在线` with CORE ONLINE,
  zero console warning/error, and no loading overlay after 45 s. The Vite
  server was stopped; the backend-served production UI is the operational
  runtime and remains open.

## 27.3 Actual runtime validation

- `PASS` — focused backend/runtime and external-policy tests: `8 passed in
  0.95s`.
- `PASS` — production Vite build completed with a separate 0.97 kB Worker
  asset. The main bundle is 840.39 kB minified / 227.02 kB gzip; Vite retains
  its standard >500 kB chunk warning.
- `PASS` — a real episode-053 route reset/configuration probe produced its
  first synchronized Worker-built RGB-D frame in 0.304 s total. RGB was uint8
  `90 x 160 x 3`, depth was `90 x 160`, and finite-depth ratio was 100%.
- `PASS` — a second real lockstep probe completed 20/20 consecutive actions
  with zero timeout and zero timestamp regression. Wall latency mean/P95/max
  was 0.254/0.319/0.324 s; simulation dt mean/max was 0.100/0.100 s.
- `PASS` — the browser reported 21 bridge frames and zero dropped frames. The
  post-probe server window reported simulator-step P95 0.910 ms,
  capture-to-packet P95 126.673 ms, WebSocket text-send P95 0.465 ms, and the
  UI WebSocket buffered amount was zero.
- `PASS` — both probes were diagnostic only. The simulator was explicitly
  stopped afterwards and no HDF5 or `.partial` file was created.

## 27.4 Regression status and limitations

- `PASS` — complete project test execution returned 50 passing tests.
- `FAIL` / unrelated existing asset limitation — one legacy CityGS collision
  test cannot run because
  `data/citygs_collision/Residence/collision_geometry.json` is absent. No fake
  replacement was created. The active Helsinki height surface, global planner,
  triangle BVH, and live renderer loaded successfully.
- `LIMITATION` — a full OffscreenCanvas renderer migration is not claimed. It
  requires coordinated transfer of scene ownership, asset loaders, input,
  resize, picking, and camera controls and needs its own visual regression gate.
- `LIMITATION` — `.git` remains absent, so clean/dirty status and exact Git
  provenance remain unavailable.

Current runtime verdict: `PASS` for the new local backend/data-plane/UI
architecture. There is no P0/P1 blocker in the validated runtime path. Dataset
status remains `USER STOPPED AT 53/100`; do not resume collection without a
new user request.

# 28. Integrated runtime optimization and desktop entry — 2026-08-28

This section supersedes the broad runtime verdict in section 27. User requested
continued cross-layer smoothness improvements, not collection resumption.

## 28.1 Scope and actual changes

- `FROZEN` — no changes to Global Planner, privileged expert, triangle geometry,
  controller, sampler, Local Goal core, source meshes, or dataset sensor schema.
- `PASS` — native WPF/.NET 9 + WebView2 shell is built at
  `desktop/publish/win-x64/UrbanFly.Desktop.exe`. It uses the local production
  frontend internally; the operator does not need to type a localhost URL.
  Launch/build scripts are `scripts/launch_desktop.ps1` and
  `scripts/build_desktop.ps1`. Build now refuses to overwrite a running shell.
- `PASS` — shell startup and real WebView2 city loading were exercised without
  foreground interaction. An owned Python engine started successfully; a later
  shell reused it. Backend logging owns its file descriptors independently of
  shell lifetime. The engine survived termination of both diagnostic shells.
- `PASS` (code + health parsing tests) — health schema validation, unknown policy
  counts on failed health checks, occupied-port fail-closed startup, single
  instance ownership handling, and no forced restart of a live engine on timeout.
  Closing an initialized sensor view is refused when collection is active or its
  health is unknown. A dead engine/startup failure can still be closed.
- `PASS` — WPF minimize visibility is forwarded to the presentation scheduler.
  Actual testing found WebView2 `document.hidden` alone stayed false when the
  host was minimized. The fixed host signal stops display work without stopping
  event-driven RGB-D.
- `PASS` — executed traces reuse one geometry/material and a mirrored 6000-point
  ring buffer. Path throttling happens before waypoint signature construction.
- `PASS` — policy bridge enable/disable no longer takes ownership of manual
  preview streaming. Actual bridge shutdown reported streaming=false.
- `PASS` — bounded single-flight HTTP health polling (2 s cadence, 1.5 s timeout),
  bounded per-client renderer diagnostics, bounded capture-start bookkeeping
  even if no packet arrives, and frame/CPU submission metrics.
- `PASS` — presentation-only smooth/detail modes; default smooth DPR <= 1 and
  no bloom, detail DPR <= 1.5 and lazy bloom. Collection display targets 30 FPS;
  interactive non-collection display targets 60 FPS. Targets are NOT measured
  achieved frame rates. Sensor target dimensions/lighting are unchanged.
- `PASS` — stopped, unchanged scenes render on demand. Asset arrival, input,
  quality, resize, diagnostics layers and real state updates invalidate the view.
  Static UI reached `VIEW IDLE`; native minimize reached zero display frames.
- `PASS` — texture warmup is time-sliced and shaders compiled before adding each
  tile to the displayed scene. This does not reduce triangle or texture content.
- `PASS` — fake flights are no longer shown automatically while the real engine
  is idle/offline. Legacy demonstration remains explicit via `?demo=1`.

## 28.2 Actual validation and output paths

Output root: `outputs/runtime_optimization_20260828/`.

- `PASS` — `npm test` in frontend: **11/11** tests.
- `PASS` — desktop health harness: **6/6** cases, including malformed schema and
  offline != idle. Run `dotnet run --project desktop/UrbanFly.Desktop.Tests`.
- `PASS` — backend runtime/external-policy tests: **10/10**, latest 1.91 s.
- `PASS` — final production frontend and native shell publish. Main JS
  **845.61 kB / 228.69 kB gzip**, Worker 0.97 kB. Standard Vite >500 kB warning
  remains; it was not hidden by changing the warning threshold.
- `FAIL` / existing unrelated asset limitation — full root suite:
  **52 passed, 1 failed in 13.54 s**. The failure is still missing
  `data/citygs_collision/Residence/collision_geometry.json`; no fake asset added.
- `PASS` — `trace_benchmark.json`: 3000 point CPU-only microbenchmark, five runs,
  median old allocation algorithm **107.3761 ms**, fixed buffer **0.7243 ms**.
  This is not a GPU benchmark or a claimed 148x end-to-end speedup.
- `PASS` — `desktop_hidden_probe.json`: real Helsinki, one minimized WebView2
  sensor surface, **5 resets, 100/100 hover lockstep actions**, **105 RGB-D
  frames, 0 drops, 0 timeouts, 0 timestamp regressions, 0 stale_action**.
  Action wall mean/P95/max **0.233671/0.264127/0.277923 s**; simulation dt
  **0.100/0.100/0.100 s**. Reset policy step IDs restart at zero. RGB uint8
  90x160x3, depth 90x160 finite, state timestamps aligned, quaternion normalized,
  executed action finite. Display stayed hidden/0 frames; streaming=false after
  bridge disable. Capture-to-packet P95 after this probe: **52.4735 ms**.
- `PASS` for transport integrity only — `browser_demand_probe.json`: **1 reset,
  20/20 actions**, no timeout/regression/stale_action. Wall mean/P95/max
  **0.477527/0.540330/0.587987 s**. Simulation dt mean **0.115 s**, max **0.4 s**;
  therefore this is **NOT a strict fixed-dt or collection qualification PASS**.
  Both probes were hover diagnostics, NOT completed navigation episodes.
- `PASS` — background Browser skill visual/semantic QA found real city loaded,
  quality selector functional, and no application warning/error in sampled logs.
  No OS mouse/keyboard control was used.
- `PASS` — final published frontend loaded to `VIEW IDLE`, sampled logs empty;
  `browser_idle_final.json` contains 3 passive samples with 0 display frames/FPS
  (intentional idle suspension). Temporary browser QA tab was closed afterward
  to release renderer resources; no native diagnostic shell remains running.
- `LIMITATION` — continuous full-city presentation samples before the final
  render-on-demand change: `browser_detail.json` median **17.406 FPS** and
  `browser_smooth_final.json` median **19.368 FPS** (range 15.976–19.917).
  These are background in-app browser measurements, not guaranteed native-window
  performance. Approximately **1021 draw calls**, **1026 textures** were observed.
  Static `VIEW IDLE` is intentional, not a low-FPS performance failure.
- `PASS` — no dataset write was performed. Recount: **33 + 2 + 18 = 53 HDF5**;
  `.partial=0` across the three existing collection directories. Existing success
  and collision results were not re-audited in this runtime-only session.

## 28.3 End state, limitations and next milestone

- `LIMITATION` — `.git` is absent; no Git clean/dirty status or provenance claim.
- `LIMITATION` — final guarded backend restart command was rejected by the
  execution policy and was not retried by another route. Existing backend PID
  **16788** remained online/stopped with zero policy clients. Its running build
  includes this session's main runtime work but not the final diagnostic `idle`
  field / numeric-sanitizer ordering; those changes are on disk and unit-tested,
  pending a normal backend restart. Do not assume process IDs remain current.
- `NOT TESTED` — native render-process crash injection, active-collector window
  close fault injection, visible-native sustained FPS, long-running leak soak,
  or complete five-navigation-episode qualification after these changes.
- `OPEN P1 / performance` — active full-city display still misses a stable
  30/60 FPS goal and competes with RGB-D. Browser probe jitter remains. No claim
  that all stutter is fixed or that Dataset v1 can now be frozen.
- `PLANNED` — next milestone: profile GPU/scene traversal on the visible native
  surface; implement measured presentation-only batching/LOD or renderer thread
  isolation without changing authoritative sensor geometry. Then perform a
  sustained capture+display test and real five-navigation-episode gate before
  resuming bulk collection on a new user request.

Verdict: **PARTIAL PERFORMANCE IMPROVEMENT; NOT YET FULL SMOOTHNESS QUALIFIED**.
Dataset status remains **USER STOPPED AT 53/100**.

## 29. Presentation-only city LOD and measured runtime improvement (2026-08-28)

### 29.1 Request, boundaries and completed architecture

- User requested continued integrated smoothness optimization, not renewed
  collection. No formal dataset collection was started.
- `FROZEN` — no edits to Global Planner, Privileged Expert, triangle geometry,
  controller, sampler or Local Goal core. No stale_action optimization.
- `PASS` — read project instructions/state and inspected current code/assets.
  `.git` is still absent; Git status/diff/provenance is unavailable.
- `PASS` — `frontend/src/city_display_lod.js` adds display-only tile selection.
  The existing 16 L18 overview tiles (307,980 triangles) are retained alongside
  all 16 original L21 tiles (3,825,064 triangles). Camera-to-tile-AABB distance
  enters high detail at 200 m and exits at 240 m; unchanged selections do not
  repeatedly rewrite mesh masks. Missing overview falls back to high detail.
- `PASS` — common actors/lights remain on layer 0; original city meshes always
  retain sensor layer 1; display uses layer 2. Sensor camera uses 0+1, display
  camera 0+2. Presentation selection does not change source geometry, material,
  object visibility, sensor target dimensions, lighting or navigation geometry.
  Smooth mode defaults to LOD; the new checkbox disables it for same-DPR A/B
  testing, and detail mode forces original display tiles.
- `PASS` — `render_benchmark.js` and the UI button perform 1 s warmup plus 10 s
  continuous rendering, bounded samples and real frame intervals. They do not
  start simulation. Active capture/hidden windows cancel the benchmark. Results
  include FPS, frame-interval P95, CPU submission P95, draw calls and triangles.
  Display counters now exclude earlier sensor renders and include all composer
  passes; the previous renderer counter policy is restored even on exception.
- `PASS` — `probe_runtime_pipeline.py --sensor-snapshot` optionally saves initial
  RGB/depth/state/intrinsics to a new NPZ without overwriting existing files.
- Tradeoff: retaining overview adds **19,429,816 bytes** of compressed assets on
  top of **183,741,924 bytes** for high-detail tiles; this is not a memory-saving
  change. Post-probe frontend reports 1285 textures / 1291 geometries. GPU byte
  allocation was not measured. Geometry/texture duplication remains bounded by
  the fixed tile set, but a long leak soak has not been performed.
- API design checked against official [Three.js Layers documentation](https://threejs.org/docs/pages/Layers.html)
  and installed Three r170 renderer implementation; no speculative framework
  migration or dependency upgrade was needed.

### 29.2 Actual validation and output paths

All new evidence is under `outputs/runtime_lod_20260828/`.

- `PASS` — frontend **17/17** tests, latest 284.974 ms. Includes persistent
  sensor masks, display XOR selection, unchanged geometry/material identity,
  hysteresis/fallback, benchmark timing/cancellation, and multi-pass counter
  restoration. Run `cd frontend; npm test`.
- `PASS` — backend runtime/external-policy focused tests **10/10**, latest
  **1.86 s**. Python diagnostic script compile check also passed.
- `PASS` — production frontend and native desktop publish. Main JS
  **850.29 kB / 230.22 kB gzip**, worker **0.97 kB**, CSS **16.75 kB**. Standard
  Vite >500 kB warning remains. Desktop output:
  `desktop/publish/win-x64/UrbanFly.Desktop.exe`.
- `PASS` — background Browser skill UI QA: complete city loaded, LOD checkbox
  and benchmark controls work, stopped view returns to `VIEW IDLE`. No OS
  mouse takeover. The user-opened native surface was observed closed before
  browser diagnostics; only one sensor surface was used.
- `PASS` — `display_comparison.json`, same initial camera, **1280x650**, smooth
  quality, DPR **1**, no policy capture, three 10-second runs:

  | Display | FPS | Frame interval P95 | CPU submission P95 | Median draws | Median triangles |
  |---|---:|---:|---:|---:|---:|
  | LOD on, first | 30.962 | 44.49 ms | 4.96 ms | 261 | 311,199 |
  | Original full display | 18.800 | 72.23 ms | 11.20 ms | 1021 | 3,827,783 |
  | LOD on, repeat | 29.220 | 47.37 ms | 4.70 ms | 261 | 311,199 |

  All runs retained all 16 original sensor tiles. The initial distant view
  selected 16 overview tiles / zero high display tiles. These are short
  background-browser measurements, not sustained visible-native FPS or a GPU
  timer result. Draw count drops about 74%; FPS rises about 55–65% in this view.

- `PASS` — real Helsinki runtime probes, no formal navigation episodes:

  | Report | Resets | Actions | Wall mean / P95 / max (s) | RGB-D packets |
  |---|---:|---:|---|---:|
  | `lod_on_probe.json` | 1 | 20 | 0.287684 / 0.371752 / 0.392449 | 21 |
  | `lod_off_probe.json` | 1 | 20 | 0.366361 / 0.533802 / 0.554741 | 21 |
  | `lod_on_reset5_probe.json` | 5 | 100 | 0.294319 / 0.387772 / 0.407718 | 105 |

  **7/7 resets, 140/140 actions, 147 RGB-D packets**, zero timeout, timestamp
  regression, stale_action, dropped or skipped-busy bridge frames. Every
  observed sim dt was 0.100 s within floating-point precision. Each reset's
  policy step IDs restart at zero. RGB uint8 90x160x3, finite depth 90x160,
  aligned state/frame timestamps, normalized quaternion and finite executed
  action checks all passed. Packet totals use per-probe counter deltas, not
  the backend's cumulative lifetime totals.
- `LIMITATION` — `lod_on_sensor.npz` versus `lod_off_sensor.npz` is **not a
  same-pose pixel-equivalence PASS**. Independent reset positions differ by
  up to 0.00009823 m and quaternion components by 0.00001695; timestamps and
  intrinsics agree. RGB maximum/mean channel difference is 25/0.037569;
  depth maximum/mean difference is 0.00366217/0.000242873 m. Full comparison
  recorded in `sensor_comparison.json`. Pose mismatch makes attribution
  inconclusive; no claim that all differences are caused solely by pose.
  Structural camera/layer isolation is separately tested and passes.
- `LIMITATION` — active-capture display health snapshots remained approximately
  3.66–10.15 FPS with LOD (not a continuous benchmark). The 29–31 FPS overview
  result must not be presented as simultaneous collection display performance.
  Shared WebGL context and synchronous sensor readback remain the P1 bottleneck.
- `NOT RERUN` — expensive full-root suite; section 28 records **52 passed /
  1 failed** due to missing legacy Residence collision asset. This turn's
  focused tests do not resolve that unrelated asset limitation.
- `PASS` — existing HDF5 recount **33 + 2 + 18 = 53**, `.partial=0` across the
  three formal collection directories listed in earlier sections. No HDF5 was
  created, deleted or modified by this optimization; existing navigation
  success/collision outcomes were not re-audited.

### 29.3 End state and next milestone

- `PASS` — temporary browser test tab closed after bridge disabled; final
  backend health is `stopped`, loop_running=false, **zero total/policy clients**.
  No native diagnostic shell remains open. Backend PID **16788** still serves
  the production build; recheck PIDs next session rather than assuming them.
- `LIMITATION` — section 28's blocked guarded backend restart was not retried
  or bypassed. This turn changes frontend/QA only. Final on-disk backend idle
  metadata/numeric-sanitizer ordering from section 28 still requires a normal
  restart. Published frontend changes are already served to a newly opened
  desktop window, independently of that pending backend restart.
- `NOT TESTED` — exact same-pose GPU pixel-equivalence, sustained visible-native
  FPS, memory leak soak, five complete navigation episodes after these changes.
- `PLANNED` — capture/display contention is next: profile and then isolate
  render-target readback/presentation scheduling with explicit same-pose RGB-D
  equivalence before considering a separate render-context/thread migration.
  Follow with sustained visible-native capture+display measurements and a
  five-navigation-episode gate before bulk collection on user request.

Verdict: **MEASURED OVERVIEW IMPROVEMENT; ACTIVE-CAPTURE SMOOTHNESS STILL P1**.
Formal dataset remains **USER STOPPED AT 53/100**, not ready for a new freeze
claim based on these runtime diagnostics alone.

## 30. User-authorized continuation to 100 episodes (2026-08-28)

### 30.1 Scope and actual preflight

- User explicitly requested resuming real collection to **100 total episodes**.
  This supersedes the previous user-stop condition, not the frozen algorithms
  or fail-closed Dataset QA requirements.
- `PASS` — independently reopened all **53 existing HDF5 files / 10,421
  transitions**, every integrity check passed, episode IDs are exactly 000–052,
  and all three existing directories contain zero `.partial` files. Old files
  were not overwritten or relabeled. Directory counts remain 33 + 2 + 18.
- `PASS` — reused the reviewed 500-route manifest; the new requested slice is
  **[53:100]**, 47 episodes. Combined with the existing data this targets 20
  episodes per task type. Route generation was not repeated.
- `PASS` — exactly one real native WebView2 sensor surface is running minimized
  in the background, with no OS mouse/focus takeover. Startup health confirmed
  scene_ready=true and no existing policy client. No browser test tab was opened.
- `PASS` — preflight real reset + **10 synchronized hover actions**, no stale
  actions or timing regression; dt=0.100 s. Wall mean/P95/max:
  **0.229810 / 0.259357 / 0.271294 s**. Evidence:
  `outputs/runtime_lod_20260828/precollect_100_probe.json`.
- `PASS` — requested and acknowledged simulation speed **1x** before collection.
  Display is hidden; full original high-detail RGB-D remains active.
- `LIMITATION` — no `.git` directory; no Git cleanliness/provenance claim.

### 30.2 Collector-only safety and durable job lifecycle

- `PASS` — collector now writes atomic `collection_progress.json` at episode
  start/close and terminal status, including PID, absolute episode index,
  completed records and reset evidence. This avoids relying on chat history or
  an unflushed terminal to recover the current position after internet loss.
- `PASS` — each newly closed HDF5 must immediately pass all integrity/readback
  checks, new policy step zero, and `.partial=0` before another episode starts.
  Existing automatic reset and checkpoint QA remain in force. High stale_action
  ratio alone is not a stop condition and no controller/planner was tuned.
- `PASS` — focused collector guard, Dataset v1 and frame tests: **14/14 in
  3.10 s**. Collector/job Python compile checks passed.
- `PASS` — `scripts/run_helsinki_collection_job.py` launches an independent
  local subprocess with file-owned stdout/stderr, rejects existing output
  directories, checks single ready sensor ownership, records frozen source
  hashes, and stops the simulator after collector termination when no policy
  owner remains. It does **not** retry crashes, reset failures or QA failures.
- `FROZEN` — Global Planner, expert command generation, triangle geometry,
  controller/dynamics, sampler and Local Goal core are unchanged. Per-run
  on-disk source SHA-256 values are in `job_status.json`.

### 30.3 Live continuation and recovery instructions

New output directory:

`outputs/helsinki_dataset_v1/real_100_dataset_v1_20260828_continuation_053_099/`

- Job process launched successfully: supervisor PID **29848**, collector PID
  **3580**, native sensor PID **28156**, existing backend PID **16788**.
  Always verify PID/command line before acting; these are launch-time IDs.
- Last observed at this handoff update: collector status **COLLECTING**,
  absolute episode **054**, **1 new HDF5 closed / 54 total preserved**, simulator
  time progressing, one lockstep policy client, no collector stderr output.
  **This is running work, not a claim that 100 episodes are complete.**
- First new episode **053 ground_to_rooftop**: **537 transitions**, success=true,
  collision=false, minimum clearance **4.752113 m**, stale_action **0/537**,
  independent HDF5 integrity/readback **PASS**. This is one completed navigation
  episode, not yet the five-episode checkpoint.
- Authoritative files: `job_status.json`, `collection_progress.json`,
  `collector.stdout.log`, `collector.stderr.log`. At success the job also writes
  `collection_summary.json` and `independent_collection_qa.json`; at failure
  preserve `collection_failure.json` and any `.partial` evidence.
- Checkpoints for this 47-episode continuation: after **5, 10, 25** new episodes,
  plus final independent QA at **47**. The first 5 new episodes serve as the
  post-runtime-change real-navigation checkpoint; no duplicate smoke batch is
  inserted or counted.
- Ordinary internet loss does not stop this local process. Power loss, sleep,
  closing the sole sensor surface, local runtime failure, or fail-closed QA
  can stop progress. Do not launch a second collector against the same runtime.
- `PLANNED` — after completion, independently aggregate the four run directories
  to report the actual 100-episode scale and combined stale/clearance/task QA.
  Distinguish within-process automatic resets from process-restart boundaries;
  do not claim 99 uninterrupted automatic resets across separate collectors.

Current verdict: **COLLECTION RUNNING TOWARD 100; FINAL GATE NOT YET TESTED**.

### 30.4 Explicit monitoring request and combined QA preparation

- User subsequently requested continuous monitoring until 100 episodes finish
  and final QA completes. A current-thread heartbeat was created with automation
  id **`urbanfly-100`**, name **UrbanFly 采集监控至100条**, every **2 minutes**.
  It must remain active while collection is healthy; pause it only after final
  reporting or a preserved serious fault requiring user direction. Do not create
  duplicate monitors or collectors. This is monitoring, not permission to bypass
  reset/integrity failures or change frozen algorithms.
- `PASS` — monitoring observed **57 total / 4 new completed**, new episodes
  053–056 all success=true, collision=false, stale_action=0; collector was on
  episode 057. The first five-new-episode checkpoint was not yet written at that
  observation. Consult live files rather than treating this count as current.
- `PASS` — `uav_wm_navigation/scripts/audit_helsinki_dataset_v1_runs.py` provides
  exact combined dt/clearance percentiles, per-episode/task/phase stale audit,
  contiguous unique ID checks, independent HDF5 readback and explicit process
  boundary evidence. It never relabels process restarts as automatic resets.
- `PASS` — exercised the combined auditor on the **existing 53 real episodes**:
  53/53 success, 0 collision, 10,421 transitions, min/median clearance
  **3.937682 / 8.941269 m**, dt mean/P95/max **0.283053 / 0.6 / 0.8 s**,
  all combined gates passed. Report: new run directory's
  `preexisting_053_combined_qa.json`. This is not a new 100-episode result.
- `PASS` — multi-run aggregation + collector guard tests **11/11 in 1.98 s**,
  including duplicate IDs, missing boundary step-zero evidence, corrupted files
  and partial files. These are synthetic test fixtures, not dataset episodes.
- `PASS` — collection plotting now follows the QA report's actual episode paths
  across runs, and labels the actual episode count rather than hardcoding 50.
  The 53-existing-episode overview was rendered and visually checked. The final
  version `preexisting_053_overview_multirun.png` labels automatic reset and
  process restart counts separately. It remains a pre-existing-data plot.
- Final audit command after the job terminates successfully: run
  `uav_wm_navigation/scripts/audit_helsinki_dataset_v1_runs.py` with the four
  collection directories (original 000–032, continuation 033–034, continuation
  035–052, current 053–099), `--expected-episodes 100`, and a new `--output`
  file named `combined_100_qa.json` in the current run directory. Then use
  `plot_helsinki_dataset_v1_collection_qa.py` with `--qa` pointing to that report
  and `--output combined_100_overview.png`. Do not overwrite previous evidence.
- Local scheduled monitoring requires the computer and Codex app to remain
  running; ordinary internet loss does not stop the independent local collector,
  but may delay agent follow-up. See official OpenAI documentation:
  https://learn.chatgpt.com/docs/automations?surface=app.

### 30.5 First continuation checkpoint actually passed

- Latest observed count is **58/100 total**, **5/47 new episodes** closed;
  collector is running episode **058** (the 59th overall episode).
- `PASS` — `checkpoint_005_qa.json`: **5/5 success**, **0 collision**, **2,297
  transitions**, minimum/median clearance **4.752113 / 9.266383 m**, dt
  mean/P95/max **0.100 / 0.100 / 0.100 s** within floating-point precision.
- `PASS` — **4/4 automatic resets**, **0 stale actions / maximum burst 0**,
  cross-episode inheritance 0, all HDF5/schema/readback gates true, corrupted
  HDF5 0, partial 0. This is the real five-navigation-episode post-runtime-change
  checkpoint, not the earlier hover-only probe.
- Collection and heartbeat **remain active**. No terminal completion claim;
  do not stop them merely because this checkpoint passed. Final total-100
  aggregate QA and final user report are still pending.

### 30.6 Ten-new-episode checkpoint actually passed

- Observed on **2026-08-28 at approximately 23:26–23:27 Asia/Shanghai**:
  **63/100 total**, **10/47 new episodes** closed; collector was running
  absolute episode **063** (the 64th overall episode). Consult live progress
  files for later counts; this is not a completed 100-episode claim.
- `PASS` — `checkpoint_010_qa.json`: **10/10 success**, **0 collision**,
  **4,229 transitions**, two episodes of each of the five task types.
  Minimum/median clearance **4.752113 / 8.369026 m**; dt mean/P95/max
  **0.100 / 0.100 / 0.100 s** within floating-point precision.
- `PASS` — **9/9 automatic resets**, stale actions **0/4,229**, maximum
  stale burst **0**, cross-episode inheritance **0**, corrupted HDF5 **0**,
  `.partial=0`, and every checkpoint Dataset integrity/readback gate true.
  These stale statistics describe the new ten episodes, not all old batches.
- `PASS` — supervisor and collector command-line identities verified;
  job `RUNNING`, progress `COLLECTING`, stderr empty, one lockstep policy
  client and exactly one ready hidden native desktop sensor surface.
  No process restart, mouse interaction, duplicate collector or core edit.
- Collection and existing heartbeat remain active. Next planned checkpoint:
  **25 new / 78 total**, followed by final **47 new / 100 total** independent
  combined QA. Final gate remains **NOT TESTED**.

### 30.7 Preserved fail-closed stop at 69 total

- `FAIL` / preserved evidence — on **2026-08-28 at 23:44:11
  Asia/Shanghai**, the collector exited with code 1 while beginning absolute
  episode 069. It timed out after 20.0 s waiting for the synchronized RGB-D
  packet corresponding to a policy action. The supervisor stopped the simulator
  and did not retry. This is a local UrbanFly response timeout, not a claim that
  the 100-episode target completed.
- `PASS` — **16 new episodes 053–068** closed before the failure: 16/16
  success, 0 collision, 7,004 transitions, 15/15 within-process resets,
  stale action 0, all per-file HDF5 readbacks/integrity checks true, and
  `.partial=0`. The incomplete episode 069 produced no HDF5 and was not counted.
- `PASS` — all four preserved batches independently reopen as **69 contiguous
  episodes 000–068 / 17,425 transitions**. The stop-time combined report is
  `stopped_069_combined_qa.json`; it reports 69/69 success, 0 collision,
  minimum/median clearance **3.937682 / 8.790605 m**, combined stale action
  **1,802/17,425 (10.3415%)**, maximum burst 53, all 65/65 within-run resets,
  three process-restart boundaries with fresh-start evidence, corrupted HDF5 0
  and partial 0. `stopped_069_overview.png` is the matching overview.
- `PASS` — the timeout-time backend log was copied without altering the source
  to `stopped_069_backend_log_snapshot.txt`. Frozen planner, simulator/dynamics,
  triangle geometry, sampler, Local Goal and navigation file hashes still match
  the job's recorded launch hashes. No core edit was made.
- `LIMITATION` — the earlier episode 035–052 batch contains the visible stale
  concentration: 1,779/2,018 transitions (88.16%) across those 18 episodes.
  Each label factually records `policy_timeout_hover` and an executed zero-action
  hover. It is not cross-episode action inheritance or HDF5 corruption, but it
  is poor imitation-learning signal and must remain separately auditable.

### 30.8 User-authorized clean continuation from 69

- The user explicitly requested continuing after the fail-closed stop. This is
  new retry authority; prior evidence remains immutable in the 053–099 directory.
- `PASS` — old collector, supervisor, backend and sensor processes were absent;
  a single native WebView2 desktop sensor was relaunched minimized/hidden with
  no mouse or focus takeover. Health showed exactly one ready surface, zero
  existing policy clients and a stopped simulator.
- `PASS` — new preflight reset plus 20 synchronized hover actions for task 069:
  20/20 actions, dt mean/P95/max **0.100/0.100/0.100 s**, wall action
  mean/P95/max **0.244893/0.266834/0.272782 s**, stale action 0. Evidence:
  `outputs/runtime_lod_20260829/precollect_100_resume_069_probe.json`.
- New immutable continuation directory:
  `outputs/helsinki_dataset_v1/real_100_dataset_v1_20260829_continuation_069_099/`.
  It requests **31 episodes**, absolute indices 069–099. Launch identities were
  supervisor PID 12700, collector PID 3820, native sensor PID 11692 and backend
  PID 27128; always verify command lines before acting because PIDs can change.
- `PASS` — launch health confirmed job `RUNNING`, progress `COLLECTING`,
  episode 069 active, one lockstep policy client and the single hidden sensor
  surface producing RGB-D packets. This is a start claim only: total remains
  **69 completed** until a new HDF5 closes.
- The existing heartbeat `urbanfly-100` was updated, not duplicated, to monitor
  this new directory every two minutes. Final QA must now merge **five** explicit
  collection directories and distinguish four process-restart boundaries from
  within-process automatic resets. Final 100-episode gate remains `NOT TESTED`.

### 30.9 First post-restart checkpoint actually passed

- Latest checkpoint observation: **74/100 total**, **5/31** episodes in the
  069–099 continuation closed; collector was running absolute episode 074.
- `PASS` — `checkpoint_005_qa.json`: 5/5 success, 0 collision, 2,258
  transitions, minimum/median clearance **4.987113 / 9.541469 m**, stale action
  0 and maximum burst 0.
- `PASS` — 4/4 automatic resets, corrupted/readback failures 0, `.partial=0`,
  and every checkpoint Dataset gate true. Collector/supervisor identities,
  one lockstep client and the single hidden sensor surface remained healthy.
- Collection and heartbeat remain active. This checkpoint is not the final
  100-episode result; next new-run checkpoints are 10 and 25 episodes.

### 30.10 Ten-episode post-restart checkpoint actually passed

- Latest observation: **80/100 total**, **11/31** episodes in the 069–099
  continuation closed; collector was running absolute episode 080.
- `PASS` — `checkpoint_010_qa.json`: 10/10 success, 0 collision, 5,205
  transitions, minimum/median clearance **4.581983 / 9.259938 m**, stale action
  0 and maximum burst 0.
- `PASS` — 9/9 checkpoint automatic resets, all Dataset integrity/readback
  gates true, corrupted HDF5 0 and `.partial=0`. The subsequently closed 11th
  episode also reports success, no collision, no stale action and valid HDF5.
- Collector and heartbeat remain active. Next checkpoint is 25 new episodes
  (94 total); final 100-episode combined gate remains `NOT TESTED`.

### 30.11 Twenty-five-episode post-restart checkpoint actually passed

- Latest observation: **94/100 total**, **25/31** episodes in the 069–099
  continuation closed; collector was running absolute episode 094.
- `PASS` — `checkpoint_025_qa.json`: 25/25 success, 0 collision, 11,449
  transitions, minimum/median clearance **4.581983 / 9.016129 m**. dt
  mean/P95/max **0.100009 / 0.100000 / 0.200000 s**.
- `PASS` — stale action 0 and maximum burst 0, 24/24 automatic resets,
  all Dataset integrity/readback gates true, corrupted HDF5 0 and `.partial=0`.
- Collector and heartbeat remain active for the final six episodes. The final
  combined 100-episode QA across five directories remains `NOT TESTED`.

### 30.12 Final 100-episode Dataset v1 QA actually completed

- `PASS` — the 069–099 continuation exited normally with collector code 0:
  **31/31 success**, 0 collision, 14,385 transitions, 30/30 automatic resets,
  stale action 0, corrupted/readback failures 0 and `.partial=0`. The
  supervisor stopped the simulator after the policy client disconnected.
- `PASS` — independent five-directory aggregation reopened all **100 HDF5
  episodes / 31,810 transitions**. Episode IDs are unique and contiguous
  000–099. Every task type has exactly 20 episodes and 20 successes.
- `PASS` — combined success **100/100 (100%)**, collision **0/100 (0%)**,
  clearance minimum/median **3.937682 / 8.878763 m**, transition dt
  mean/P95/max **0.159981 / 0.500000 / 0.800000 s**.
- `PASS` / audited limitation — combined stale action is **1,802/31,810
  (5.6649%)**, maximum continuous burst 53. All stale entries originate from
  earlier preserved batches, are factually recorded executed zero-action
  timeout hovers, and have no within-run or cross-run action inheritance.
  The new 47 episodes 053–099 contain stale action 0. Phase ratios are start
  5.9791%, middle 5.4136%, end 5.6008%; the distribution is not localized to
  only one trajectory phase.
- `PASS` — reset accounting is **95/95 within-process automatic resets** plus
  **four explicitly labeled process-restart boundaries**, each with fresh
  policy step zero, cleared action buffer, reset pose/speed/acceleration/yaw and
  previous writer closure. This correctly accounts for all 99 episode
  boundaries without claiming one uninterrupted collector process.
- `PASS` — independent readback verifies RGB, Depth, State, Next State,
  commanded and executed action alignment, Local Goal, yaw/quaternion,
  timestamps, finite values, termination/metadata consistency and route
  corridor integrity for every episode. Corrupted HDF5 0; `.partial` count 0.
- `PASS` — every final combined gate is true, including 100 episode count,
  balanced tasks, >=98% success, zero collision, minimum clearance >=2.5 m,
  resets, fresh process boundaries, HDF5 integrity, no partials and no
  cross-episode stale action. Report:
  `outputs/helsinki_dataset_v1/real_100_dataset_v1_20260829_continuation_069_099/combined_100_qa.json`.
- `PASS` — `combined_100_overview.png` was generated and visually inspected;
  coverage, task balance, stale concentration and summary labels are readable.
  A representative episode-069 RGB-D telemetry MP4 was reconstructed directly
  from synchronized HDF5: 568 frames, 960x420, 10 fps, 56.8 s simulation time;
  first/middle/final frames independently decoded.
- `LIMITATION` — stale-action-heavy episodes 035–052 remain in Dataset v1 and
  should retain their explicit labels or be filterable during training. The
  user-specified freeze gate did not set a maximum stale ratio, and stale action
  was explicitly audit-only; therefore this is not classified as P0/P1 blocker.
- `LIMITATION` — `.git` is absent, so no repository cleanliness or commit
  provenance claim is possible.

Final verdict: **`READY FOR DATASET V1 FREEZE`**. P0 blockers: none. P1
blockers: none. The `urbanfly-100` heartbeat can now be deleted because its
100-episode collection and final-QA objective is complete.

### 30.13 Zero-stale original-route replacement collection started

- The user explicitly requested that every stale-action-affected episode be
  recollected on its **original route** before Dataset v1 is discussed or
  frozen, and authorized removal of the superseded bad data after validation.
- Exact affected IDs from `combined_100_qa.json` are **011 and 034–052**:
  20 episodes / 1,802 stale transitions. Episodes 053–099 remain zero-stale
  and are not being repeated.
- `PASS` — the pre-collection runtime probe used original manifest record 011
  and completed 20/20 synchronized RGB-D actions with stale action 0. Result:
  `outputs/runtime_lod_20260829/precollect_recollect_011_probe.json`.
- A no-retry sequential collection queue is actually running from the original
  immutable 500-route manifest
  `outputs/helsinki_dataset_v1/real_500_dataset_v1_20260825/smoke_tasks.json`:
  first `episode-index-offset=11, episodes=1`, then only if that run exits 0,
  `episode-index-offset=34, episodes=19`.
- New immutable replacement directories are:
  `outputs/helsinki_dataset_v1/real_100_dataset_v1_20260829_recollect_011/`
  and
  `outputs/helsinki_dataset_v1/real_100_dataset_v1_20260829_recollect_034_052/`.
  At this checkpoint the first job is `RUNNING`; completion is **not yet
  claimed**.
- `FROZEN` — planner, expert, controller, sampler, triangle geometry and Local
  Goal were not changed. The runtime has one hidden ready sensor surface and no
  mouse automation. The `urbanfly-stale` heartbeat monitors the queue and
  fail-stop gates without starting duplicate collectors.
- Old HDF5 files have **not** been deleted. They will be moved from the active
  dataset into a recoverable, hash-manifested quarantine only after all 20
  replacements independently pass HDF5/readback/reset/timestamp QA with zero
  stale action and a replacement-aware 100-episode audit passes. A second audit
  after the move is required before any new freeze verdict.
- `LIMITATION` — `.git` is absent, so no Git cleanliness or commit provenance
  claim is possible.

Current replacement milestone: **COLLECTING / NOT YET VALIDATED**. The previous
freeze verdict is superseded pending this user-requested zero-stale audit.

### 30.14 Zero-stale replacement and two-stage 100-episode QA completed

- `PASS` — original-route recollection completed normally: episode 011 and
  episodes 034–052 are **20/20 success**, 0 collision, **10,391 transitions**,
  stale action 0 / maximum burst 0, 18/18 within-run automatic resets,
  corrupted/readback failures 0 and `.partial=0`. Both collector supervisors
  exited 0 and stopped the simulator; no collector remains running.
- A general replacement-aware read-only auditor was added at
  `uav_wm_navigation/scripts/audit_helsinki_dataset_v1_replacements.py`.
  Replacement precedence is explicit; duplicate/unexpected IDs fail. It also
  compares episode ID, task type, start, goal, ENU route and backend route and
  accounts for every adjacent episode boundary as either a factual within-run
  reset or a fresh process/replacement boundary. Regression result: **7 tests
  passed** across the original multi-run and new replacement QA tests.
- `PASS` — pre-quarantine QA selected the new 20 episodes over the old files
  without modifying either source. All **20/20 route signatures matched** the
  original routes exactly. The selected dataset contained exactly 100 unique,
  contiguous episode IDs 000–099 and every gate was true. Report:
  `outputs/helsinki_dataset_v1/real_100_dataset_v1_20260829_recollect_034_052/replacement_aware_100_prequarantine_qa.json`.
- After that PASS only, the superseded episode-011 and episode-034–052 HDF5 and
  metadata pairs were moved out of the active dataset. This is a recoverable
  quarantine, not irreversible deletion: **20 episodes / 40 files**, with
  source/destination, size and SHA-256 for every file. Manifest status is
  `COMPLETE` at
  `outputs/helsinki_dataset_v1/quarantine_stale_replaced_20260829/quarantine_manifest.json`.
- `PASS` — the second, post-quarantine independent audit reopened the active
  files and verified **100 unique contiguous HDF5 episodes / 39,767
  transitions**, 100/100 success, 0 collision, and exactly 20 episodes / 20
  successes in each of the five task types.
- `PASS` — clearance minimum/median **4.288824 / 8.994843 m**; transition dt
  mean/P95/max **0.123238 / 0.300000 / 0.700000 s**; stale action **0/39,767
  (0%)**, maximum burst 0. RGB, Depth, State, Next State, commanded/executed
  action, Local Goal, yaw/quaternion, timestamps and all HDF5 integrity/readback
  checks pass.
- `PASS` — all 99 episode boundaries are accounted for as **93/93 within-run
  automatic resets** plus **6/6 fresh process/replacement boundaries**. There
  is no cross-episode stale inheritance. Active HDF5 count is exactly 100,
  corrupted HDF5 0 and active `.partial=0`. Every post-quarantine gate is true.
  Final report:
  `outputs/helsinki_dataset_v1/real_100_dataset_v1_20260829_recollect_034_052/replacement_aware_100_postquarantine_qa.json`.
- `PASS` — a new zero-stale overview PNG was generated and visually inspected:
  `outputs/helsinki_dataset_v1/real_100_dataset_v1_20260829_recollect_034_052/replacement_aware_100_zero_stale_overview.png`.
- `FROZEN` — planner, expert, controller, sampler, triangle geometry and Local
  Goal were not modified. Changes were limited to QA, visualization and the
  explicit quarantine utility.
- `LIMITATION` — `.git` is absent, so no Git cleanliness or commit provenance
  claim is possible.

Final zero-stale replacement verdict: **`READY FOR DATASET V1 FREEZE`**. P0
blockers: none. P1 blockers: none. The `urbanfly-stale` monitor is obsolete and
may be deleted.

## 31. Minimal learned observation-policy loop and canonical data cleanup (2026-08-29)

### 31.1 Compact RGB-D / Local Goal policy

- `PASS` — a compact non-privileged policy was added in
  `urbanfly_vln/observation_policy.py`. Its inputs are two synchronized RGB-D
  frames plus Local Goal in body FLU, body linear/angular velocity, gravity and
  the preceding physical action. Global position, global route, task ID,
  triangle geometry, clearance labels and expert state are not policy inputs.
- `PASS` — `scripts/train_helsinki_observation_policy.py` trains directly from
  the selected Dataset v1 HDF5 files and saves only the best model plus one
  metrics JSON. No per-epoch checkpoints are written. The model has **253,412
  parameters** and the final checkpoint is
  `models/helsinki_observation_policy_v1.pt`, SHA-256
  `ef39c5f8184f99abdc55fcd6b62250974d081f0742ce42f49af4325ea343c1dc`.
- `PASS` / honest split limitation — episodes 000–079 are training and 080–099
  are held-out-route validation, with 16/4 episodes per task type. Dataset v1
  metadata labels all 100 source episodes `train`; this is therefore not an
  official spatial-unseen or second-city split.
- `PASS` — best epoch 7 offline validation reopened **8,852 transitions**.
  Forward/left/up/yaw MAE is **0.045224 m/s, 0.022541 m/s, 0.030416 m/s and
  0.010519 rad/s**. Total four-axis MAE is **13.5884%** of the constant
  train-mean baseline; meaningful-yaw sign accuracy is **100%**.
- The first online ground-to-rooftop attempt exposed a general terminal
  covariate-shift failure: after passing near the goal, the model entered a
  goal-behind state absent from expert demonstrations and later collided. That
  run was correctly `FAIL`, stopped the sequence and was not retained as a
  formal result.
- `PASS` — a transparent general terminal capture was added only in the policy
  wrapper `scripts/run_helsinki_observation_policy.py`. Inside the final 8 m it
  smoothly blends the learned command with a bounded Local-Goal vector
  controller; logs distinguish learned, blended and executed actions. The
  success tolerance is the existing online evaluation value **3.0 m**. Frozen
  planner, expert, controller, sampler, triangle geometry and Local Goal core
  are unchanged.
- `PASS` — the final consistent wrapper completed one held-out route from each
  task class, episodes 080–084: **5/5 success, 0 collision, 0 stale action,
  1,778 steps, 0 backend safety interventions**. Maximum cross-track error was
  **6.823606 m**. Inference latency mean/P95/max was
  **4.5586/7.0088/269.5917 ms**; the maximum is per-process CUDA warm-up, not a
  stale command. Full results are embedded in
  `models/helsinki_observation_policy_v1.metrics.json` with status
  `ONLINE_SMOKE_PASS`.
- `LIMITATION` — this is a five-episode learned-policy smoke, not a 100-episode
  learned-policy QA. No second city, 3DGS domain, appearance perturbation or
  recovery-data evaluation has been performed. It closes the minimal simulator
  observation/action loop but does not establish CityFly-level generalization.

### 31.2 Canonical main dataset and artifact minimization

- `PASS` — `scripts/canonicalize_helsinki_dataset_v1.py` created one canonical
  main dataset using verified NTFS hardlinks, avoiding a second 0.96 GiB copy:
  `outputs/helsinki_dataset_v1/main_100_zero_stale_v1/`.
- `PASS` — canonical QA independently reopened and hashed all **100 HDF5 / 39,767
  transitions**. IDs are unique and contiguous 000–099; five task classes have
  20 episodes each; success 100, collision 0, stale action 0 and `.partial=0`.
  Clearance min/median remains **4.288824/8.994843 m** and transition dt
  mean/P95/max remains **0.123238/0.300000/0.700000 s**. QA:
  `outputs/helsinki_dataset_v1/main_100_zero_stale_v1/dataset_qa.json`.
- `PASS` — after a second full hash and HDF5 readback, **147** HDF5 files or old
  path links outside the canonical directory were irreversibly removed. This
  included 100 superseded source path links and 47 historical/failed/replaced
  HDF5 files. The active Dataset v1 tree now contains exactly **100 HDF5**, all
  under the canonical directory, with zero noncanonical HDF5 and zero partials.
  Old JSON/log/PNG audit records were not bulk-deleted.
- `PASS` — temporary policy smoke checkpoints, detailed online step logs and
  CityFly PDF render intermediates were deleted. `models/` contains only the
  main checkpoint and its compact metrics JSON.
- `PASS` — canonical `dataset_qa.json` plus `dataset_manifest.json` is accepted
  directly by the training loader after old HDF5 paths are removed. New focused
  policy/canonical-data regression result: **4 tests passed**.
- `LIMITATION` — `.git` is absent, so no Git cleanliness, diff or commit
  provenance claim is possible.

Current learned-loop verdict: **`MINIMAL CLOSED LOOP PASS / FULL GENERALIZATION
NOT TESTED`**. Dataset/storage verdict: **`CANONICAL MAIN DATA PASS`**. Next
milestone is recovery/perturbation data followed by at least 25, then 100,
held-out learned-policy episodes before any stronger CityFly-level claim.

## 32. Qwen semantic small-fleet closed loop (2026-08-29)

### 32.1 Architecture and implementation

- `PASS` — a hierarchical semantic fleet layer was added without modifying the
  frozen Global Planner, Privileged Expert, triangle geometry, controller,
  sampler, or Local Goal core. Qwen is restricted to a slow semantic proposal
  role; only the existing policy/controller path can generate high-rate flight
  actions.
- `PASS` — `backend/agents/semantic_fleet.py` now contains a strict event
  schema, deterministic simulator interpreter, local OpenAI-compatible
  Qwen3-VL client, fail-closed event gate, deterministic 3–5 UAV coordinator,
  no-fly/obstacle route adapters, assignment continuity, and bounded audit.
- `PASS` — `backend/agents/simulator_bridge.py` integrates accepted temporary
  obstacles, no-fly zones, weather wind offsets and drone failures into the
  existing simulator. Dynamic routes still pass through the Helsinki static
  collision-map repair/readback. A failure during carriage creates an explicit
  recovery pickup at the failed vehicle; the failed drone remains unavailable
  for the mission.
- `PASS` — the external proposal protocol checks Qwen results against the
  current simulator clock, not merely the old observation timestamp. Stale,
  future, low-confidence, unknown-source, invalid-radius/TTL and unsupported
  proposals fail closed. A generative model cannot self-certify temporal,
  depth, telemetry, or authoritative evidence: externally supplied support
  fields are discarded before the deterministic gate. Temporary obstacles and
  landmarks require temporal plus depth support; weather/failure requires
  telemetry; no-fly requires an authoritative notice.
- `PASS` — `scripts/run_qwen_semantic_observer.py` converts live synchronized
  UFWM RGB-D packets into in-memory RGB/depth montages, calls either the pinned
  local Transformers model or an OpenAI-compatible Qwen endpoint, and sends
  only semantic JSON. It writes no frames, never sends flight actions, and now
  reports asynchronous inference failures instead of silently discarding them.
- `PASS` — the built-in `qwen_semantic_fleet` scenario uses four UAVs and four
  tasks with scripted temporary-obstacle, no-fly, gust and mid-mission failure
  events. The formal provider remains `deterministic_simulator_cues` so the
  downstream closed loop is reproducible; external Qwen proposals are optional
  and separately gated.

### 32.2 Formal validation actually completed

- `PASS` — the post-continuity 5,000-case component QA covered 3–5 UAV fleets
  and 30,038 tasks. Constraint/invariant failures 0, determinism failures 0;
  4,334/4,334 valid events accepted and 4,334/4,334 low-confidence invalid
  events rejected. Allocation P95 latency was **5.51477 ms** against a 20 ms
  gate. Report:
  `outputs/qwen_fleet_system_v1/semantic_coordinator_qa.json`.
- The first active-failure 180 s and 240 s runs correctly `FAIL`ed at 3/4 task
  completion. This exposed that the original 105 s failure happened after the
  relevant mission and was not a sufficient takeover test. The formal scenario
  was changed to a 55 s loaded-aircraft failure, explicit payload recovery and
  a realistic 300 s completion window. Failed reports were overwritten rather
  than retained as canonical artifacts.
- `PASS` — the final Helsinki run completed **300.0 simulated seconds / 15,000
  50 Hz 6DOF steps**, 4/4 tasks, 4/4 accepted events, one factual loaded-payload
  failure recovery, 0 collisions, static minimum clearance **42.301777 m**,
  fleet minimum separation **5.635219 m** against a 3 m gate, and zero static
  path-validation failures. Wall time was about 13.8 s / 21.7x realtime.
  Report: `outputs/qwen_fleet_system_v1/helsinki_6dof_qa.json`.
- `PASS` — focused backend semantic/observer/server/6DOF regression: **22
  passed**. Frontend regression: **19 passed** and production Vite build PASS.
  Semantic volumes are presentation-only and explicitly excluded from the
  RGB-D sensor layer to prevent annotation leakage.
- `LIMITATION` — the final full root suite is **69 passed / 1 failed**. The only
  failure is a missing legacy CityGS Residence artifact at
  `data/citygs_collision/Residence/collision_geometry.json`; it is unrelated to
  the Helsinki semantic-fleet changes, but the full suite cannot be claimed as
  PASS.

### 32.3 Frontend and artifact policy

- `PASS` — the formal interface now has a Semantic Agent view showing provider,
  control authority, event evidence/confidence/radius, applied plans, path
  validation failures and per-UAV task assignments. The 3D scene renders
  accepted obstacle/no-fly/weather/failure volumes.
- `PASS` — Vite manual chunks separate the approximately 122 kB application
  bundle from the approximately 686 kB Three engine and 49 kB spatial-index
  bundles. Three remains a large vendor chunk/build warning, but application
  code no longer shares its cache lifetime.
- `PASS` — this section originally retained three canonical QA JSON files under
  `outputs/qwen_fleet_system_v1/`; section 34 adds one final gated long-range
  mission-plan JSON. No per-frame RGB-D montages, transient model downloads, or
  failed scenario reports are retained. The single architecture and acceptance
  document is `docs/QWEN_SEMANTIC_FLEET_SYSTEM.md`.

### 32.4 Honest current boundary

- `PASS (negative-only QA)` — actual pinned Qwen3-VL-2B inference on all 100
  real canonical Helsinki RGB-D negative clips. This is not a positive-event
  recognition result and must not be reported as such; exact results are in
  section 32.5.
- `NOT TESTED` — positive dynamic-event recognition, live WebSocket inference,
  and Qwen-triggered closed-loop replanning.
- `NOT INTEGRATED` — all four semantic-scenario vehicles running the trained
  `helsinki_observation_policy_v1.pt`. The formal four-UAV run uses the existing
  backend 6DOF waypoint tracker. The learned observation policy remains a
  separately verified five-episode single-UAV loop.
- `NOT TESTED` — a labeled dynamic-event perception set, second city/3DGS
  generalization, PX4/HIL or real hardware.
- `LIMITATION` — `.git` remains absent, so no repository cleanliness, diff, or
  commit provenance claim is possible.

### 32.5 Pinned local Qwen negative perception QA

- `PASS` — exactly one official main model is retained at
  `models/qwen3_vl_2b_instruct/`: `Qwen/Qwen3-VL-2B-Instruct`, pinned revision
  `89644892e4d85e24eaac8bacfd4f463576704203`. The 4,255,140,312-byte
  `model.safetensors` SHA256 is
  `7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0`.
  Hugging Face cache verification passed all 12 files; the project-local
  download cache was removed after verification.
- `PASS` — direct Transformers loading and generation ran on the local NVIDIA
  GeForce RTX 5060 Laptop GPU in bfloat16. Installed runtime is
  `transformers 5.16.1`, `accelerate 1.14.0`, and `qwen-vl-utils 0.0.14`;
  vLLM is not required by the direct observer path.
- `PASS (negative-only QA)` — all 100 canonical dataset episodes were sampled,
  four real RGB-D frames per episode / 400 frames total, with 20 clips from
  each Helsinki task class. JSON validity 100/100, schema validity 100/100,
  raw false-positive clips 0, and gate-accepted false-positive clips 0. All
  responses were exactly `{\"events\":[]}`.
- Measured model load was **5.6491 s** and peak GPU allocation
  **4,620,361,216 bytes**. Inference latency mean/P50/P95/max was
  **478.36/467.00/512.65/1512.37 ms**. The maximum includes cold-start cost.
  Report: `outputs/qwen_fleet_system_v1/qwen_negative_perception_qa.json`.
- During development, an earlier prompt produced one building/tree
  hallucination and a longer response could truncate malformed JSON. The final
  conservative prompt plus independent corroboration gate eliminated both in
  this 100-clip run. Earlier failed reports were overwritten rather than kept
  as project artifacts.
- `LIMITATION` — the complete 100-episode negative set validates local loading,
  RGB-D montage, structured output and false-positive behavior on the retained
  main dataset. Because it contains no dynamic-event positives, event precision,
  recall, F1, localization and radius error remain `NOT TESTED`; the next
  research gate is one labeled 250-clip positive/negative benchmark, followed
  by matched-seed A/B/C closed-loop evaluation.

Current verdict: **`SEMANTIC MULTI-UAV EXECUTION LOOP PASS / REAL QWEN
NEGATIVE INFERENCE PASS (100 CLIPS / 400 FRAMES) / POSITIVE EVENT PERCEPTION AND
QWEN-TRIGGERED CLOSED LOOP NOT TESTED`**.

### 32.6 Final regression and retained artifacts

- `PASS` — final Python compilation passed for the semantic fleet, simulator
  bridge, server, simulator, local observer, and Qwen benchmark scripts.
- `PASS` — final frontend rerun is **19/19 tests PASS** and production Vite
  build PASS. The approximately 685.5 kB minified Three vendor chunk still
  emits the documented >500 kB warning.
- `LIMITATION` — final root regression is **69 PASS / 1 FAIL**; the only failure
  remains the absent legacy CityGS Residence collision artifact described in
  section 32.2.
- `PASS` — the official pinned Qwen revision was reverified against the Hub:
  12/12 local files checked, no missing or extra files, and the main weight
  SHA256 was rechecked. At this checkpoint,
  `outputs/qwen_fleet_system_v1/` contained three canonical QA JSON reports;
  section 34 subsequently adds the gated long-range mission-plan JSON. It still
  contains zero `.partial` files.
- `PASS` — the canonical Dataset v1 remains unchanged at exactly **100 HDF5**,
  **39,767 transitions**, **100/100 success**, **0 collisions**, **0 stale
  actions**, and **0 `.partial`** according to its independently read-back
  `dataset_qa.json` and current file count.
- `LIMITATION` — `.git` is absent, so this final rerun still cannot provide a
  clean/dirty worktree or commit-diff claim.

## 33. Action-conditioned latent world-model flight and recorded closed loop (2026-08-30)

### 33.1 Implemented world-model boundary

- `PASS` — `urbanfly_vln/navigation_world_model.py` implements a three-member,
  action-conditioned ensemble over the observation policy's exact 192-D public
  RGB-D/state latent. It predicts body-frame position delta, route-progress
  delta, next clearance and next latent mean/uncertainty. The retained main
  checkpoint is `models/helsinki_latent_world_model_v1.pt` with compact metrics
  at `models/helsinki_latent_world_model_v1.metrics.json`; it has 502,095
  parameters and status `OFFLINE_PASS`.
- `PASS` — the model was trained from the canonical Dataset v1 only: episodes
  000–079 / 30,835 transitions for training and 080–099 / 8,832 transitions for
  validation. Validation position-delta RMSE is **0.042407 m** versus a
  **0.082449 m** mean baseline; progress-delta RMSE **0.076563 m** versus
  **0.126598 m**; next-clearance MAE **1.979044 m** versus **2.326027 m**; latent
  delta RMSE **0.004724** versus **0.006557** persistence. Action sensitivity
  is **0.066317**, so the learned model did not collapse to ignoring action.
- `PASS` — `scripts/run_helsinki_world_model_video.py` uses the frozen public
  observation policy to propose an action, evaluates 15 bounded nearby actions
  through the latent ensemble, and gives the world model local candidate-
  reranking authority. It does not replace the frozen global route/controller
  stack, and the independent backend safety shield remains enabled.
- `LIMITATION` — Dataset v1 has no collisions and no clearance below
  **4.288824 m**. Therefore learned collision probability is not qualified and
  is not used as a substitute for geometric/backend safety. This is a learned
  local dynamics/prediction loop, not a claim of Dreamer/PX4-class long-horizon
  world-model control.

### 33.2 Held-out real browser flight and video

- `PASS` — canonical episode **097**, task `rooftop_to_ground`, was executed in
  the Helsinki browser digital twin. It is in the held-out 080–099 split and
  was not used for policy/world-model training. The run reached the goal in
  **363** lockstep steps with final/minimum goal distance **2.995697 m**,
  maximum route cross-track error **4.509795 m**, **0 collision**, **0 stale
  action**, and **0 safety interventions**.
- `PASS` — the world model ran on all **363/363** actions and selected a
  non-base candidate on **229** steps, proving it affected executed navigation
  rather than serving only as a visualization. Total policy plus reranking
  latency was mean/P95/max **10.7499/9.6813/913.3709 ms**; the maximum is the
  one-time CUDA cold start and explains why mean exceeds P95.
- `PASS (historical, superseded)` — the continuous browser recording was
  `outputs/world_model_flight_v1/helsinki_rooftop_to_ground_world_model.mp4`.
  It is a **1920×1080 H.264 MP4**, **2,599 frames**, **22.284 FPS**, **116.631 s**,
  SHA256 `4dab568ef5d23ba14105cf8eb697bd72b47300b2e1b61db49b4ef35141cd68f9`.
  It is not screenshot stitching. The four panels are independently generated
  near-drone third person with the visible vehicle/executed trace, synchronized
  onboard RGB, metric depth, and 192-D latent/candidate/risk telemetry. MP4
  readback and three non-black temporal samples pass; a manually inspected
  middle frame confirmed all four panels contained the intended content. This
  shorter artifact was removed after the stronger 1 km result in section 34
  passed; its exact metrics remain here as the audit record.
- The first recording exposed a factual hidden-window bug: the old recorder
  copied a presentation canvas that is intentionally suspended while hidden,
  producing a black third-person panel. That recording was deleted. The final
  implementation renders a dedicated 640×360 follow camera off-screen during
  sensor capture; no visible window or mouse ownership is required.
- `LIMITATION` — this proves a held-out stored roof-to-ground route, not yet an
  arbitrary user-selected roof and ground coordinate. Arbitrary endpoints still
  require planner feasibility validation followed by a broader held-out online
  evaluation before that wording is justified.

### 33.3 Regression, data integrity and retained-artifact policy

- `PASS` — focused world-model/policy/backend regression: **16/16**. Frontend
  regression after independent third-person rendering: **20/20**, and the Vite
  production build plus Windows desktop publish pass.
- `LIMITATION` — full root regression is **71 PASS / 1 FAIL**. The only failure
  is still the missing unrelated legacy CityGS Residence file
  `data/citygs_collision/Residence/collision_geometry.json`.
- `PASS` — after cleanup, an independent readback opened all **100/100**
  canonical HDF5 files and checked RGB, depth, state, next state, commanded and
  executed actions, Local Goal, quaternion and timestamp lengths. It found
  **39,767 transitions**, **100 unique episode IDs**, **0 schema/corruption
  failures**, **0 collision labels**, **0 stale actions**, and **0 partials**.
- `PASS` — the output tree is reduced to three top-level retained products:
  `helsinki_dataset_v1/main_100_zero_stale_v1`, `qwen_fleet_system_v1`, and
  `world_model_long_range_v1` (section 34 supersedes the earlier short
  `world_model_flight_v1`). All historical dataset siblings, failed/smoke
  planner/runtime outputs, old 35-video batch, quarantine records,
  `uav_wm_navigation/outputs` smoke artifacts, runtime logs and project
  `__pycache__` directories were removed. Approximately **340 MiB** of obsolete
  generated artifacts were irreversibly deleted. Scene assets, source code,
  dependencies, canonical data and the three main model families (observation
  policy, latent world model, pinned Qwen) were retained.
- `LIMITATION` — `.git` remains absent, so no Git cleanliness or commit
  provenance claim is possible.

Current verdict: **`HELD-OUT ROOFTOP-TO-GROUND LATENT WORLD-MODEL CLOSED LOOP
PASS / ARBITRARY-ENDPOINT GENERALIZATION NOT YET QUALIFIED`**.

## 34. One-kilometre multipoint world-model flight and four-panel video (2026-08-30)

### 34.1 Route, semantic planning and geometry gate

- `PASS` — a new custom long-range route was planned and flown from backend
  `[-400, 40, 200]` through `[-150, 40, 0]`, `[100, 40, 100]`, and
  `[250, 40, -50]` to `[400, 40, -200]`. The four planned segments total
  **1,013.678521 m**, compared with **894.427191 m** straight planar distance,
  so this is a real non-straight multipoint route rather than a renamed direct
  flight.
- `PASS` — every planned point is at the scene-valid 40 m altitude. Independent
  route validation checked 4,072 height samples and 5,089 triangle samples;
  all four segments are valid, triangle collision is false, minimum heightmap
  clearance is **7.277287 m**, minimum triangle distance is **7.282718 m**, and
  the 2.5 m required-clearance gate passes.
- `PASS (gated semantic authority)` — the pinned local
  `Qwen/Qwen3-VL-2B-Instruct` checkpoint selected waypoint order **A -> B -> C**.
  A deterministic schema/uniqueness/strict-east-monotonic gate passed before
  the order was accepted. Qwen has waypoint-order authority only and cannot
  issue flight actions. Report:
  `outputs/qwen_fleet_system_v1/qwen_long_range_mission_plan.json`.
- `LIMITATION` — no DashScope/Qwen API endpoint or key is configured, so this
  run used the actual pinned local Qwen checkpoint (`api_called=false`) rather
  than claiming an API call. The generated short rationale misstated candidate
  coordinates; the machine-checked order itself was correct and passed the
  independent gate. This is direct evidence for why free-form Qwen text is not
  trusted as control authority.
- A first configuration at 55 m was rejected before simulation because the
  actual scene upper bound is 46.1187223 m. It generated no flight/video data.
  The invalid configuration and an earlier low-altitude development run were
  deleted; neither is retained or reported as a completed experiment.

### 34.2 Real hidden-browser closed-loop flight

- `PASS` — the flight reached the final goal in **1,243** lockstep control
  steps / **264.9 s** simulator time. Minimum final-goal distance is
  **2.994887 m**, maximum route cross-track error is **13.969647 m** against a
  fixed 15 m gate, with **0 collision**, **0 stale action**, and **0 backend
  safety interventions**.
- `PASS` — the learned action-conditioned latent world model reranked all
  **1,243/1,243** actions and selected a non-base candidate on **640** steps.
  Combined policy/reranking latency mean/P95/max is
  **8.4175/10.0181/491.4307 ms**; the maximum includes one-time CUDA warm-up.
  This establishes actual executed-action influence, not telemetry-only use.
- `PASS` — the hidden desktop runtime used exactly one lockstep policy and one
  off-screen sensor surface. It produced **2,563** sensor packets with **0
  bridge drops**. Final simulator-step P95 was **5.8671 ms** and RGB-D capture-
  to-packet P95 was **153.8070 ms**.
- `LIMITATION` — this qualifies one arbitrary custom 1 km route inside the
  current Helsinki bounds. It is not statistical arbitrary-endpoint
  generalization, PX4/HIL validation, or proof of learned collision prediction;
  the frozen geometric planner and independent backend shield remain the safety
  authorities.

### 34.3 Final 3x four-panel video and artifact policy

- `PASS` — retained video:
  `outputs/world_model_long_range_v1/helsinki_1km_multipoint_world_model_3x.mp4`.
  It is continuous H.264 browser recording, not screenshot stitching:
  **1920x1080**, **30 FPS**, **3,777 frames**, **125.9 s** at **3x playback**,
  **53,029,481 bytes**, SHA256
  `f991cc8e9d31b888c8e396bc999204675677cb5e9ac194d506db231b8c528b4e`.
- `PASS` — the four synchronized panels are: whole-map real-time planned and
  executed trajectory with live vehicle marker; onboard RGB; independent
  near-vehicle third-person view with executed trace; and world-model 192-D
  latent/candidate/risk visualization. MP4 readback and three temporal samples
  pass. A manually inspected middle frame confirmed that every panel contains
  real non-black content.
- Full machine-readable QA is
  `outputs/world_model_long_range_v1/flight_qa.json`; this final directory
  contains only the MP4 and QA JSON. The superseded short
  `outputs/world_model_flight_v1` video, runtime recordings/logs, failed-run
  artifacts, and temporary visual-QA frame were removed after final PASS.
- `PASS` — frontend regression is **20/20**. Full Python regression is **71
  PASS / 1 FAIL**; the sole failure is the pre-existing missing legacy CityGS
  Residence asset `data/citygs_collision/Residence/collision_geometry.json`.
  `.git` is still absent, so repository cleanliness/commit provenance remains
  a `LIMITATION`.

Current verdict: **`1 KM MULTIPOINT LATENT WORLD-MODEL CLOSED LOOP PASS /
FOUR-PANEL 3X VIDEO PASS / QWEN SEMANTIC ORDER LOCAL-ONLY AND GATED`**.

## 35. 中文开源发布、桌面封装与 Qwen API 边界（2026-08-30）

### 35.1 精简源码与用户文档

- `PASS` — 本地目录已初始化为 Git 仓库，主分支为 `main`。当前候选提交由 **428** 个文件组成，工作树内被跟踪内容约 **8.37 MB**；最大单文件是 2,022,060 bytes 的主世界模型，不存在超过 GitHub 100 MB 上限的候选文件。
- `PASS` — `.gitignore` 明确排除 Helsinki 二进制资产、HDF5、视频、构建目录、缓存、运行日志、个人论文/参考资料、Qwen 目录和 `*.safetensors`。候选跟踪文件复核没有 Qwen 权重、HDF5、MP4 或城市 `data/` 文件。
- `PASS` — 根 `README.md`、`docs/RELEASE_v1.0.0.md`、`CONTRIBUTING.md`、`SECURITY.md`、第三方声明、环境变量说明和 Windows/Linux 启动提示均已改为中文。README 明确区分已实现能力、运行方法、实测结果与 PX4/AirSim/任意端点泛化限制。
- `PASS` — Helsinki 主数据、城市资产和最终 1 km MP4 不进入 Git 历史，而由 Release 独立分发；源码仓库只保留主模型、关键 manifest/QA 和真实演示预览图。

### 35.2 Qwen 正式发布边界

- `PASS` — `backend/agents/semantic_fleet.py`、`scripts/run_qwen_semantic_observer.py` 和 `scripts/plan_helsinki_mission_with_qwen.py` 的正式默认路径已统一为 OpenAI-compatible Qwen API。Key 仅从 `URBANFLY_QWEN_API_KEY` 或 `DASHSCOPE_API_KEY` 读取，不写入报告。
- `PASS` — 发布默认模型是 API 模型；本地 checkpoint 只能通过显式 `--direct-model` 开发参数使用，并单列在 `requirements-local-qwen.txt`。Windows stage 的 `urbanfly-release.json` 明确记录 `qwen_weights_included=false`，静默安装后的实际目录复核也确认不存在 Qwen 权重。
- `PASS` — Qwen/API、服务运行时和桌面安全的聚焦回归为 **20/20 PASS**；桌面健康/离线安全回归为 **6/6 PASS**。

### 35.3 Windows Helsinki 数字孪生安装包

- `PASS` — 后端支持 `URBANFLY_ROOT` 运行根，桌面监督器优先启动打包后的 `bin/UrbanFly.Backend/UrbanFly.Backend.exe`，开发模式仍可回退源码 Python。关闭桌面时只终止由本桌面拥有且没有采集策略占用的打包后端。
- `PASS` — PyInstaller 冻结后端在真实 stage 中通过 `/api/health` 和前端 HTTP 200 回读；补齐了 Conda Python 的 `ffi.dll` 和 SciPy Array API 隐式依赖，没有绕过健康门。
- `PASS` — Windows self-contained 桌面、完整 259,466,798 bytes / 43 文件的 `HelsinkiCentral1km` 运行资产、前端、两份主模型和中文文档已组装。便携包 `dist/release/UrbanFly-Windows-x64-1.0.0-portable.zip` 为 **329,541,237 bytes**。
- `PASS` — 简体中文 Inno Setup 安装程序 `dist/release/UrbanFly-Windows-x64-1.0.0-Setup.exe` 为 **269,403,674 bytes**。在 `dist/install-test` 进行的静默安装返回 0，启动器、后端和城市 manifest 均存在，Qwen 权重不存在；静默卸载返回 0 且测试目录完全清理。
- `PASS` — 前端生产依赖 `npm audit --omit=dev` 为 **0** 个已知漏洞。`npm ci` 报告的 6 个 high 项均来自不随运行 bundle 发布的开发依赖，因此没有执行可能破坏兼容性的 `npm audit fix --force`。

### 35.4 尚未完成的发布动作

- `PLANNED` — Helsinki 城市资产 ZIP、100-episode 主数据集 ZIP、最终 1 km 演示 MP4 的 Release staging 正在生成；不能在生成与哈希完成前记为发布成功。
- `PLANNED` — GitHub 远端仓库、v1.0.0 Release 和 Ubuntu 原生 Linux x64 包尚未完成，因此本节当前不能声称 GitHub/Windows/Linux 全部发布。

Current verdict: **`WINDOWS HELSINKI INSTALLER LOCAL QA PASS / GITHUB AND LINUX RELEASE PENDING`**.

## 36. GitHub Release 完成与 UrbanFly × Swarm 跨环境实验支线（2026-08-30）

### 36.1 GitHub 与跨平台发布真实结果

- `PASS` — Git 仓库已建立并推送到
  `https://github.com/Dev-EmbodiedAI/UrbanFly`；`v1.0.0` Release 已发布：
  `https://github.com/Dev-EmbodiedAI/UrbanFly/releases/tag/v1.0.0`。
- `PASS` — Windows installer、portable、Helsinki 城市资产、100-episode
  Dataset v1、1 km 演示 MP4 和各 manifest 均已作为 Release asset 上传；GitHub
  digest 与本地 SHA256 一致。
- `PASS` — Linux workflow run `33310660418` 在 Ubuntu 24.04 完成前端构建、
  PyInstaller 后端冻结、Helsinki 资产组装、`/api/health` 与首页 HTTP 回读并上传
  `UrbanFly-Linux-x64-1.0.0.tar.gz`。文件为 **287,411,127 bytes**，SHA256
  `6f42de3f4abbde92dd58cfe66316a758572ad7fe0a4eeb9774ef63b4069302bb`。
- Linux 首次失败来自 Windows `Compress-Archive` 的反斜杠路径；第二次失败来自
  反斜杠结尾目录项。workflow 现使用带路径穿越检查的 Python zipfile 归一化，
  没有跳过资产或健康检查。

### 36.2 Swarm 上游 contract 与原生链路

- 用户批准的跨环境实验支线不是仓库拼接，而是同一 policy 在 Swarm 程序化
  环境与 UrbanFly Helsinki 的跨域比较；它不替代 UrbanFly 的 Agent +
  Action-Conditioned World Model 主线。上游固定为
  `swarm-subnet/swarm@112a0592dab131f644cd6afdf7c6a9acd9de0a37`。
- `PASS` — 上游 `tests/test_swarm_autopilot_family.py` 为 **14/14 PASS**，实际
  创建并步进了多机 PyBullet 环境。
- `PASS` — `scripts/audit_swarm_native_contract.py` 在 City/Open/Mountain/
  Village/Forest × 2–8 UAV 的 **35/35** 组合中验证 depth `[N,128,128,1]`、
  state `[N,190]`、action `[N,5]`、shared clue、邻机槽、评分范围与 per-drone
  算术平均。
- `PASS` — 强制两机同位姿接触产生 2 个 PyBullet contacts，双方均为
  `OBSTACLE_COLLISION`，per-drone score **0.01/0.01**，final score **0.01**；
  collision → failure reason → score 链路通过。
- `LIMITATION` — 本地 Windows 只能获取 `swarm-bullet3==2.0.0.1`，上游锁定
  `2.0.0.3` 没有 Windows 包；Forest 训练期间出现旧版 collision-shape warning。
  本地报告是 native contract PASS，不是 Docker/Cap'n Proto 官方 Benchmark PASS。

### 36.3 环境无关 policy adapter

- `PASS` — `backend/integrations/swarm_policy.py` 建立不依赖 Swarm 源码的统一
  contract encoder，显式处理 UrbanFly `[east,up,north]` ↔ policy ENU
  `[east,north,up]`、公制深度归一化、25 步 action history、shared clue、7 个
  最近邻机槽和动作反变换。
- `PASS` — contract 聚焦回归 **4/4**，覆盖 shape、数值范围、坐标/yaw、邻机
  排序、padding 和动作边界；没有修改任何 Helsinki frozen component。

### 36.4 Baseline 路线修正：训练不等于 RL

- `PASS (trainability only)` — 上游共享参数 PPO 实际训练到 **51,200
  timesteps**（请求 50k，SB3 rollout 向上取整），CUDA backward、loss 更新、
  checkpoint 和 submission 打包均成功。
- `FAIL (navigation quality)` — 已完成的 City/Open/Mountain 完整 rollout 均
  失败或发生碰撞，score 为 0.01。用户质疑无需强制 RL 后，剩余矩阵被停止；
  不存在完整 35-episode PPO Benchmark 报告，也不得写成已完成。
- `PASS (cleanup)` — 失败 PPO checkpoint、submission 和生成目录已删除，未进入
  `models/`、Git 或 Release。
- `PASS (trainable non-RL baseline)` —
  `backend/integrations/swarm_imitation.py` 实现共享 depth/state 编码、动态 2–8
  机 self-attention、动作头、collision auxiliary head、masked imitation loss 和
  checkpoint schema。forward/bounds/backward/mask/checkpoint **4/4 PASS**；与
  contract 合计 **8/8 PASS**。
- `NOT TRAINED` — imitation 网络目前没有主 checkpoint 或成功率。Swarm 实验
  支线的 baseline 训练路径变为 classical privileged teacher → BC → DAgger；
  RL 仅是可选 residual，不能成为基本可飞性的前提。该 baseline 只用于统一
  policy 的预训练、benchmark 与消融，不替代 UrbanFly Agent 的任务推理、Qwen
  API 能力或 World Model 的未来隐状态/风险预测和在线轨迹选择。

详细方案：`docs/SWARM_CROSS_DOMAIN.md`。

Current verdict: **`SWARM NATIVE CONTRACT 35/35 PASS / COLLISION-SCORE CHAIN PASS /
DOCKER BENCHMARK NOT TESTED / DYNAMIC IMITATION BASELINE TRAINABLE BUT NOT TRAINED`**.

## 37. 跨环境数字孪生闭环与平台化 Helsinki 1 km 复核（2026-08-31）

### 37.1 Swarm 五环境同生命周期闭环

- `PASS` — 新增 `backend/digital_twin/goal_world_model.py`、
  `swarm_adapter.py` 与 `qa.py`，以及正式运行/合并审计脚本。City、Open、
  Mountain、Village、Forest 在固定 seed `20260831`、每类 2 UAV 下均使用
  “任务分配 → policy → 一步预测式 World Model 重排 → PyBullet 原生执行 →
  fresh depth/state feedback”同一因果生命周期。
- `PASS` — 五类合计 **10/10 UAV 成功、0 collision、12,231 控制步**。各环境
  控制步为 1,727 / 1,838 / 2,935 / 2,988 / 2,743；原生分数为
  0.8514 / 0.7530 / 0.5727 / 0.6028 / 0.6185。独立 QA 位于
  `outputs/cross_environment_digital_twin_v1/cross_environment_qa.json`，上游固定为
  `swarm-subnet/swarm@112a0592dab131f644cd6afdf7c6a9acd9de0a37`。
- `PASS (cleanup)` — 44 个开发/失败 JSON 已不可恢复删除，正式目录只保留五个
  final 环境报告和一个合并 QA。
- `LIMITATION` — 这是向策略暴露 exact goal 的数字孪生模式，所有报告均明确
  `benchmark_eligible=false`，不是官方隐藏目标 Swarm Benchmark。Swarm 端目前是
  解析式一步预测器，不是 Helsinki 的 learned latent checkpoint；单 seed、2 UAV
  不能证明统计泛化。

### 37.2 统一 Helsinki 平台适配与 P0 修复

- `PASS` — `backend/digital_twin/helsinki_adapter.py` 将真实隐藏浏览器运行时封装为
  `connect/reset → RGB-D+6DoF observation → velocity action → factual executed action /
  collision / fresh observation` 强顺序 session，并加入 shape、finite、timestamp、
  stale/collision fail-closed 检查。冻结的 Global Planner、Privileged Expert、三角
  几何、controller、sampler 和 Local Goal 核心均未修改。
- `P0 FIXED` — 初始包名 `backend/platform` 会在旧 server 把 `backend` 插入
  `sys.path` 后遮蔽 Python 标准库 `platform`，导致 aiohttp/numpy/scipy 启动失败。
  包已整体更名为 `backend/digital_twin`，标准库 `platform.machine` 与 backend import
  gate 均恢复通过。
- `P1 FIXED` — `scripts/launch_desktop.ps1` 原来寻找不存在的
  `UrbanFly.exe`；开发 publish 的真实入口是 `UrbanFly.Desktop.exe`。启动器现使用
  正确入口并在构建未产生它时明确失败。

### 37.3 新的真实 Agent → World Model → Helsinki → feedback 复核

- `PASS` — 使用 100 条零-stale 主数据 QA、
  `models/helsinki_observation_policy_v1.pt`、
  `models/helsinki_latent_world_model_v1.pt` 和已 gated 的语义 waypoint plan，按原
  1,013.678521 m 四段路线完成新的隐藏桌面闭环。运行时始终只有一个隐藏 ready
  surface 和一个 lockstep policy，无鼠标操作、无第二传感器表面。
- `PASS` — 正式结果位于
  `outputs/digital_twin_platform_v1/helsinki_1km_final_v2`：**1,114 步成功**，
  `AgentStatus=COMPLETE`，4/4 mission waypoints reached，因果链完整，fresh
  feedback **1,114/1,114**；learned latent World Model 重排 **1,114/1,114**，
  并在 **633** 步改变 base policy 选择。
- `PASS` — 0 collision、0 stale action、0 backend safety intervention；最终最小
  goal distance **2.6344 m**，最大 cross-track **11.8748 m**，通过 3 m/15 m gate。
  四段 frozen planner 路线均 valid，最小 heightmap clearance **7.2773 m**，最小
  triangle distance **7.2827 m**，triangle collision=false。
- `PASS` — 连续四分屏 3× MP4 为 1920×1080、30 FPS、3,667 frames、
  **122.233 s**、48,517,036 bytes，SHA256
  `975c4022f848b888da186210b864de9225f13f43f259fc0b6bfa4cdb0d3e6826`。
  独立哈希回读一致；60 s 抽帧人工确认第三视角/真实轨迹、RGB、Depth、192-D
  learned latent/候选风险四个 panel 均为有效非黑内容。
- `FAIL (preserved fact, artifact removed from project)` — 首次平台复核运行误用
  0.1 s command horizon，在 step 918 触发 cross-track 15.035 m > 15 m gate，脚本
  正确 fail-closed，未伪装为成功。对照旧 PASS 轨迹确定已验证 horizon 实为 0.5 s；
  保持所有 gate 不变重跑后通过。脚本默认值已固化为 0.5 s，并在 QA 中记录
  `action_duration_s`。失败目录和更早 backend-disconnect 失败目录已移出项目到可
  恢复临时清理目录 `C:/Users/caste/AppData/Local/Temp/UrbanFly_cleanup_20260831`；
  本轮 17 个 runtime 开发日志也移到同一目录。正式输出只保留 PASS 成品。
- `LIMITATION` — 当前环境未配置 `URBANFLY_QWEN_API_KEY`、
  `DASHSCOPE_API_KEY` 或 Qwen base URL，因此本次继续消费历史 gated local-Qwen
  语义顺序，`api_called=false`。正式发布代码默认仍是 OpenAI-compatible Qwen API，
  不携带或提交 Qwen 原始权重。单条 1 km PASS 仍不等于多 seed/任意端点统计泛化。

### 37.4 验证与下一 Gate

- `PASS` — 最终聚焦回归为 **23/23 PASS**，覆盖五环境 digital-twin policy/QA、
  Helsinki 强顺序 adapter、Agent 闭环、Swarm contract/imitation、latent World Model，
  并固定验证正式视频 runner 的默认 command horizon 为 0.5 s。九个相关 Python
  模块/脚本显式 `py_compile` 通过；更名后的 Python stdlib/backend import gate 通过。
- `LIMITATION` — 本轮代码与小型 QA 已提交到本地 `main`。向现有 GitHub origin
  push 时，全局 `127.0.0.1:10809` 代理无服务；仅对单次命令禁用代理后直连又被
  网络层 reset，因此本轮提交尚未到达远端。未修改用户全局 Git/网络配置；网络
  恢复后执行 `git push origin main` 即可。
- `NEXT` — 先补齐最终回归与源码/文档一致性检查；随后扩展 Swarm 多 seed、
  2–8 UAV teacher 数据，并让同一 learned checkpoint 进入 Swarm 与 Helsinki
  zero-shot/adaptation 对照。Qwen API 只在用户配置 key 后作为高层任务规划器，
  不进入低层安全控制。

Current verdict: **`AGENT → LEARNED WORLD MODEL → HELSINKI EXECUTION → FRESH
FEEDBACK CLOSED LOOP PASS / SWARM FIVE-ENV EXACT-GOAL DIGITAL TWIN PASS /
FORMAL CROSS-DOMAIN GENERALIZATION NOT YET QUALIFIED`**.

## 38. Helsinki 低空建筑走廊穿梭演示（2026-08-31）

### 38.1 路线筛选与 runner 泛化

- 用户要求的不只是楼顶上方巡航，而是能在视频中明确看到建筑立面、树列、车辆、
  路口与局部避障的低空城市穿梭。对 canonical Dataset v1 的 20 条
  `street_canyon` 和 20 条 `building_blocked` 真实 RGB 中段进行临时视觉筛选后，
  发现很多 `street_canyon` 实际位于水道、绿地或宽路，不能仅凭任务标签声称是
  建筑街谷。
- `PASS` — `scripts/run_helsinki_world_model_video.py` 已从仅允许
  `rooftop_to_ground` 的旧演示限制，泛化为五类 canonical 任务均可运行，并根据
  episode 索引自动标记 training/held-out；这只修改 runner，不改变任何 frozen
  planner/controller/geometry/sampler/Local Goal 核心。
- `PASS` — `scripts/launch_desktop.ps1` 现在显式设置
  `URBANFLY_START_MINIMIZED=1`，实测可在不抢鼠标的情况下启动恰好一个 hidden、
  ready 的离屏传感器表面。

### 38.2 最终 held-out 建筑走廊结果

- `PASS` — 最终主演示选择 canonical episode **095**，任务类型
  `building_blocked`，属于 held-out 080–099，未参与 policy/World Model 训练。
  历史专家轨迹约 127.5 m，高度约 10.3–17.8 m，最小 clearance 约 5.9 m；路线
  从住宅楼立面旁进入道路/停车区走廊，经过树列、车辆和城市路口。
- `PASS` — 新闭环在 **177** 步到达，最终 goal distance **2.956 m**，最大
  cross-track **5.842 m**；抽样实际高度 **10.74–15.06 m**。0 collision、0 stale
  action，后台独立安全层真实介入 **3** 次。
- `PASS` — learned latent World Model 重排 **177/177** 步，并在 **156** 步改变
  base policy 选择；Agent causal chain complete，fresh feedback **177/177**。
- `PASS` — 主视频：
  `outputs/digital_twin_platform_v1/helsinki_building_canyon_095/helsinki_building_canyon_world_model_2x.mp4`。
  连续浏览器录制，非截图拼接；1920×1080、30 FPS、674 frames、2×、
  **22.467 s**、8,533,850 bytes，SHA256
  `4b605bafbf3b66711286c8f27cf95e2852e7f8e6348703e890f286fe7d1715e5`。
  独立哈希一致；3 s / 11 s / 19 s 抽帧人工确认第三视角、机载 RGB、Depth 和
  192-D latent/候选风险四屏均有效，且建筑立面持续处于近距离视野。
- `PASS (honest comparison)` — 先运行的 held-out episode 086 也通过，243 步、
  5 次安全介入，但人工视觉 QA 发现主体是水道/桥梁走廊；episode 076 的建筑
  视觉更明显但属于 training 路线。两者都没有被冒充成最终 held-out 建筑演示，
  筛选视频与临时联系表在 095 PASS 后移出项目，仅保留主成品；开发运行日志也
  移到 `C:/Users/caste/AppData/Local/Temp/UrbanFly_cleanup_20260831`。结束时
  simulator stopped、policy 0，专用 hidden desktop/backend 进程均已关闭。
- `PASS` — runner/adapter/Agent/Swarm/latent World Model 最终聚焦回归仍为
  **23/23 PASS**，runner 显式 `py_compile` 通过。
- `LIMITATION` — 当前 Helsinki 资产主要是中低层住宅与道路，不是香港式高密
  摩天楼峡谷；本结果证明真实低空建筑走廊闭环和安全介入，不等于动态行人/车辆
  预测避障或多 seed 的低空统计泛化。后续可增加经三角网格验证的多转弯街谷
  scenario suite，而不是通过降低安全阈值制造危险画面。

Current verdict: **`HELD-OUT LOW-ALTITUDE BUILDING-CORRIDOR CLOSED LOOP PASS /
VISIBLE OBSTACLE CONTEXT AND SAFETY INTERVENTION PASS / DENSE HIGH-RISE AND
DYNAMIC-OBSTACLE GENERALIZATION NOT YET QUALIFIED`**.
