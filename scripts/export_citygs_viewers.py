from __future__ import annotations

import argparse
import base64
import json
import math
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib import animation
from mpl_toolkits.mplot3d.art3d import Line3DCollection

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


SH_C0 = 0.28209479177387814
THREE_VERSION = "0.128.0"
THREE_URL = f"https://cdn.jsdelivr.net/npm/three@{THREE_VERSION}/build/three.min.js"
ORBIT_URL = (
    f"https://cdn.jsdelivr.net/npm/three@{THREE_VERSION}/examples/js/controls/OrbitControls.js"
)
TRACKBALL_URL = (
    f"https://cdn.jsdelivr.net/npm/three@{THREE_VERSION}/examples/js/controls/TrackballControls.js"
)


@dataclass(frozen=True)
class SceneSpec:
    name: str
    point_cloud_path: Path
    cameras_path: Path


DEFAULT_SCENES = [
    SceneSpec(
        name="Residence",
        point_cloud_path=Path(
            r"C:\Users\caste\Downloads\Residence\residence_c20_r4_light_60_vq\point_cloud.ply"
        ),
        cameras_path=Path(r"C:\Users\caste\Downloads\Residence\residence_c20_r4\cameras.json"),
    ),
    SceneSpec(
        name="SciArt",
        point_cloud_path=Path(
            r"C:\Users\caste\Downloads\SciArt\sciart_c9_r4\point_cloud\iteration_30000\point_cloud.ply"
        ),
        cameras_path=Path(r"C:\Users\caste\Downloads\SciArt\sciart_c9_r4\cameras.json"),
    ),
]


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def ensure_vendor_assets(vendor_dir: Path) -> None:
    vendor_dir.mkdir(parents=True, exist_ok=True)
    assets = {
        "three.min.js": THREE_URL,
        "OrbitControls.js": ORBIT_URL,
        "TrackballControls.js": TRACKBALL_URL,
    }
    for filename, url in assets.items():
        target = vendor_dir / filename
        if target.exists() and target.stat().st_size > 0:
            continue
        print(f"Downloading {filename} ...")
        urllib.request.urlretrieve(url, target)


def parse_ply_header(ply_path: Path) -> tuple[int, int, np.dtype]:
    props: list[tuple[str, str]] = []
    vertex_count = 0
    with ply_path.open("rb") as handle:
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"Unexpected EOF while reading {ply_path}")
            decoded = line.decode("ascii", errors="strict").strip()
            if decoded.startswith("element vertex"):
                vertex_count = int(decoded.split()[2])
            elif decoded.startswith("property"):
                _, data_type, name = decoded.split()
                props.append((name, data_type))
            elif decoded == "end_header":
                header_end = handle.tell()
                break
    dtype_map = {
        "float": "<f4",
        "float32": "<f4",
        "uchar": "u1",
        "uint8": "u1",
        "int": "<i4",
        "int32": "<i4",
    }
    dtype_fields = []
    for name, data_type in props:
        if data_type not in dtype_map:
            raise ValueError(f"Unsupported property type {data_type!r} in {ply_path}")
        dtype_fields.append((name, dtype_map[data_type]))
    return vertex_count, header_end, np.dtype(dtype_fields)


def read_gaussian_sample(
    ply_path: Path,
    target_points: int,
    opacity_min: float,
    oversample_factor: float,
    chunk_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertex_count, header_end, dtype = parse_ply_header(ply_path)
    rng = np.random.default_rng(seed)
    stride_target = max(int(target_points * oversample_factor), target_points)
    stride = max(vertex_count // stride_target, 1)
    offset = int(rng.integers(0, stride))

    sampled_xyz: list[np.ndarray] = []
    sampled_rgb: list[np.ndarray] = []
    sampled_alpha: list[np.ndarray] = []

    itemsize = dtype.itemsize
    processed = 0

    print(f"Reading {ply_path.name}: {vertex_count:,} gaussians, stride={stride}")
    with ply_path.open("rb") as handle:
        handle.seek(header_end)
        while processed < vertex_count:
            record_count = min(chunk_size, vertex_count - processed)
            raw = handle.read(record_count * itemsize)
            if not raw:
                break
            chunk = np.frombuffer(raw, dtype=dtype, count=record_count)
            alphas = sigmoid(chunk["opacity"].astype(np.float32))
            global_indices = processed + np.arange(record_count, dtype=np.int64)
            mask = ((global_indices + offset) % stride == 0) & (alphas >= opacity_min)
            if np.any(mask):
                colors = np.stack(
                    [chunk["f_dc_0"][mask], chunk["f_dc_1"][mask], chunk["f_dc_2"][mask]],
                    axis=1,
                ).astype(np.float32)
                colors = np.clip(0.5 + SH_C0 * colors, 0.0, 1.0)
                xyz = np.stack(
                    [chunk["x"][mask], chunk["y"][mask], chunk["z"][mask]],
                    axis=1,
                ).astype(np.float32)
                sampled_xyz.append(xyz)
                sampled_rgb.append((colors * 255.0).round().astype(np.uint8))
                sampled_alpha.append(alphas[mask].astype(np.float32))
            processed += record_count
            if processed % max(chunk_size * 4, 1) == 0 or processed == vertex_count:
                kept = sum(part.shape[0] for part in sampled_xyz)
                print(f"  processed {processed:,}/{vertex_count:,}, kept {kept:,}")

    xyz = np.concatenate(sampled_xyz, axis=0) if sampled_xyz else np.empty((0, 3), np.float32)
    rgb = np.concatenate(sampled_rgb, axis=0) if sampled_rgb else np.empty((0, 3), np.uint8)
    alpha = (
        np.concatenate(sampled_alpha, axis=0) if sampled_alpha else np.empty((0,), np.float32)
    )

    if xyz.shape[0] > target_points:
        weights = np.clip(alpha, 1e-3, 1.0)
        probabilities = weights / weights.sum()
        indices = rng.choice(xyz.shape[0], size=target_points, replace=False, p=probabilities)
        indices.sort()
        xyz = xyz[indices]
        rgb = rgb[indices]
        alpha = alpha[indices]

    return xyz, rgb, alpha


def write_binary_color_ply(output_path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = "\n".join(
        [
            "ply",
            "format binary_little_endian 1.0",
            "comment generated from CityGS gaussian splats",
            f"element vertex {xyz.shape[0]}",
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "end_header",
            "",
        ]
    ).encode("ascii")
    packed = np.empty(
        xyz.shape[0],
        dtype=np.dtype(
            [
                ("x", "<f4"),
                ("y", "<f4"),
                ("z", "<f4"),
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
            ]
        ),
    )
    packed["x"] = xyz[:, 0]
    packed["y"] = xyz[:, 1]
    packed["z"] = xyz[:, 2]
    packed["red"] = rgb[:, 0]
    packed["green"] = rgb[:, 1]
    packed["blue"] = rgb[:, 2]
    with output_path.open("wb") as handle:
        handle.write(header)
        handle.write(packed.tobytes())


def load_cameras(cameras_path: Path) -> list[dict]:
    with cameras_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def infer_orientation_basis(
    rotation_matrices: np.ndarray,
    camera_positions: np.ndarray,
    scene_center: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    candidate_sets = []
    row_right = rotation_matrices[:, 0, :]
    row_up = -rotation_matrices[:, 1, :]
    row_forward = rotation_matrices[:, 2, :]
    col_right = rotation_matrices[:, :, 0]
    col_up = -rotation_matrices[:, :, 1]
    col_forward = rotation_matrices[:, :, 2]

    candidate_sets.append((row_right, row_up, row_forward))
    candidate_sets.append((row_right, row_up, -row_forward))
    candidate_sets.append((col_right, col_up, col_forward))
    candidate_sets.append((col_right, col_up, -col_forward))

    view_to_center = scene_center[None, :] - camera_positions
    view_norm = np.linalg.norm(view_to_center, axis=1, keepdims=True).clip(min=1e-6)
    view_to_center = view_to_center / view_norm

    best_score = -np.inf
    best_basis = None
    for right, up, forward in candidate_sets:
        normed_forward = forward / np.linalg.norm(forward, axis=1, keepdims=True).clip(min=1e-6)
        score = np.mean(np.sum(normed_forward * view_to_center, axis=1))
        if score > best_score:
            best_score = score
            best_basis = (right, up, normed_forward)
    assert best_basis is not None

    right, up, forward = best_basis
    right = right / np.linalg.norm(right, axis=1, keepdims=True).clip(min=1e-6)
    up = up / np.linalg.norm(up, axis=1, keepdims=True).clip(min=1e-6)
    forward = forward / np.linalg.norm(forward, axis=1, keepdims=True).clip(min=1e-6)
    return right.astype(np.float32), up.astype(np.float32), forward.astype(np.float32)


def build_camera_geometry(
    cameras: list[dict],
    scene_center: np.ndarray,
    bbox_diag: float,
) -> tuple[np.ndarray, np.ndarray]:
    positions = np.asarray([cam["position"] for cam in cameras], dtype=np.float32)
    rotations = np.asarray([cam["rotation"] for cam in cameras], dtype=np.float32)
    right, up, forward = infer_orientation_basis(rotations, positions, scene_center)

    step = max(len(cameras) // 48, 1)
    subset = np.arange(0, len(cameras), step, dtype=np.int32)
    if subset[-1] != len(cameras) - 1:
        subset = np.append(subset, len(cameras) - 1)

    frustum_depth = max(bbox_diag * 0.018, 0.8)
    frustum_width = frustum_depth * 0.55
    frustum_height = frustum_depth * 0.32

    segments: list[np.ndarray] = []
    for idx in subset:
        origin = positions[idx]
        fwd = forward[idx]
        side = right[idx]
        vertical = up[idx]
        face_center = origin + fwd * frustum_depth
        corners = [
            face_center + side * frustum_width + vertical * frustum_height,
            face_center - side * frustum_width + vertical * frustum_height,
            face_center - side * frustum_width - vertical * frustum_height,
            face_center + side * frustum_width - vertical * frustum_height,
        ]
        for corner in corners:
            segments.append(np.stack([origin, corner], axis=0))
        for a, b in zip(corners, corners[1:] + corners[:1]):
            segments.append(np.stack([a, b], axis=0))

    frustum_lines = np.stack(segments, axis=0).astype(np.float32) if segments else np.empty((0, 2, 3), np.float32)
    return positions.astype(np.float32), frustum_lines


def estimate_alignment_transform(
    xyz: np.ndarray,
    camera_positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    origin = np.median(xyz, axis=0).astype(np.float32)
    camera_mean = camera_positions.mean(axis=0).astype(np.float32)
    up_axis = camera_mean - origin
    up_norm = float(np.linalg.norm(up_axis))
    if up_norm < 1e-6:
        up_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        up_axis = up_axis / up_norm

    centered = xyz - origin[None, :]
    projected = centered - np.outer(centered @ up_axis, up_axis)
    covariance = projected.T @ projected / max(projected.shape[0] - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    forward_axis = eigenvectors[:, int(np.argmax(eigenvalues))].astype(np.float32)
    forward_axis = forward_axis - up_axis * float(np.dot(forward_axis, up_axis))
    forward_norm = float(np.linalg.norm(forward_axis))
    if forward_norm < 1e-6:
        fallback = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if abs(float(np.dot(fallback, up_axis))) > 0.9:
            fallback = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        forward_axis = fallback - up_axis * float(np.dot(fallback, up_axis))
        forward_norm = float(np.linalg.norm(forward_axis))
    forward_axis = forward_axis / max(forward_norm, 1e-6)

    lateral_axis = np.cross(up_axis, forward_axis).astype(np.float32)
    lateral_norm = float(np.linalg.norm(lateral_axis))
    if lateral_norm < 1e-6:
        lateral_axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        lateral_axis = lateral_axis - up_axis * float(np.dot(lateral_axis, up_axis))
        lateral_axis = lateral_axis / max(float(np.linalg.norm(lateral_axis)), 1e-6)
    else:
        lateral_axis = lateral_axis / lateral_norm

    forward_axis = np.cross(lateral_axis, up_axis).astype(np.float32)
    forward_axis = forward_axis / max(float(np.linalg.norm(forward_axis)), 1e-6)

    basis = np.stack([forward_axis, lateral_axis, up_axis], axis=1).astype(np.float32)
    return origin, basis


def apply_alignment(values: np.ndarray, origin: np.ndarray, basis: np.ndarray) -> np.ndarray:
    original_shape = values.shape
    flattened = values.reshape(-1, 3).astype(np.float32)
    transformed = (flattened - origin[None, :]) @ basis
    return transformed.reshape(original_shape).astype(np.float32)


def set_axes_bounds(ax: plt.Axes, *arrays: np.ndarray) -> None:
    valid_arrays = [arr for arr in arrays if arr.size]
    if not valid_arrays:
        return
    stacked = np.concatenate(valid_arrays, axis=0)
    mins = np.percentile(stacked, 1.0, axis=0)
    maxs = np.percentile(stacked, 99.0, axis=0)
    center = (mins + maxs) / 2.0
    radius = float(np.max(maxs - mins) / 2.0)
    radius = max(radius, 1.0)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def make_camera_pose_figure(
    output_path: Path,
    scene_name: str,
    xyz: np.ndarray,
    rgb: np.ndarray,
    camera_path: np.ndarray,
    frustum_lines: np.ndarray,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    plot_count = min(60000, xyz.shape[0])
    indices = rng.choice(xyz.shape[0], size=plot_count, replace=False) if xyz.shape[0] > plot_count else np.arange(xyz.shape[0])
    sampled_xyz = xyz[indices]
    sampled_rgb = rgb[indices] / 255.0

    fig = plt.figure(figsize=(15, 8.6), dpi=240, facecolor="white")
    views = [
        (1, 24, 44, "Perspective"),
        (2, 88, -90, "Top View"),
    ]
    for subplot_index, elev, azim, title in views:
        ax = fig.add_subplot(1, 2, subplot_index, projection="3d")
        ax.set_title(f"{scene_name} - {title}", fontsize=13, pad=16)
        ax.scatter(
            sampled_xyz[:, 0],
            sampled_xyz[:, 1],
            sampled_xyz[:, 2],
            c=sampled_rgb,
            s=0.32,
            alpha=0.62,
            linewidths=0,
        )
        ax.plot(
            camera_path[:, 0],
            camera_path[:, 1],
            camera_path[:, 2],
            color="#d62728",
            linewidth=1.65,
            alpha=0.98,
        )
        if frustum_lines.size:
            ax.add_collection3d(
                Line3DCollection(
                    frustum_lines,
                    colors="#1f77b4",
                    linewidths=0.8,
                    alpha=0.62,
                )
            )
        ax.view_init(elev=elev, azim=azim)
        set_axes_bounds(ax, sampled_xyz, camera_path)
        ax.set_axis_off()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def encode_array(array: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode("ascii")


def make_turntable_gif(
    output_path: Path,
    scene_name: str,
    xyz: np.ndarray,
    rgb: np.ndarray,
    camera_path: np.ndarray,
    frustum_lines: np.ndarray,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    plot_count = min(45000, xyz.shape[0])
    indices = (
        rng.choice(xyz.shape[0], size=plot_count, replace=False)
        if xyz.shape[0] > plot_count
        else np.arange(xyz.shape[0])
    )
    sampled_xyz = xyz[indices]
    sampled_rgb = rgb[indices] / 255.0

    fig = plt.figure(figsize=(7.2, 5.4), dpi=150, facecolor="white")
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.scatter(
        sampled_xyz[:, 0],
        sampled_xyz[:, 1],
        sampled_xyz[:, 2],
        c=sampled_rgb,
        s=0.42,
        alpha=0.55,
        linewidths=0,
    )
    ax.plot(
        camera_path[:, 0],
        camera_path[:, 1],
        camera_path[:, 2],
        color="#d62728",
        linewidth=1.75,
        alpha=0.98,
    )
    if frustum_lines.size:
        ax.add_collection3d(
            Line3DCollection(
                frustum_lines,
                colors="#1f77b4",
                linewidths=0.75,
                alpha=0.6,
            )
        )
    ax.set_title(f"{scene_name} - Camera Orbit", fontsize=13, pad=10)
    set_axes_bounds(ax, sampled_xyz, camera_path)
    ax.set_axis_off()

    def update(azimuth: float) -> list[plt.Axes]:
        ax.view_init(elev=21, azim=float(azimuth))
        return [ax]

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=np.linspace(48.0, 408.0, 28, endpoint=False),
        interval=120,
        blit=False,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(output_path, writer=animation.PillowWriter(fps=8))
    plt.close(fig)


def build_viewer_html(
    scene_name: str,
    xyz: np.ndarray,
    rgb: np.ndarray,
    camera_path: np.ndarray,
    frustum_lines: np.ndarray,
    vendor_path_prefix: str,
) -> str:
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    center = (mins + maxs) / 2.0
    bbox_size = maxs - mins
    bbox_diag = float(np.linalg.norm(bbox_size))
    point_size = max(bbox_diag * 0.0032, 0.5)

    data = {
        "positions": encode_array(xyz.astype(np.float32)),
        "colors": encode_array(rgb.astype(np.uint8)),
        "cameraPath": encode_array(camera_path.astype(np.float32)),
        "frustums": encode_array(frustum_lines.astype(np.float32)),
        "pointCount": int(xyz.shape[0]),
        "cameraCount": int(camera_path.shape[0]),
        "bboxDiag": bbox_diag,
        "center": center.tolist(),
        "pointSize": point_size,
    }
    json_blob = json.dumps(data)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{scene_name} CityGS Viewer</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f5f7fb;
      --panel: rgba(255, 255, 255, 0.88);
      --text: #172033;
      --accent: #2251cc;
      --muted: #5a647a;
      --border: rgba(23, 32, 51, 0.12);
    }}
    html, body {{
      margin: 0;
      width: 100%;
      height: 100%;
      background: radial-gradient(circle at top, #ffffff 0%, #eef2fb 48%, #dfe6f4 100%);
      font-family: "Segoe UI", Arial, sans-serif;
      overflow: hidden;
      color: var(--text);
    }}
    #viewer {{
      width: 100%;
      height: 100%;
    }}
    .panel {{
      position: fixed;
      top: 18px;
      left: 18px;
      width: min(360px, calc(100vw - 36px));
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 16px;
      box-shadow: 0 16px 48px rgba(25, 39, 72, 0.16);
      backdrop-filter: blur(10px);
      padding: 16px 18px;
      z-index: 10;
    }}
    .panel h1 {{
      margin: 0 0 10px;
      font-size: 22px;
      line-height: 1.15;
    }}
    .meta {{
      font-size: 13px;
      color: var(--muted);
      line-height: 1.55;
      margin-bottom: 12px;
    }}
    .controls {{
      display: grid;
      gap: 10px;
      font-size: 14px;
    }}
    .row {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }}
    .row input[type="range"] {{
      width: 150px;
    }}
    .row button {{
      border: 0;
      background: var(--accent);
      color: white;
      padding: 8px 12px;
      border-radius: 999px;
      cursor: pointer;
      font-weight: 600;
    }}
    .hint {{
      margin-top: 10px;
      font-size: 12px;
      color: var(--muted);
      line-height: 1.45;
    }}
    @media (max-width: 700px) {{
      .panel {{
        left: 12px;
        top: 12px;
        width: calc(100vw - 24px);
      }}
    }}
  </style>
</head>
<body>
  <div class="panel">
    <h1>{scene_name}</h1>
    <div class="meta">
      Dense CityGS point cloud for PowerPoint and local inspection.<br />
      Points: <strong id="pointCount"></strong> | Cameras: <strong id="cameraCount"></strong>
    </div>
    <div class="controls">
      <label class="row">
        <span>Point size</span>
        <input id="pointSize" type="range" min="0.25" max="{max(point_size * 4.0, 1.5):.3f}" step="0.05" value="{point_size:.3f}" />
      </label>
      <label class="row">
        <span>Camera path</span>
        <input id="showPath" type="checkbox" checked />
      </label>
      <label class="row">
        <span>Camera frustums</span>
        <input id="showFrustums" type="checkbox" checked />
      </label>
      <label class="row">
        <span>Auto rotate</span>
        <input id="autoRotate" type="checkbox" />
      </label>
      <div class="row">
        <span>View</span>
        <button id="resetView" type="button">Reset</button>
      </div>
    </div>
    <div class="hint">
      Mouse: left drag for free rotation, wheel to zoom, right drag to pan. This HTML is fully local and can be opened directly.
    </div>
  </div>
  <div id="viewer"></div>

  <script src="{vendor_path_prefix}/three.min.js"></script>
  <script src="{vendor_path_prefix}/TrackballControls.js"></script>
  <script>
    const DATA = {json_blob};

    function decodeArray(base64, ArrayType) {{
      const binary = atob(base64);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i += 1) {{
        bytes[i] = binary.charCodeAt(i);
      }}
      return new ArrayType(bytes.buffer);
    }}

    const positions = decodeArray(DATA.positions, Float32Array);
    const colors = decodeArray(DATA.colors, Uint8Array);
    const cameraPath = decodeArray(DATA.cameraPath, Float32Array);
    const frustums = decodeArray(DATA.frustums, Float32Array);

    document.getElementById("pointCount").textContent = DATA.pointCount.toLocaleString();
    document.getElementById("cameraCount").textContent = DATA.cameraCount.toLocaleString();

    const container = document.getElementById("viewer");
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf5f7fb);
    scene.up.set(0, 0, 1);

    const renderer = new THREE.WebGLRenderer({{ antialias: true }});
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(window.innerWidth, window.innerHeight);
    container.appendChild(renderer.domElement);

    const camera = new THREE.PerspectiveCamera(52, window.innerWidth / window.innerHeight, 0.1, 100000);
    camera.up.set(0, 0, 1);
    const controls = new THREE.TrackballControls(camera, renderer.domElement);
    controls.rotateSpeed = 4.3;
    controls.zoomSpeed = 1.25;
    controls.panSpeed = 0.85;
    controls.dynamicDampingFactor = 0.12;
    controls.staticMoving = false;

    const ambient = new THREE.HemisphereLight(0xffffff, 0xaab4cf, 1.25);
    scene.add(ambient);
    const directional = new THREE.DirectionalLight(0xffffff, 0.45);
    directional.position.set(1, 1.5, 2.0);
    scene.add(directional);

    const pointGeometry = new THREE.BufferGeometry();
    pointGeometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    pointGeometry.setAttribute("color", new THREE.BufferAttribute(colors, 3, true));
    pointGeometry.computeBoundingSphere();

    const pointMaterial = new THREE.PointsMaterial({{
      size: DATA.pointSize,
      vertexColors: true,
      sizeAttenuation: true,
      transparent: true,
      opacity: 0.96,
      depthWrite: false
    }});
    const pointCloud = new THREE.Points(pointGeometry, pointMaterial);
    scene.add(pointCloud);

    const pathGeometry = new THREE.BufferGeometry();
    pathGeometry.setAttribute("position", new THREE.BufferAttribute(cameraPath, 3));
    const pathMaterial = new THREE.LineBasicMaterial({{ color: 0xd62728, transparent: true, opacity: 0.92 }});
    const cameraLine = new THREE.Line(pathGeometry, pathMaterial);
    scene.add(cameraLine);

    const frustumGeometry = new THREE.BufferGeometry();
    frustumGeometry.setAttribute("position", new THREE.BufferAttribute(frustums, 3));
    const frustumMaterial = new THREE.LineBasicMaterial({{ color: 0x2251cc, transparent: true, opacity: 0.5 }});
    const frustumLines = new THREE.LineSegments(frustumGeometry, frustumMaterial);
    scene.add(frustumLines);

    const center = new THREE.Vector3(DATA.center[0], DATA.center[1], DATA.center[2]);
    const radius = Math.max(DATA.bboxDiag * 0.72, 10.0);
    function resetView() {{
      camera.position.set(center.x + radius * 1.18, center.y - radius * 1.02, center.z + radius * 0.82);
      controls.target.copy(center);
      controls.update();
    }}
    resetView();

    document.getElementById("pointSize").addEventListener("input", (event) => {{
      pointMaterial.size = parseFloat(event.target.value);
    }});
    document.getElementById("showPath").addEventListener("change", (event) => {{
      cameraLine.visible = event.target.checked;
    }});
    document.getElementById("showFrustums").addEventListener("change", (event) => {{
      frustumLines.visible = event.target.checked;
    }});
    document.getElementById("autoRotate").addEventListener("change", (event) => {{
      controls.autoRotate = event.target.checked;
    }});
    document.getElementById("resetView").addEventListener("click", resetView);

    function animate() {{
      requestAnimationFrame(animate);
      if (controls.autoRotate) {{
        const offset = camera.position.clone().sub(controls.target);
        const rotationStep = 0.0085;
        offset.applyAxisAngle(new THREE.Vector3(0, 0, 1), rotationStep);
        camera.position.copy(controls.target.clone().add(offset));
        camera.lookAt(controls.target);
      }}
      controls.update();
      renderer.render(scene, camera);
    }}
    animate();

    window.addEventListener("resize", () => {{
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
      if (controls.handleResize) {{
        controls.handleResize();
      }}
    }});
  </script>
</body>
</html>
"""


def write_scene_readme(
    scene_dir: Path,
    scene_name: str,
    simplified_ply: str,
    html_name: str,
    image_name: str,
    gif_name: str,
    source_ply: Path,
    source_cameras: Path,
) -> None:
    content = "\n".join(
        [
            f"{scene_name} outputs",
            "",
            f"1. {simplified_ply}",
            "   Low-detail color point cloud for PowerPoint insertion.",
            "",
            f"2. {html_name}",
            "   Local drag-to-view viewer with camera path and frustums.",
            "",
            f"3. {image_name}",
            "   Camera pose overview image with perspective and top views.",
            "",
            f"4. {gif_name}",
            "   Turntable GIF with city, camera path, and frustums for slides.",
            "",
            f"Source point cloud: {source_ply}",
            f"Source cameras: {source_cameras}",
        ]
    )
    (scene_dir / "README.txt").write_text(content, encoding="utf-8")


def write_root_readme(output_root: Path, scene_names: list[str]) -> None:
    lines = [
        "CityGS visualization export",
        "",
        "Folders:",
    ]
    for name in scene_names:
        lines.append(f"- {name}")
    lines.extend(
        [
            "",
            "Usage:",
            "- Open each scene folder and double-click the HTML viewer for mouse-drag inspection.",
            "- Insert the simplified PLY into PowerPoint as a 3D model candidate.",
            "- Use the PNG image directly in slides for camera pose overview.",
        ]
    )
    (output_root / "README.txt").write_text("\n".join(lines), encoding="utf-8")


def process_scene(
    scene: SceneSpec,
    output_root: Path,
    target_points: int,
    opacity_min: float,
    oversample_factor: float,
    chunk_size: int,
    seed: int,
) -> None:
    print(f"\n=== Processing {scene.name} ===")
    xyz, rgb, _alpha = read_gaussian_sample(
        ply_path=scene.point_cloud_path,
        target_points=target_points,
        opacity_min=opacity_min,
        oversample_factor=oversample_factor,
        chunk_size=chunk_size,
        seed=seed,
    )
    if xyz.size == 0:
        raise RuntimeError(f"No points sampled from {scene.point_cloud_path}")

    cameras = load_cameras(scene.cameras_path)
    raw_camera_path, raw_frustum_lines = build_camera_geometry(
        cameras=cameras,
        scene_center=xyz.mean(axis=0),
        bbox_diag=float(np.linalg.norm(xyz.max(axis=0) - xyz.min(axis=0))),
    )
    alignment_origin, alignment_basis = estimate_alignment_transform(xyz, raw_camera_path)
    xyz = apply_alignment(xyz, alignment_origin, alignment_basis)
    camera_path = apply_alignment(raw_camera_path, alignment_origin, alignment_basis)
    frustum_lines = apply_alignment(raw_frustum_lines, alignment_origin, alignment_basis)

    scene_dir = output_root / scene.name
    scene_dir.mkdir(parents=True, exist_ok=True)
    simplified_ply = scene_dir / f"{scene.name}_simplified_points.ply"
    write_binary_color_ply(simplified_ply, xyz, rgb)
    print(f"Wrote {simplified_ply}")

    viewer_html = scene_dir / f"{scene.name}_viewer.html"
    html = build_viewer_html(
        scene_name=scene.name,
        xyz=xyz,
        rgb=rgb,
        camera_path=camera_path,
        frustum_lines=frustum_lines,
        vendor_path_prefix="../vendor",
    )
    viewer_html.write_text(html, encoding="utf-8")
    print(f"Wrote {viewer_html}")

    camera_pose_png = scene_dir / f"{scene.name}_camera_poses.png"
    make_camera_pose_figure(
        output_path=camera_pose_png,
        scene_name=scene.name,
        xyz=xyz,
        rgb=rgb,
        camera_path=camera_path,
        frustum_lines=frustum_lines,
        seed=seed + 7,
    )
    print(f"Wrote {camera_pose_png}")

    turntable_gif = scene_dir / f"{scene.name}_turntable.gif"
    make_turntable_gif(
        output_path=turntable_gif,
        scene_name=scene.name,
        xyz=xyz,
        rgb=rgb,
        camera_path=camera_path,
        frustum_lines=frustum_lines,
        seed=seed + 13,
    )
    print(f"Wrote {turntable_gif}")

    write_scene_readme(
        scene_dir=scene_dir,
        scene_name=scene.name,
        simplified_ply=simplified_ply.name,
        html_name=viewer_html.name,
        image_name=camera_pose_png.name,
        gif_name=turntable_gif.name,
        source_ply=scene.point_cloud_path,
        source_cameras=scene.cameras_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export CityGS models to dense local viewers.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/citygs_visualization"),
        help="Directory for generated viewers and PPT assets.",
    )
    parser.add_argument(
        "--target-points",
        type=int,
        default=400_000,
        help="Target number of points per exported scene.",
    )
    parser.add_argument(
        "--opacity-min",
        type=float,
        default=0.02,
        help="Minimum sigmoid(opacity) value kept during sampling.",
    )
    parser.add_argument(
        "--oversample-factor",
        type=float,
        default=1.65,
        help="Sample slightly above target, then weighted-downsample.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=200_000,
        help="Number of gaussian records read per chunk.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260623,
        help="Random seed for reproducible sampling.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    ensure_vendor_assets(output_root / "vendor")

    scene_names = []
    for index, scene in enumerate(DEFAULT_SCENES):
        process_scene(
            scene=scene,
            output_root=output_root,
            target_points=args.target_points,
            opacity_min=args.opacity_min,
            oversample_factor=args.oversample_factor,
            chunk_size=args.chunk_size,
            seed=args.seed + index * 1000,
        )
        scene_names.append(scene.name)

    write_root_readme(output_root, scene_names)
    print(f"\nAll outputs written to: {output_root}")


if __name__ == "__main__":
    main()
