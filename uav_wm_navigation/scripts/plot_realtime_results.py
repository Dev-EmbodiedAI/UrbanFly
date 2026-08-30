from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import yaml


def _load_run(path: Path) -> tuple[dict, list[dict]]:
    return (
        json.loads((path / "summary.json").read_text(encoding="utf-8")),
        json.loads((path / "monitoring.json").read_text(encoding="utf-8")),
    )


def _style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False, "axes.edgecolor": "#425466", "axes.labelcolor": "#233447",
        "xtick.color": "#425466", "ytick.color": "#425466", "figure.facecolor": "white",
        "axes.facecolor": "#F7F9FC", "grid.color": "#D9E1EA", "grid.alpha": 0.8,
    })


def speed_ladder(runs: list[Path], output: Path) -> None:
    summaries = [_load_run(path)[0] for path in runs]
    targets = np.asarray([item["metrics"]["speed_target_mps"] for item in summaries])
    p50 = np.asarray([item["metrics"]["speed_p50_mps"] for item in summaries])
    p95 = np.asarray([item["metrics"]["speed_p95_mps"] for item in summaries])
    sensor = np.asarray([item["metrics"]["sensor_rate_hz"] for item in summaries])
    planner = np.asarray([item["metrics"]["planner_rate_hz"] for item in summaries])
    control = np.asarray([item["metrics"]["control_rate_hz"] for item in summaries])
    latency = np.asarray([item["metrics"]["planner_latency_p95_ms"] for item in summaries])
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), constrained_layout=True)
    x = np.arange(len(targets)); width = 0.25
    axes[0].bar(x - width, targets, width, label="目标速度", color="#334E68")
    axes[0].bar(x, p50, width, label="实际 P50", color="#2CB67D")
    axes[0].bar(x + width, p95, width, label="实际 P95", color="#F4A261")
    axes[0].set_xticks(x, [f"{value:g} m/s" for value in targets])
    axes[0].set_ylabel("速度 / (m/s)"); axes[0].set_title("速度阶梯：目标与实际")
    axes[0].grid(axis="y"); axes[0].legend(frameon=False, ncol=3, loc="upper left")
    for index, value in enumerate(p95):
        axes[0].text(index + width, value + 0.10, f"{value:.2f}", ha="center", fontsize=9)
    axes[1].plot(targets, sensor, "o-", lw=2.2, label="深度传感", color="#3A86FF")
    axes[1].plot(targets, planner, "o-", lw=2.2, label="YOPO 规划", color="#FF006E")
    axes[1].plot(targets, control, "o-", lw=2.2, label="控制下发", color="#2CB67D")
    axes[1].axhline(50, color="#2CB67D", ls="--", lw=1, alpha=0.5)
    axes[1].set_xlabel("配置速度 / (m/s)"); axes[1].set_ylabel("实测频率 / Hz")
    axes[1].set_title("异步闭环频率"); axes[1].grid(); axes[1].legend(frameon=False)
    second = axes[1].twinx()
    second.plot(targets, latency, "D--", color="#7B2CBF", label="推理 P95")
    second.set_ylabel("YOPO 推理 P95 / ms", color="#7B2CBF"); second.set_ylim(0, max(80, latency.max() * 1.3))
    fig.suptitle("纯 YOPO 三档真实闭环结果", fontsize=16, fontweight="bold", color="#102A43")
    fig.savefig(output, dpi=220, bbox_inches="tight"); plt.close(fig)


def trajectory(run: Path, config_path: Path, output: Path) -> None:
    _, monitor = _load_run(run)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    positions = np.asarray([item["position_nwu"] for item in monitor], dtype=float)
    start = np.asarray(config["flight_start_nwu"], dtype=float); goal = np.asarray(config["goal_nwu"], dtype=float)
    direction = goal[:2] - start[:2]; length = np.linalg.norm(direction); forward = direction / length
    left = np.array([-forward[1], forward[0]])
    forest = config["procedural_forest"]
    obstacles = np.asarray([
        start[:2] + along * forward + lateral * left
        for along, lateral in zip(forest["challenge_along_m"], forest["challenge_lateral_m"])
    ])
    progress = (positions[:, :2] - start[:2]) @ forward
    lateral = (positions[:, :2] - start[:2]) @ left
    fig, axes = plt.subplots(2, 1, figsize=(11.6, 7.8), height_ratios=[2.1, 1], constrained_layout=True)
    axes[0].plot(positions[:, 0], positions[:, 1], color="#2CB67D", lw=3, label="实际轨迹")
    axes[0].plot([start[0], goal[0]], [start[1], goal[1]], "--", color="#829AB1", lw=1.5, label="直线参考")
    axes[0].scatter(obstacles[:, 0], obstacles[:, 1], s=260, marker="X", color="#D62828", edgecolor="white", lw=1.5, label="强制绕障点")
    axes[0].scatter(*start[:2], s=130, marker="^", color="#102A43", label="起点")
    axes[0].scatter(*goal[:2], s=180, marker="*", color="#7B2CBF", label="目标")
    xy_all = np.vstack([positions[:, :2], start[:2], goal[:2], obstacles])
    xy_span = np.ptp(xy_all, axis=0)
    xy_pad = np.maximum(3.0, xy_span * 0.08)
    axes[0].set_xlim(xy_all[:, 0].min() - xy_pad[0], xy_all[:, 0].max() + xy_pad[0])
    axes[0].set_ylim(xy_all[:, 1].min() - xy_pad[1], xy_all[:, 1].max() + xy_pad[1])
    axes[0].grid(); axes[0].legend(frameon=False, ncol=3)
    axes[0].set_xlabel("NWU x / m"); axes[0].set_ylabel("NWU y / m"); axes[0].set_title("6 m/s 配置：120 m 三障碍真实闭环轨迹")
    axes[1].plot(progress, lateral, color="#3A86FF", lw=2.0)
    for along, offset in zip(forest["challenge_along_m"], forest["challenge_lateral_m"]):
        axes[1].axvline(along, color="#D62828", lw=1.2, ls="--")
        axes[1].scatter([along], [offset], marker="X", color="#D62828", s=90)
    axes[1].axhline(0, color="#829AB1", lw=1); axes[1].grid()
    axes[1].set_xlabel("路线进度 / m"); axes[1].set_ylabel("横向偏移 / m"); axes[1].set_title("绕障横向响应（UrbanFly 位姿）")
    fig.savefig(output, dpi=220, bbox_inches="tight"); plt.close(fig)


def timeseries(run: Path, output: Path) -> None:
    with h5py.File(run / "telemetry.h5", "r") as handle:
        control_t = handle["control/monotonic_s"][:]; control_t -= control_t[0]
        velocity = np.linalg.norm(handle["control/actual_velocity_nwu"][:], axis=1)
        yaw_rate = np.abs(handle["control/command_yaw_rate_rps"][:])
        plan_t = handle["plans/elapsed_s"][:]
        latency = handle["plans/planner_latency_ms"][:]
        selected = handle["plans/selected_index"][:]
    fig, axes = plt.subplots(3, 1, figsize=(11.6, 7.8), sharex=True, constrained_layout=True)
    axes[0].plot(control_t, velocity, color="#2CB67D", lw=1.8); axes[0].axhline(6, color="#334E68", ls="--", lw=1)
    axes[0].set_ylabel("速度 / (m/s)"); axes[0].set_title("6 m/s 配置回合：速度、推理与候选切换"); axes[0].grid()
    axes[1].plot(plan_t, latency, color="#7B2CBF", lw=1.3); axes[1].axhline(80, color="#D62828", ls="--", lw=1)
    axes[1].set_ylabel("推理 / ms"); axes[1].grid()
    axes[2].step(plan_t, selected, where="post", color="#F4A261", lw=1.2, label="selected lattice")
    axes[2].plot(control_t, yaw_rate * 5.0, color="#3A86FF", alpha=0.7, lw=1.0, label="|yaw rate| ×5")
    axes[2].set_xlabel("飞行时间 / s"); axes[2].set_ylabel("候选 ID / 缩放量"); axes[2].set_yticks(range(0, 15, 2)); axes[2].grid(); axes[2].legend(frameon=False, ncol=2)
    fig.savefig(output, dpi=220, bbox_inches="tight"); plt.close(fig)


def keyframes(run: Path, output: Path) -> None:
    cap = cv2.VideoCapture(str(run / "front_camera_60fps.mp4"))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for fraction in (0.08, 0.28, 0.52, 0.78, 0.94):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int((count - 1) * fraction))
        ok, frame = cap.read()
        if ok:
            frame = cv2.resize(frame, (640, 360))
            cv2.putText(frame, f"t={fraction * count / 60.0:.1f}s", (20, 36), cv2.FONT_HERSHEY_DUPLEX, 0.8, (20, 25, 30), 4, cv2.LINE_AA)
            cv2.putText(frame, f"t={fraction * count / 60.0:.1f}s", (20, 36), cv2.FONT_HERSHEY_DUPLEX, 0.8, (245, 245, 245), 1, cv2.LINE_AA)
            frames.append(frame)
    cap.release()
    canvas = np.full((720, 1920, 3), 245, dtype=np.uint8)
    for index, frame in enumerate(frames[:3]): canvas[0:360, index * 640:(index + 1) * 640] = frame
    for index, frame in enumerate(frames[3:]): canvas[360:720, (index + 1) * 640:(index + 2) * 640] = frame
    cv2.imwrite(str(output), canvas)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier3", type=Path, required=True); parser.add_argument("--tier4p5", type=Path, required=True)
    parser.add_argument("--tier6", type=Path, required=True); parser.add_argument("--tier6-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True); _style()
    speed_ladder([args.tier3, args.tier4p5, args.tier6], args.output_dir / "speed_ladder_summary.png")
    trajectory(args.tier6, args.tier6_config, args.output_dir / "tier6_trajectory.png")
    timeseries(args.tier6, args.output_dir / "tier6_timeseries.png")
    keyframes(args.tier6, args.output_dir / "tier6_front_keyframes.jpg")
    print(args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
