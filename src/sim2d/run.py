import argparse
import math
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from src.sim2d.config import Sim2DConfig, K_from_cfg
from src.sim2d.motor_backend import SimMotorBackend
from src.sim2d.geometry import pixel_depth_to_cam_xyz
from src.sim2d.frames import cam_xyz_to_robot_xy, robot_xy_to_world_xy
from src.sim2d.detector_sim import world_from_detector_json, simulate_detector_json


# ----------------------------
# Helpers
# ----------------------------

def wrap_pi(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def apply_axis_cfg(xyz: np.ndarray, cfg: Sim2DConfig) -> np.ndarray:
    """Optional axis tweaks (keep defaults as camera frame)."""
    x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])

    if getattr(cfg, "AXIS_SWAP_XY", False):
        x, y = y, x

    if getattr(cfg, "AXIS_FLIP_X", False):
        x = -x
    if getattr(cfg, "AXIS_FLIP_Y", False):
        y = -y
    if getattr(cfg, "AXIS_FLIP_Z", False):
        z = -z

    return np.array([x, y, z], dtype=np.float32)


def select_target_obs(det_json: dict, cfg: Sim2DConfig) -> Optional[Tuple[float, float, float, int]]:
    """
    Returns (u, v, depth_m, det_id) for selected berry from detector json.
    Uses cfg.TARGET_SELECTION (default: "closest").
    """
    dets = det_json.get("detections", [])
    stats = det_json.get("statistics", {})

    if not dets:
        return None

    mode = getattr(cfg, "TARGET_SELECTION", "closest")
    if mode == "closest":
        tid = stats.get("closest_strawberry_id", None)
        if tid is None:
            tid = dets[0].get("id", 0)
    else:
        # fallback: first detection
        tid = dets[0].get("id", 0)

    chosen = None
    for d in dets:
        if d.get("id") == tid:
            chosen = d
            break
    if chosen is None:
        chosen = dets[0]

    bbox = chosen.get("bbox", {})
    depth = chosen.get("depth", {})

    u = float(bbox.get("center_x", cfg.CX))
    v = float(bbox.get("center_y", cfg.CY))
    z = float(depth.get("center_meters", depth.get("mean_meters", 1.0)))

    det_id = int(chosen.get("id", 0))
    return u, v, z, det_id


def obs_to_goal_world(u: float, v: float, depth_m: float, pose: np.ndarray, cfg: Sim2DConfig, K: np.ndarray) -> np.ndarray:
    """(u,v,depth) -> cam xyz -> robot xy -> world xy"""
    xyz_cam = pixel_depth_to_cam_xyz(u, v, depth_m, K)
    xyz_cam = apply_axis_cfg(xyz_cam, cfg)
    p_robot = cam_xyz_to_robot_xy(xyz_cam)        # [x_forward, y_left] in robot frame
    p_world = robot_xy_to_world_xy(p_robot, pose) # [x,y] in world
    return p_world


# ----------------------------
# Diff-drive "motor math" backend
# ----------------------------

@dataclass
class WheelCmd:
    wl: float  # rad/s
    wr: float  # rad/s


def vw_to_wheels(v: float, w: float, cfg: Sim2DConfig) -> WheelCmd:
    L = float(cfg.WHEEL_BASE_M)
    R = float(cfg.WHEEL_RADIUS_M)
    wl = (v - w * L / 2.0) / R
    wr = (v + w * L / 2.0) / R
    return WheelCmd(wl=wl, wr=wr)


def wheels_to_vw(cmd: WheelCmd, cfg: Sim2DConfig) -> Tuple[float, float]:
    L = float(cfg.WHEEL_BASE_M)
    R = float(cfg.WHEEL_RADIUS_M)
    v = R * (cmd.wl + cmd.wr) / 2.0
    w = R * (cmd.wr - cmd.wl) / L
    return float(v), float(w)


def integrate_pose(pose: np.ndarray, cmd: WheelCmd, dt: float, cfg: Sim2DConfig) -> np.ndarray:
    x, y, th = float(pose[0]), float(pose[1]), float(pose[2])
    v, w = wheels_to_vw(cmd, cfg)
    x += v * math.cos(th) * dt
    y += v * math.sin(th) * dt
    th = wrap_pi(th + w * dt)
    return np.array([x, y, th], dtype=np.float32)


# ----------------------------
# Controller + FSM
# ----------------------------

def compute_control_to_goal(pose: np.ndarray, goal_xy: np.ndarray, cfg: Sim2DConfig) -> Tuple[float, float]:
    """
    Returns desired (v_des, w_des) without motor/acc limiting.
    Motor-like limiting is applied later via limit_vw(...).
    - linear slowdown starts at cfg.SLOWDOWN_START_M
    - stop boundary at cfg.STOP_RADIUS_M
    """
    x, y, th = float(pose[0]), float(pose[1]), float(pose[2])
    gx, gy = float(goal_xy[0]), float(goal_xy[1])

    dx = gx - x
    dy = gy - y
    dist = math.hypot(dx, dy)

    stop_r = float(cfg.STOP_RADIUS_M)
    slow_r = float(cfg.SLOWDOWN_START_M)

    # distance to "stop boundary"
    d_err = max(0.0, dist - stop_r)

    v_max = float(cfg.MAX_V)

    # desired linear speed (linear ramp down inside slowdown zone)
    if d_err >= slow_r:
        v_des = v_max
    else:
        v_des = v_max * (d_err / max(1e-6, slow_r))

    # heading control
    ang_to_goal = math.atan2(dy, dx)
    ang_err = wrap_pi(ang_to_goal - th)

    k_ang = 2.0
    w_max = float(getattr(cfg, "MAX_W", 0.8))
    w_des = clamp(k_ang * ang_err, -w_max, w_max)

    return v_des, w_des
def run(args) -> int:
    cfg = Sim2DConfig()
    K = K_from_cfg(cfg)

    pose = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    pose0 = pose.copy()

    backend = SimMotorBackend(cfg)

    world = world_from_detector_json(args.json, K, pose0)

    # detector timing
    det_period = 1.0 / max(1e-6, float(cfg.DETECTOR_RATE_HZ))
    det_latency = max(0.0, float(cfg.DETECTOR_LATENCY_S))
    det_queue = deque()  # (available_time, json)

    # control timing
    ctrl_period = 1.0 / max(1e-6, float(cfg.CONTROL_RATE_HZ))
    dt = float(cfg.SIM_DT)

    t = 0.0
    next_det_t = 0.0
    next_ctrl_t = 0.0

    last_det = None
    state = "SEARCH"
    wheel_cmd = WheelCmd(0.0, 0.0)

    # visualization (optional)
    viz = False
    if not args.headless:
        try:
            import matplotlib.pyplot as plt
            viz = True
        except Exception:
            viz = False

    if viz:
        import matplotlib.pyplot as plt
        plt.ion()
        fig, ax = plt.subplots()
        ax.set_aspect("equal", adjustable="box")
        ax.set_title("sim2d top-down (SEARCH -> APPROACH -> STOP)")

        # draw berries
        berries = world.get("berries", world.get("world_berries", []))
        bx = [float(b.p_world[0]) for b in berries]
        by = [float(b.p_world[1]) for b in berries]
        ax.scatter(bx, by, s=50, marker="o", label="berries")

        robot_pt, = ax.plot([pose[0]], [pose[1]], marker="o", markersize=8, linestyle="None", label="robot")
        heading_ln, = ax.plot([], [], "-", linewidth=2, label="heading")
        path_ln, = ax.plot([], [], "-", linewidth=1, label="path")
        goal_pt, = ax.plot([], [], marker="x", markersize=10, linestyle="None", label="goal(obs)")
        txt = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")

        ax.legend(loc="upper right")
        ax.set_xlim(-1.0, 2.0)
        ax.set_ylim(-1.5, 1.5)

        path_x, path_y = [], []

    # loop
    steps = 0
    max_steps = int(args.steps)

    while steps < max_steps:
        # 1) detector produces observation at DETECTOR_RATE_HZ
        if t + 1e-9 >= next_det_t:
            det = simulate_detector_json(world, pose, cfg, K)
            det_queue.append((t + det_latency, det))
            next_det_t += det_period

        # 2) deliver observation after latency
        while det_queue and det_queue[0][0] <= t + 1e-9:
            _, last_det = det_queue.popleft()

        # 3) controller update at CONTROL_RATE_HZ
        if t + 1e-9 >= next_ctrl_t:
            obs = None if last_det is None else select_target_obs(last_det, cfg)

            if state == "SEARCH":
                if obs is None:
                    # rotate in place (motor-like limits)
                    v_des, w_des = 0.0, float(getattr(cfg, 'SEARCH_W', 0.8))
                    out = backend.set_cmd(v_des, w_des, ctrl_period)
                    wheel_cmd = vw_to_wheels(out.v, out.w, cfg)

                else:
                    state = "APPROACH"

            if state == "APPROACH":
                if obs is None:
                    # lost target -> search
                    state = "SEARCH"
                    v_des, w_des = 0.0, float(getattr(cfg, "SEARCH_W", 0.8))
                    out = backend.set_cmd(v_des, w_des, ctrl_period)
                    wheel_cmd = vw_to_wheels(out.v, out.w, cfg)

                else:
                    u, v, z, det_id = obs
                    goal_xy = obs_to_goal_world(u, v, z, pose, cfg, K)

                    # stop condition (distance check in world)
                    dx = float(goal_xy[0] - pose[0])
                    dy = float(goal_xy[1] - pose[1])
                    dist = math.hypot(dx, dy)

                    if dist <= float(cfg.STOP_RADIUS_M) + float(cfg.STOP_EPS_M):
                        state = "STOP"
                        wheel_cmd = WheelCmd(0.0, 0.0)
                    else:
                        v_des, w_des = compute_control_to_goal(pose, goal_xy, cfg)
                        out = backend.set_cmd(v_des, w_des, ctrl_period)
                        wheel_cmd = vw_to_wheels(out.v, out.w, cfg)

            if state == "STOP":
                wheel_cmd = WheelCmd(0.0, 0.0)
                backend.reset()

            next_ctrl_t += ctrl_period

        # 4) physics/odometry integration (always)
        pose = integrate_pose(pose, wheel_cmd, dt, cfg)
        t += dt
        steps += 1

        # 5) visualization update
        if viz and steps % 2 == 0:
            path_x.append(float(pose[0]))
            path_y.append(float(pose[1]))

            robot_pt.set_data([pose[0]], [pose[1]])
            hx = float(pose[0] + 0.2 * math.cos(float(pose[2])))
            hy = float(pose[1] + 0.2 * math.sin(float(pose[2])))
            heading_ln.set_data([pose[0], hx], [pose[1], hy])
            path_ln.set_data(path_x, path_y)

            # show current obs goal if available
            if last_det is not None:
                obs = select_target_obs(last_det, cfg)
                if obs is not None:
                    u, v, z, _ = obs
                    gxy = obs_to_goal_world(u, v, z, pose, cfg, K)
                    goal_pt.set_data([gxy[0]], [gxy[1]])

            txt.set_text(f"t={t:0.2f}s  state={state}  pose=({pose[0]:0.2f},{pose[1]:0.2f},{pose[2]:0.2f})")
            import matplotlib.pyplot as plt
            plt.pause(0.001)

        if state == "STOP":
            break

    print(f"DONE: steps={steps}, t={t:0.2f}s, state={state}, pose={pose}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="Path to detector json (e.g. results/strawberries_sample.json)")
    ap.add_argument("--steps", type=int, default=3000, help="Max sim steps")
    ap.add_argument("--headless", action="store_true", help="Run without visualization")
    args = ap.parse_args()
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
