"""
Joint-space RRT for Franka Panda in robosuite Lift environment.
Goal: reach 8 cm above the red cube using task-space goal checking.

Run with:
  /path/to/env/python rrt_panda.py [--step_size 0.1] [--max_iters 5000] [--render] [--show]
"""

from __future__ import annotations

import argparse
import copy
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

# Output directories, resolved relative to this script's location
_SCRIPT_DIR = Path(__file__).parent
_DIR_TREES  = _SCRIPT_DIR / "trees"
_DIR_VIDEOS = _SCRIPT_DIR / "videos"
_DIR_PLANS  = _SCRIPT_DIR / "plans"

for _d in (_DIR_TREES, _DIR_VIDEOS, _DIR_PLANS):
    _d.mkdir(parents=True, exist_ok=True)

import mujoco
import numpy as np
import robosuite as suite
from robosuite import load_part_controller_config

try:
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend; safe outside the main thread
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class RRTConfig:
    step_size: float = 0.1        # rad, max joint change per extension step
    goal_threshold: float = 0.02  # m, EEF distance to declare goal reached
    goal_bias: float = 0.15       # fraction of iters biased toward IK seed
    max_iters: int = 5000
    seed: int = 0
    interp_step: float = 0.02     # rad, step for edge collision interpolation
    exec_tol: float = 0.02        # rad, joint error for waypoint done
    exec_max_steps: int = 150     # env steps per waypoint


@dataclass
class Node:
    q: np.ndarray
    parent: Optional[int] = None
    eef_pos: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def make_env(render: bool = False) -> suite.environments.base.MujocoEnv:
    """
    Create Lift env with JOINT_POSITION controller in absolute mode.
    This lets execute_path command joint angles directly without oscillation.
    """
    base_cfg = suite.load_composite_controller_config(robot="Panda")
    cfg = copy.deepcopy(base_cfg)
    jp_cfg = load_part_controller_config(default_controller="JOINT_POSITION")
    # Absolute mode: action IS the goal joint angle (no delta accumulation)
    jp_cfg["input_type"] = "absolute"
    jp_cfg["output_max"] = 5.0   # large enough to cover full joint range
    jp_cfg["output_min"] = -5.0
    jp_cfg["kp"] = 50            # moderate gain — visible motion in viewer
    jp_cfg["gripper"] = {"type": "GRIP"}
    cfg["body_parts"]["right"] = jp_cfg

    env = suite.make(
        env_name="Lift",
        robots="Panda",
        controller_configs=cfg,
        has_renderer=render,
        has_offscreen_renderer=True,   # always on: needed for video recording
        use_camera_obs=False,
        reward_shaping=False,
        horizon=20000,
    )
    return env


def get_arm_qpos_idx() -> np.ndarray:
    """Panda arm joints occupy qpos indices 0–6."""
    return np.arange(7, dtype=int)


def get_joint_limits(env, arm_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Read joint limits from MuJoCo model for the 7 arm joints."""
    model = env.sim.model
    lo, hi = [], []
    for j in range(len(arm_idx)):
        l, h = model.jnt_range[j]
        lo.append(l)
        hi.append(h)
    return np.array(lo), np.array(hi)


def get_eef_site_id(env) -> int:
    """Return site id for the Panda EEF grip site."""
    model = env.sim.model
    for sid in range(model.nsite):
        name = model.site_id2name(sid)
        if name and "grip_site" in name and "cylinder" not in name:
            return sid
    raise RuntimeError("Could not find gripper site in MuJoCo model.")


def get_cube_body_id(env) -> int:
    """Return body id for cube_main."""
    model = env.sim.model
    for bid in range(model.nbody):
        name = model.body_id2name(bid)
        if name and "cube_main" in name:
            return bid
    raise RuntimeError("Could not find cube_main body in MuJoCo model.")


def build_contact_sets(env) -> tuple[set[int], set[int]]:
    """Build body-id sets for robot links and obstacles (table, cube)."""
    model = env.sim.model
    robot_bodies: set[int] = set()
    obstacle_bodies: set[int] = set()
    for bid in range(model.nbody):
        name = model.body_id2name(bid)
        if not name:
            continue
        if "robot0" in name or "gripper0" in name:
            robot_bodies.add(bid)
        if "table" in name or "cube" in name:
            obstacle_bodies.add(bid)
    return robot_bodies, obstacle_bodies


# ---------------------------------------------------------------------------
# IK for goal seed
# ---------------------------------------------------------------------------

def ik_goal_seed(
    env,
    target_pos: np.ndarray,
    arm_idx: np.ndarray,
    eef_sid: int,
    lo: np.ndarray,
    hi: np.ndarray,
    q_init: Optional[np.ndarray] = None,
    max_iters: int = 500,
    tol: float = 0.008,
    lam: float = 0.05,
    alpha: float = 0.8,
) -> Optional[np.ndarray]:
    """
    Jacobian damped-least-squares IK to get a goal joint configuration.
    Returns q s.t. fk_eef(q) ≈ target_pos, or None if not converged.
    """
    model_raw = env.sim.model._model
    data_raw = env.sim.data._data
    nv = model_raw.nv
    n = len(arm_idx)

    q = q_init.copy() if q_init is not None else env.sim.data.qpos[arm_idx].copy()

    for _ in range(max_iters):
        env.sim.data.qpos[arm_idx] = q
        mujoco.mj_forward(model_raw, data_raw)
        eef = env.sim.data.site_xpos[eef_sid].copy()
        err = target_pos - eef
        if np.linalg.norm(err) < tol:
            break
        jacp = np.zeros((3, nv))
        mujoco.mj_jacSite(model_raw, data_raw, jacp, None, eef_sid)
        J = jacp[:, :n]
        JJT = J @ J.T + lam ** 2 * np.eye(3)
        dq = J.T @ np.linalg.solve(JJT, err)
        q = np.clip(q + alpha * dq, lo, hi)

    env.sim.data.qpos[arm_idx] = q
    mujoco.mj_forward(model_raw, data_raw)
    final_dist = float(np.linalg.norm(env.sim.data.site_xpos[eef_sid] - target_pos))
    if final_dist < tol * 4:
        print(f"IK seed: dist={final_dist:.4f}m, q={np.round(q, 3)}")
        return q.copy()
    print(f"IK did not converge (dist={final_dist:.4f}m); using uniform sampling only")
    return None


# ---------------------------------------------------------------------------
# Forward kinematics and collision
# ---------------------------------------------------------------------------

def fk_eef(
    env,
    q: np.ndarray,
    arm_idx: np.ndarray,
    eef_sid: int,
) -> np.ndarray:
    """Set joint angles and return EEF position via MuJoCo FK."""
    env.sim.data.qpos[arm_idx] = q
    mujoco.mj_forward(env.sim.model._model, env.sim.data._data)
    return env.sim.data.site_xpos[eef_sid].copy()


def is_collision(
    env,
    q: np.ndarray,
    arm_idx: np.ndarray,
    robot_bodies: set[int],
    obstacle_bodies: set[int],
) -> bool:
    """Return True if q is in collision (self-collision or robot vs obstacle)."""
    env.sim.data.qpos[arm_idx] = q
    mujoco.mj_forward(env.sim.model._model, env.sim.data._data)

    data = env.sim.data
    if data.ncon == 0:
        return False

    model = env.sim.model
    for i in range(data.ncon):
        c = data.contact[i]
        b1 = model.geom_bodyid[c.geom1]
        b2 = model.geom_bodyid[c.geom2]
        if b1 == b2:
            continue
        if b1 in robot_bodies and b2 in robot_bodies:
            return True
        if (b1 in robot_bodies and b2 in obstacle_bodies) or \
           (b2 in robot_bodies and b1 in obstacle_bodies):
            return True
    return False


def edge_collision_free(
    env,
    q_from: np.ndarray,
    q_to: np.ndarray,
    arm_idx: np.ndarray,
    robot_bodies: set[int],
    obstacle_bodies: set[int],
    interp_step: float,
) -> bool:
    """Interpolate the edge and check each intermediate config is collision-free."""
    diff = q_to - q_from
    dist = np.linalg.norm(diff)
    if dist < 1e-9:
        return not is_collision(env, q_to, arm_idx, robot_bodies, obstacle_bodies)
    n_steps = max(2, int(np.ceil(dist / interp_step)))
    for k in range(n_steps + 1):
        q_mid = q_from + diff * (k / n_steps)
        if is_collision(env, q_mid, arm_idx, robot_bodies, obstacle_bodies):
            return False
    return True


# ---------------------------------------------------------------------------
# RRT
# ---------------------------------------------------------------------------

def nearest_node_idx(tree: list[Node], q: np.ndarray) -> int:
    dists = np.array([np.linalg.norm(nd.q - q) for nd in tree])
    return int(np.argmin(dists))


def steer(q_from: np.ndarray, q_to: np.ndarray, step_size: float) -> np.ndarray:
    diff = q_to - q_from
    dist = np.linalg.norm(diff)
    if dist < 1e-9:
        return q_from.copy()
    return q_from + diff * (min(step_size, dist) / dist)


def reconstruct_path(tree: list[Node], goal_idx: int) -> list[np.ndarray]:
    path: list[np.ndarray] = []
    idx: Optional[int] = goal_idx
    while idx is not None:
        path.append(tree[idx].q.copy())
        idx = tree[idx].parent
    path.reverse()
    return path


def rrt(
    env,
    q_start: np.ndarray,
    target_pos: np.ndarray,
    arm_idx: np.ndarray,
    eef_sid: int,
    cfg: RRTConfig,
    rng: np.random.Generator,
    lo: np.ndarray,
    hi: np.ndarray,
    q_goal_seed: Optional[np.ndarray] = None,
) -> tuple[Optional[list[np.ndarray]], list[Node], int]:
    """
    Joint-space RRT with task-space goal check.
    Goal biasing uses the IK seed q_goal_seed when provided.
    Returns (path_or_None, tree, iterations_used).
    """
    robot_bodies, obstacle_bodies = build_contact_sets(env)

    start_eef = fk_eef(env, q_start, arm_idx, eef_sid)
    tree: list[Node] = [Node(q=q_start.copy(), parent=None, eef_pos=start_eef)]

    print(f"\nRRT  step={cfg.step_size}  thr={cfg.goal_threshold}m  "
          f"bias={cfg.goal_bias}  max_iters={cfg.max_iters}")
    print(f"Start EEF:  {np.round(start_eef, 3)}")
    print(f"Target pos: {np.round(target_pos, 3)}")
    print(f"Init dist:  {np.linalg.norm(start_eef - target_pos):.3f} m\n")

    t0 = time.perf_counter()
    iters_used = 0

    for iteration in range(cfg.max_iters):
        iters_used = iteration + 1

        # Biased sampling: steer toward IK seed + Gaussian noise
        if q_goal_seed is not None and rng.random() < cfg.goal_bias:
            q_rand = q_goal_seed + rng.normal(0, 0.08, size=len(q_goal_seed))
            q_rand = np.clip(q_rand, lo, hi)
        else:
            q_rand = rng.uniform(lo, hi)

        near_idx = nearest_node_idx(tree, q_rand)
        q_near = tree[near_idx].q
        q_new = steer(q_near, q_rand, cfg.step_size)

        if not edge_collision_free(
            env, q_near, q_new, arm_idx, robot_bodies, obstacle_bodies, cfg.interp_step
        ):
            continue

        eef_new = fk_eef(env, q_new, arm_idx, eef_sid)
        tree.append(Node(q=q_new.copy(), parent=near_idx, eef_pos=eef_new))
        new_idx = len(tree) - 1

        dist_to_goal = float(np.linalg.norm(eef_new - target_pos))
        if dist_to_goal < cfg.goal_threshold:
            elapsed = time.perf_counter() - t0
            print(f"[SUCCESS] iter={iters_used}  tree={len(tree)}  "
                  f"EEF_dist={dist_to_goal:.4f}m  time={elapsed:.2f}s")
            return reconstruct_path(tree, new_idx), tree, iters_used

        if iters_used % 500 == 0:
            best = min(
                np.linalg.norm(n.eef_pos - target_pos)
                for n in tree if n.eef_pos is not None
            )
            print(f"  iter={iters_used}  tree={len(tree)}  best_dist={best:.3f}m")

    elapsed = time.perf_counter() - t0
    print(f"[FAILURE] Max iters={cfg.max_iters}  tree={len(tree)}  time={elapsed:.2f}s")
    return None, tree, iters_used


# ---------------------------------------------------------------------------
# Video recording helpers
# ---------------------------------------------------------------------------

# Video frame dimensions
_VID_H, _VID_W = 480, 640
_CAM = "frontview"


def _capture_frame(
    env,
    elapsed: float,
    label: str,
    joint_err: float,
    eef_dist: float,
) -> np.ndarray:
    """
    Grab one offscreen frame, flip it right-side-up, convert to BGR,
    and burn in the HUD overlay.  Returns an (H, W, 3) uint8 BGR array.
    """
    rgb = env.sim.render(height=_VID_H, width=_VID_W, camera_name=_CAM)
    bgr = cv2.cvtColor(rgb[::-1], cv2.COLOR_RGB2BGR)   # flip + convert

    # Semi-transparent dark banner at the top
    overlay = bgr.copy()
    cv2.rectangle(overlay, (0, 0), (_VID_W, 60), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, bgr, 0.45, 0, bgr)

    font   = cv2.FONT_HERSHEY_SIMPLEX
    white  = (255, 255, 255)
    yellow = (0, 220, 255)

    cv2.putText(bgr, f"t = {elapsed:6.2f} s", (10, 20),  font, 0.55, white,  1)
    cv2.putText(bgr, label,                    (10, 40),  font, 0.55, yellow, 1)
    cv2.putText(bgr, f"joint err: {joint_err:.3f} rad  EEF dist: {eef_dist:.3f} m",
                (220, 40), font, 0.50, white, 1)

    # Bottom-right datetime stamp
    ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    cv2.putText(bgr, ts, (_VID_W - 210, _VID_H - 8), font, 0.42, (180, 180, 180), 1)

    return bgr


def save_video(frames: list[np.ndarray], out_path: str, fps: int) -> None:
    """
    Write BGR frames to an MP4 using H.264 (avc1) for broad player compatibility.
    Falls back to mp4v if avc1 is unavailable on the current platform.
    """
    if not frames:
        print("No frames to save.")
        return

    # Prefer H.264 — natively supported by QuickTime, browsers, and VLC
    for fourcc_str in ("avc1", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        writer = cv2.VideoWriter(out_path, fourcc, fps, (_VID_W, _VID_H))
        if writer.isOpened():
            break
        writer.release()
    else:
        print("Could not open a VideoWriter — skipping video save.")
        return

    for f in frames:
        writer.write(f)
    writer.release()

    import os
    size_mb = os.path.getsize(out_path) / 1_048_576
    print(f"Video saved to {out_path}  "
          f"({len(frames)} frames @ {fps} fps, "
          f"{len(frames)/fps:.1f} s, {size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# Path execution
# ---------------------------------------------------------------------------

def _hold_frames(
    env,
    steps: int,
    elapsed_start: float,
    label: str,
    eef_sid: int,
    target_pos: np.ndarray,
    render: bool,
    frames: list[np.ndarray],
) -> float:
    """Step without moving the arm; collect frames and return updated elapsed."""
    action = np.zeros(env.action_dim)
    action[:7] = env.sim.data.qpos[:7].copy()
    t0 = time.perf_counter()
    for _ in range(steps):
        env.step(action)
        if render:
            env.render()
        eef = env.sim.data.site_xpos[eef_sid].copy()
        dist = float(np.linalg.norm(eef - target_pos))
        elapsed = elapsed_start + time.perf_counter() - t0
        frames.append(_capture_frame(env, elapsed, label, 0.0, dist))
    return elapsed_start + time.perf_counter() - t0


def execute_path(
    env,
    path: list[np.ndarray],
    arm_idx: np.ndarray,
    eef_sid: int,
    target_pos: np.ndarray,
    cfg: RRTConfig,
    render: bool,
    frames: list[np.ndarray],
) -> np.ndarray:
    """
    Execute planned path via absolute JOINT_POSITION controller.
    Captures every step into `frames` for video export.
    action_dim=8: first 7 = arm joints, last 1 = gripper (0 = closed).
    """
    print(f"\nExecuting {len(path)} waypoints...")
    action_dim = env.action_dim
    n = len(arm_idx)
    n_wp = len(path)
    t0 = time.perf_counter()

    # 1 s hold at start so the viewer has time to open
    if render:
        print("  [viewer] holding at start pose for 1 s...")
    elapsed = _hold_frames(env, 20, 0.0, "Stabilised at q_start",
                           eef_sid, target_pos, render, frames)

    for wi, q_target in enumerate(path):
        label = f"WP {wi+1}/{n_wp}"
        for _ in range(cfg.exec_max_steps):
            q_curr = env.sim.data.qpos[arm_idx].copy()
            err = float(np.linalg.norm(q_curr - q_target))
            if err < cfg.exec_tol:
                break
            action = np.zeros(action_dim)
            action[:n] = q_target
            env.step(action)
            if render:
                env.render()
            eef = env.sim.data.site_xpos[eef_sid].copy()
            dist = float(np.linalg.norm(eef - target_pos))
            elapsed = time.perf_counter() - t0
            frames.append(_capture_frame(env, elapsed, label, err, dist))

        q_curr = env.sim.data.qpos[arm_idx].copy()
        err_final = float(np.linalg.norm(q_curr - q_target))
        print(f"  wp {wi+1:3d}/{n_wp}: joint_err={err_final:.4f} rad")

    # 3 s hold at final pose
    if render:
        print("  [viewer] holding at final pose for 3 s...")
    elapsed = _hold_frames(env, 60, elapsed, "Goal reached — final pose",
                           eef_sid, target_pos, render, frames)

    return env.sim.data.site_xpos[eef_sid].copy()


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize(
    tree: list[Node],
    path: Optional[list[np.ndarray]],
    target_pos: np.ndarray,
    cube_pos: np.ndarray,
    arm_idx: np.ndarray,
    eef_sid: int,
    env,
    show: bool,
    out_path: str,
) -> None:
    if not HAS_MPL:
        print("matplotlib not available; skipping visualization.")
        return

    node_pos = np.array([
        n.eef_pos if n.eef_pos is not None else np.zeros(3)
        for n in tree
    ])

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    # Tree edges (thin blue)
    edge_segs = [
        [node_pos[nd.parent], node_pos[i]]
        for i, nd in enumerate(tree)
        if nd.parent is not None
    ]
    if edge_segs:
        ax.add_collection3d(
            Line3DCollection(edge_segs, colors="steelblue", linewidths=0.4, alpha=0.4)
        )

    # Tree nodes (green)
    ax.scatter(node_pos[:, 0], node_pos[:, 1], node_pos[:, 2],
               c="green", s=2, alpha=0.5, zorder=2)

    # Solution path (thick red)
    if path is not None and len(path) > 1:
        pp = np.array([fk_eef(env, q, arm_idx, eef_sid) for q in path])
        path_segs = [[pp[k], pp[k + 1]] for k in range(len(pp) - 1)]
        ax.add_collection3d(
            Line3DCollection(path_segs, colors="red", linewidths=2.5, zorder=5)
        )
        ax.scatter(pp[:, 0], pp[:, 1], pp[:, 2], c="red", s=10, zorder=6)

    # Start and goal markers
    ax.scatter(*node_pos[0], c="black", s=100, marker="s", zorder=7, label="Start")
    ax.scatter(*target_pos, c="gold", s=250, marker="*", zorder=8, label="Goal")

    # Cube wireframe (red box)
    h = 0.02
    cx, cy, cz = cube_pos
    corners = np.array([
        [cx + dx, cy + dy, cz + dz]
        for dx in (-h, h) for dy in (-h, h) for dz in (-h, h)
    ])
    cube_edges = [(0,1),(2,3),(4,5),(6,7),(0,2),(1,3),(4,6),(5,7),(0,4),(1,5),(2,6),(3,7)]
    ax.add_collection3d(
        Line3DCollection([[corners[a], corners[b]] for a, b in cube_edges],
                         colors="red", linewidths=1.5)
    )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    # --- Axis limits: focus on the path region, not the full random-sample extent ---
    # Use the solution path bounds if available, otherwise fall back to node bounds.
    focus_pts = pp if (path is not None and len(path) > 1) else node_pos
    margin = 0.25   # m padding around the path on each side
    ax.set_xlim(focus_pts[:, 0].min() - margin, focus_pts[:, 0].max() + margin)
    ax.set_ylim(focus_pts[:, 1].min() - margin, focus_pts[:, 1].max() + margin)
    ax.set_zlim(focus_pts[:, 2].min() - margin, focus_pts[:, 2].max() + margin)

    # Equal physical scaling: each metre of data occupies the same number of pixels
    x_span = (focus_pts[:, 0].max() - focus_pts[:, 0].min()) + 2 * margin
    y_span = (focus_pts[:, 1].max() - focus_pts[:, 1].min()) + 2 * margin
    z_span = (focus_pts[:, 2].max() - focus_pts[:, 2].min()) + 2 * margin
    ax.set_box_aspect([x_span, y_span, z_span])

    # Viewing angle: elev lifts the eye so Z is clearly the vertical axis;
    # azim rotates in the XY plane for the best diagonal view of the path.
    ax.view_init(elev=25, azim=45)

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ax.set_title(f"RRT Tree in Task Space  |  {len(tree)} nodes  |  {ts}")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Visualization saved to {out_path}")
    if show:
        plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# CLI and main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Joint-space RRT for Panda in robosuite Lift environment"
    )
    p.add_argument("--step_size",      type=float, default=0.1,
                   help="Max joint change per RRT step (rad)")
    p.add_argument("--goal_threshold", type=float, default=0.03,
                   help="EEF dist to goal for success (m)")
    p.add_argument("--goal_bias",      type=float, default=0.15,
                   help="Fraction of iters biased toward IK goal seed")
    p.add_argument("--max_iters",      type=int,   default=5000)
    p.add_argument("--seed",           type=int,   default=0)
    p.add_argument("--interp_step",    type=float, default=0.02,
                   help="Interpolation step for edge collision check (rad)")
    p.add_argument("--exec_tol",       type=float, default=0.02,
                   help="Joint error tolerance for waypoint completion (rad)")
    p.add_argument("--exec_max_steps", type=int,   default=150,
                   help="Max env steps per waypoint during execution")
    p.add_argument("--render",         action="store_true",
                   help="Enable MuJoCo viewer during execution")
    p.add_argument("--show",           action="store_true",
                   help="Show matplotlib tree plot interactively")
    p.add_argument("--no_exec",        action="store_true",
                   help="Skip path execution (plan only)")
    p.add_argument("--out_plan",       type=str,
                   default=str(_DIR_PLANS  / "plan.npz"),
                   help="Output path for planned qpos array")
    p.add_argument("--out_tree",       type=str,
                   default=str(_DIR_TREES  / "tree.png"),
                   help="Output path for tree visualization image")
    p.add_argument("--out_video",      type=str,
                   default=str(_DIR_VIDEOS / "execution.mp4"),
                   help="Output path for execution video")
    p.add_argument("--video_fps",      type=int,   default=20,
                   help="Frames per second for the output video")
    p.add_argument("--cam",            type=str,   default="frontview",
                   help="Camera name for offscreen rendering (frontview/agentview/sideview)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Timestamp used in all output filenames for this run
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Inject timestamp before the extension, preserving the directory prefix.
    # e.g. rrt/trees/tree.png → rrt/trees/tree_20240603_143022.png
    def _stamp(path: str) -> str:
        p = Path(path)
        return str(p.with_stem(f"{p.stem}_{run_ts}"))

    out_tree  = _stamp(args.out_tree)
    out_video = _stamp(args.out_video)
    out_plan  = _stamp(args.out_plan)

    # Override global camera name from CLI
    global _CAM
    _CAM = args.cam

    cfg = RRTConfig(
        step_size=args.step_size,
        goal_threshold=args.goal_threshold,
        goal_bias=args.goal_bias,
        max_iters=args.max_iters,
        seed=args.seed,
        interp_step=args.interp_step,
        exec_tol=args.exec_tol,
        exec_max_steps=args.exec_max_steps,
    )

    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    # --- Environment ---
    print("Creating robosuite Lift environment (Panda, JOINT_POSITION absolute)...")
    env = make_env(render=args.render)
    env.reset()

    arm_idx = get_arm_qpos_idx()
    lo, hi = get_joint_limits(env, arm_idx)
    eef_sid = get_eef_site_id(env)
    cube_bid = get_cube_body_id(env)

    print(f"Arm qpos indices: {arm_idx}")
    print(f"Joint limits lo:  {np.round(lo, 3)}")
    print(f"Joint limits hi:  {np.round(hi, 3)}")
    print(f"EEF site: {eef_sid} ({env.sim.model.site_id2name(eef_sid)})")
    print(f"Cube body: {cube_bid} ({env.sim.model.body_id2name(cube_bid)})")
    print(f"Action dim: {env.action_dim}")

    # --- Start and goal ---
    q_start = env.sim.data.qpos[arm_idx].copy()
    cube_pos = env.sim.data.body_xpos[cube_bid].copy()
    target_pos = cube_pos + np.array([0.0, 0.0, 0.08])

    print(f"\nq_start:    {np.round(q_start, 3)}")
    print(f"Cube pos:   {np.round(cube_pos, 3)}")
    print(f"Target pos: {np.round(target_pos, 3)}")

    # --- IK goal seed ---
    print("\nComputing IK goal seed...")
    q_goal_seed = ik_goal_seed(
        env=env,
        target_pos=target_pos,
        arm_idx=arm_idx,
        eef_sid=eef_sid,
        lo=lo,
        hi=hi,
        q_init=q_start.copy(),
    )

    # --- Plan ---
    t_plan = time.perf_counter()
    path, tree, iters_used = rrt(
        env=env,
        q_start=q_start,
        target_pos=target_pos,
        arm_idx=arm_idx,
        eef_sid=eef_sid,
        cfg=cfg,
        rng=rng,
        lo=lo,
        hi=hi,
        q_goal_seed=q_goal_seed,
    )
    plan_time = time.perf_counter() - t_plan

    # --- Visualize tree (saves timestamped PNG) ---
    visualize(
        tree=tree,
        path=path,
        target_pos=target_pos,
        cube_pos=cube_pos,
        arm_idx=arm_idx,
        eef_sid=eef_sid,
        env=env,
        show=args.show,
        out_path=out_tree,
    )

    # --- Save plan ---
    if path is not None:
        np.savez(out_plan, qpos=np.array(path))
        print(f"Plan saved to {out_plan} ({len(path)} waypoints)")

    # --- Execute + record video ---
    exec_success = False
    eef_final = None
    dist_final = None
    if path is not None and not args.no_exec:
        env.reset()
        frames: list[np.ndarray] = []

        # Drive arm to q_start via the controller (recorded into frames)
        print("Stabilising at q_start before execution...")
        action_init = np.zeros(env.action_dim)
        action_init[:7] = q_start
        t_stab = time.perf_counter()
        for _ in range(150):
            env.step(action_init)
            if args.render:
                env.render()
            eef = env.sim.data.site_xpos[eef_sid].copy()
            dist = float(np.linalg.norm(eef - target_pos))
            elapsed = time.perf_counter() - t_stab
            q_curr = env.sim.data.qpos[arm_idx].copy()
            err = float(np.linalg.norm(q_curr - q_start))
            frames.append(_capture_frame(env, elapsed, "Stabilising at q_start", err, dist))

        eef_final = execute_path(
            env=env,
            path=path,
            arm_idx=arm_idx,
            eef_sid=eef_sid,
            target_pos=target_pos,
            cfg=cfg,
            render=args.render,
            frames=frames,
        )
        dist_final = float(np.linalg.norm(eef_final - target_pos))
        exec_success = dist_final < cfg.goal_threshold * 2.0
        print(f"\nFinal EEF position: {np.round(eef_final, 3)}")
        print(f"Distance to goal:   {dist_final:.4f} m")
        print(f"Execution success:  {exec_success}")

        if HAS_CV2:
            save_video(frames, out_video, args.video_fps)
        else:
            print("cv2 not available — skipping video save.")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Run timestamp:     {run_ts}")
    print(f"  Iterations used:   {iters_used} / {cfg.max_iters}")
    print(f"  Tree size:         {len(tree)} nodes")
    print(f"  Path found:        {path is not None}")
    print(f"  Path length:       {len(path) if path else 'N/A'} waypoints")
    print(f"  Plan time:         {plan_time:.2f} s")
    print(f"  Tree image:        {out_tree}")
    if path is not None:
        print(f"  Plan file:         {out_plan}")
    if path is not None and not args.no_exec:
        print(f"  Exec success:      {exec_success}")
        if dist_final is not None:
            print(f"  Final EEF dist:    {dist_final:.4f} m")
        print(f"  Video:             {out_video}")
    print("=" * 60)

    env.close()


if __name__ == "__main__":
    main()
