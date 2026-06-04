"""
Joint-space RRT for Franka Panda in robosuite Lift environment.
Goal: reach 8 cm above the red cube using task-space goal checking.

Path smoothing is applied between planning and execution:
  - Method A (shortcut_bspline): randomized shortcutting + cubic B-spline fit
  - Method B (bspline_only): direct B-spline fit on the raw RRT path

Run with:
  python rrt_panda_smooth.py [--smoother {shortcut_bspline,bspline_only}]
                             [--execute {original,smoothed,both}]
                             [--n_shortcut 200] [--spline_s 0.0] [--n_samples 100]
                             [--plot_per_joint] [--render] [--show]
"""

from __future__ import annotations

import argparse
import copy
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

try:
    from scipy.interpolate import splprep, splev
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class RRTConfig:
    step_size: float = 0.1        # rad, max joint change per extension step
    goal_threshold: float = 0.01  # m, EEF distance to declare goal reached
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
# Path smoothing
# ---------------------------------------------------------------------------

def _densify(path: List[np.ndarray], n: int) -> List[np.ndarray]:
    """Linearly resample path to exactly n points."""
    arr = np.array(path)
    if len(arr) == 1:
        return [arr[0].copy() for _ in range(n)]
    indices = np.linspace(0, len(arr) - 1, n)
    result = []
    for idx in indices:
        lo_i = int(np.floor(idx))
        hi_i = min(lo_i + 1, len(arr) - 1)
        t = idx - lo_i
        result.append(arr[lo_i] * (1.0 - t) + arr[hi_i] * t)
    return result


def path_length(path: List[np.ndarray]) -> float:
    """Sum of Euclidean distances between consecutive waypoints."""
    if len(path) < 2:
        return 0.0
    arr = np.array(path)
    return float(np.sum(np.linalg.norm(np.diff(arr, axis=0), axis=1)))


def smooth_path(
    path: List[np.ndarray],
    method: str,
    env,
    arm_idx: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    interp_step: float,
    n_shortcut: int = 200,
    spline_s: float = 0.0,
    n_samples: int = 100,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray], Dict[str, Any]]:
    """
    Smooth an RRT path using randomized shortcutting and/or B-spline fitting.

    Parameters
    ----------
    path        : raw RRT path (list of joint configs or task-space positions)
    method      : 'shortcut_bspline' or 'bspline_only'
    env         : robosuite environment (used for collision checking)
    arm_idx     : indices of arm joints in qpos
    lo, hi      : joint limits (only used for joint-space paths)
    interp_step : step size for edge collision interpolation
    n_shortcut  : number of shortcut attempts
    spline_s    : B-spline smoothing factor (0 = interpolating, >0 = approximating)
    n_samples   : number of points to sample from the fitted spline
    rng         : NumPy random generator

    Returns
    -------
    smoothed_path   : list of n_samples configs along the smoothed trajectory
    shortcut_anchors: list of anchor configs after shortcutting (before spline)
    info            : dict with smoothing statistics
    """
    if not HAS_SCIPY:
        raise ImportError(
            "scipy is required for path smoothing. Install with: pip install scipy"
        )

    if rng is None:
        rng = np.random.default_rng(0)

    robot_bodies, obstacle_bodies = build_contact_sets(env)

    path_arr = np.array(path)
    dim = path_arr.shape[1]

    if dim == 7:
        space = "joint"
        print(f"[smooth_path] Smoothing in JOINT space ({dim} dimensions)")
    elif dim == 3:
        space = "task"
        print(f"[smooth_path] Smoothing in TASK space ({dim} dimensions)")
    else:
        space = f"generic-{dim}d"
        print(f"[smooth_path] Smoothing in generic {dim}-D space")

    info: Dict[str, Any] = {
        "space": space,
        "original_len": len(path),
        "shortcut_len": None,
        "smoothed_len": n_samples,
        "n_collisions_smoothed": 0,
        "fallback": False,
        "spline_fitted": False,
    }

    # ------------------------------------------------------------------
    # Step 1: Shortcutting
    # ------------------------------------------------------------------
    anchors: List[np.ndarray] = [p.copy() for p in path]

    if method == "shortcut_bspline":
        for _ in range(n_shortcut):
            if len(anchors) <= 2:
                break
            i = int(rng.integers(0, len(anchors) - 1))
            j = int(rng.integers(i + 1, len(anchors)))
            if i >= j:
                continue
            if edge_collision_free(
                env, anchors[i], anchors[j], arm_idx,
                robot_bodies, obstacle_bodies, interp_step,
            ):
                anchors = anchors[: i + 1] + anchors[j:]
        print(f"[smooth_path] Shortcutting: {len(path)} → {len(anchors)} anchors")
    else:
        print(f"[smooth_path] No shortcutting (bspline_only), {len(anchors)} anchors")

    info["shortcut_len"] = len(anchors)

    # ------------------------------------------------------------------
    # Step 2: B-spline fitting
    # ------------------------------------------------------------------
    anchors_arr = np.array(anchors)
    n_anchors = len(anchors_arr)

    if n_anchors < 2:
        print("[smooth_path] Not enough anchors for spline; returning original path")
        return list(path), anchors, info

    degree = min(3, n_anchors - 1)
    coords = [anchors_arr[:, i] for i in range(dim)]

    try:
        tck, _ = splprep(coords, s=spline_s, k=degree)
        u_new = np.linspace(0.0, 1.0, n_samples)
        evaluated = splev(u_new, tck)
        smoothed_arr = np.column_stack(evaluated)   # (n_samples, dim)
        info["spline_fitted"] = True
    except Exception as exc:
        print(f"[smooth_path] Spline fitting failed ({exc}); falling back to shortcut path")
        info["fallback"] = True
        return _densify(anchors, n_samples), anchors, info

    # ------------------------------------------------------------------
    # Step 3: Collision validation (joint space only)
    # ------------------------------------------------------------------
    if space == "joint":
        # Clamp any spline overshoot back into joint limits
        limit_violations = int(np.sum(
            np.any((smoothed_arr < lo) | (smoothed_arr > hi), axis=1)
        ))
        if limit_violations:
            print(f"[smooth_path] {limit_violations} samples violated joint limits; clamping")
            smoothed_arr = np.clip(smoothed_arr, lo, hi)

        # Dense re-sample for collision checking (200 points)
        u_val = np.linspace(0.0, 1.0, 200)
        val_pts = np.clip(np.column_stack(splev(u_val, tck)), lo, hi)

        n_collisions = sum(
            1 for q in val_pts
            if is_collision(env, q, arm_idx, robot_bodies, obstacle_bodies)
        )
        info["n_collisions_smoothed"] = n_collisions

        if n_collisions > 0:
            print(
                f"[smooth_path] {n_collisions}/200 smoothed samples in collision; "
                f"falling back to shortcut path"
            )
            info["fallback"] = True
            return _densify(anchors, n_samples), anchors, info

    smoothed_path = [smoothed_arr[i] for i in range(n_samples)]
    return smoothed_path, anchors, info


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
    pp = None
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

    focus_pts = pp if pp is not None else node_pos
    margin = 0.25
    ax.set_xlim(focus_pts[:, 0].min() - margin, focus_pts[:, 0].max() + margin)
    ax.set_ylim(focus_pts[:, 1].min() - margin, focus_pts[:, 1].max() + margin)
    ax.set_zlim(focus_pts[:, 2].min() - margin, focus_pts[:, 2].max() + margin)

    x_span = (focus_pts[:, 0].max() - focus_pts[:, 0].min()) + 2 * margin
    y_span = (focus_pts[:, 1].max() - focus_pts[:, 1].min()) + 2 * margin
    z_span = (focus_pts[:, 2].max() - focus_pts[:, 2].min()) + 2 * margin
    ax.set_box_aspect([x_span, y_span, z_span])

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


def visualize_comparison(
    original_path: List[np.ndarray],
    smoothed_path: List[np.ndarray],
    target_pos: np.ndarray,
    cube_pos: np.ndarray,
    arm_idx: np.ndarray,
    eef_sid: int,
    env,
    show: bool,
    out_path: str,
    task_name: str = "Panda Lift",
) -> None:
    """
    Side-by-side 3D task-space comparison: original RRT vs. smoothed path.
    Applies FK if paths are in joint space. Saves to out_path at 150 DPI.
    """
    if not HAS_MPL:
        print("matplotlib not available; skipping comparison visualization.")
        return

    def to_task(p: List[np.ndarray]) -> np.ndarray:
        if not p:
            return np.zeros((0, 3))
        arr = np.array(p)
        if arr.shape[1] == 7:
            return np.array([fk_eef(env, q, arm_idx, eef_sid) for q in arr])
        return arr

    orig_eef   = to_task(original_path)
    smooth_eef = to_task(smoothed_path)

    # Shared axis limits across both subplots
    all_pts = np.vstack([
        orig_eef, smooth_eef,
        target_pos.reshape(1, 3), cube_pos.reshape(1, 3),
    ])
    margin = 0.15
    xlim = (all_pts[:, 0].min() - margin, all_pts[:, 0].max() + margin)
    ylim = (all_pts[:, 1].min() - margin, all_pts[:, 1].max() + margin)
    zlim = (all_pts[:, 2].min() - margin, all_pts[:, 2].max() + margin)

    fig = plt.figure(figsize=(16, 7))
    fig.suptitle(
        f"{task_name}  |  Original: {len(original_path)} wp  |  "
        f"Smoothed: {len(smoothed_path)} wp",
        fontsize=13,
    )

    panels = [
        (orig_eef,   "Original RRT path", 1.0),
        (smooth_eef, "Smoothed path",      2.5),
    ]
    for col, (eef_pts, label, lw) in enumerate(panels):
        ax = fig.add_subplot(1, 2, col + 1, projection="3d")
        ax.set_title(label, fontsize=11)

        # Path markers and edges
        if len(eef_pts) > 0:
            ax.scatter(eef_pts[:, 0], eef_pts[:, 1], eef_pts[:, 2],
                       c="steelblue", s=8, alpha=0.7, zorder=3)
        if len(eef_pts) > 1:
            segs = [[eef_pts[k], eef_pts[k + 1]] for k in range(len(eef_pts) - 1)]
            ax.add_collection3d(
                Line3DCollection(segs, colors="steelblue", linewidths=lw, alpha=0.85)
            )

        # Start, goal, cube markers
        if len(eef_pts) > 0:
            ax.scatter(*eef_pts[0], c="black", s=120, marker="s", zorder=7, label="Start")
        ax.scatter(*target_pos, c="gold",  s=300, marker="*", zorder=8, label="Goal")
        ax.scatter(*cube_pos,   c="red",   s=80,  marker="o", zorder=6, label="Cube")

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_zlim(*zlim)
        ax.set_xlabel("X (m)", fontsize=8)
        ax.set_ylabel("Y (m)", fontsize=8)
        ax.set_zlabel("Z (m)", fontsize=8)
        ax.view_init(elev=25, azim=45)
        ax.legend(loc="upper left", fontsize=7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Path comparison saved to {out_path}")
    if show:
        plt.show()
    plt.close()


def visualize_joints(
    original_path: List[np.ndarray],
    smoothed_path: List[np.ndarray],
    show: bool,
    out_path: str,
) -> None:
    """
    7-panel plot of each joint angle vs. normalised path index for original
    and smoothed paths. Useful for diagnosing jerk reduction.
    """
    if not HAS_MPL:
        print("matplotlib not available; skipping per-joint plot.")
        return

    orig_arr   = np.array(original_path)
    smooth_arr = np.array(smoothed_path)

    if orig_arr.ndim < 2 or orig_arr.shape[1] != 7:
        print("[visualize_joints] Per-joint plot requires joint-space paths; skipping.")
        return

    fig, axes = plt.subplots(7, 1, figsize=(12, 14), sharex=False)
    fig.suptitle(
        "Per-joint angle vs. normalised path index  (Original vs. Smoothed)",
        fontsize=12,
    )

    orig_u   = np.linspace(0.0, 1.0, len(orig_arr))
    smooth_u = np.linspace(0.0, 1.0, len(smooth_arr))

    for j in range(7):
        ax = axes[j]
        ax.plot(orig_u,   orig_arr[:, j],   color="steelblue",  lw=1.4, alpha=0.85,
                label="Original" if j == 0 else "_")
        ax.plot(smooth_u, smooth_arr[:, j],  color="orangered",  lw=2.0, alpha=0.90,
                label="Smoothed" if j == 0 else "_")
        ax.set_ylabel(f"J{j + 1} (rad)", fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[0].legend(loc="upper right", fontsize=9)
    axes[-1].set_xlabel("Normalised path index", fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Per-joint plot saved to {out_path}")
    if show:
        plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# CLI and main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Joint-space RRT with path smoothing for Panda in robosuite Lift"
    )
    # --- RRT parameters ---
    p.add_argument("--step_size",      type=float, default=0.1,
                   help="Max joint change per RRT step (rad)")
    p.add_argument("--goal_threshold", type=float, default=0.01,
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
    # --- Smoothing parameters ---
    p.add_argument("--smoother",    type=str, default="shortcut_bspline",
                   choices=["shortcut_bspline", "bspline_only"],
                   help="Path smoothing method (default: shortcut_bspline)")
    p.add_argument("--n_shortcut",  type=int,   default=200,
                   help="Number of random shortcut attempts")
    p.add_argument("--spline_s",    type=float, default=0.0,
                   help="B-spline smoothing factor s (0=interpolating, >0=approximating)")
    p.add_argument("--n_samples",   type=int,   default=100,
                   help="Number of waypoints sampled from the fitted spline")
    # --- Execution mode ---
    p.add_argument("--execute",     type=str, default="smoothed",
                   choices=["original", "smoothed", "both"],
                   help="Which path to execute (default: smoothed)")
    # --- Visualization ---
    p.add_argument("--plot_per_joint", action="store_true",
                   help="Also produce per-joint angle comparison plot")
    p.add_argument("--render",         action="store_true",
                   help="Enable MuJoCo viewer during execution")
    p.add_argument("--show",           action="store_true",
                   help="Show matplotlib plots interactively")
    p.add_argument("--no_exec",        action="store_true",
                   help="Skip path execution (plan + smooth only)")
    # --- Output paths ---
    p.add_argument("--out_plan",    type=str,
                   default=str(_DIR_PLANS  / "plan.npz"),
                   help="Output path for original planned qpos array")
    p.add_argument("--out_tree",    type=str,
                   default=str(_DIR_TREES  / "tree.png"),
                   help="Output path for tree visualization image")
    p.add_argument("--out_video",   type=str,
                   default=str(_DIR_VIDEOS / "execution.mp4"),
                   help="Output path for execution video")
    p.add_argument("--video_fps",   type=int,   default=20,
                   help="Frames per second for the output video")
    p.add_argument("--cam",         type=str,   default="frontview",
                   help="Camera name for offscreen rendering")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    def _stamp(path: str) -> str:
        p = Path(path)
        return str(p.with_stem(f"{p.stem}_{run_ts}"))

    out_tree            = _stamp(args.out_tree)
    out_video           = _stamp(args.out_video)
    out_plan            = _stamp(args.out_plan)
    out_plan_smoothed   = _stamp(str(Path(args.out_plan).with_stem("plan_smoothed")))
    out_comparison      = _stamp(str(_DIR_TREES / "path_comparison.png"))
    out_joints          = _stamp(str(_DIR_TREES / "joints_comparison.png"))

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

    arm_idx  = get_arm_qpos_idx()
    lo, hi   = get_joint_limits(env, arm_idx)
    eef_sid  = get_eef_site_id(env)
    cube_bid = get_cube_body_id(env)

    print(f"Arm qpos indices: {arm_idx}")
    print(f"Joint limits lo:  {np.round(lo, 3)}")
    print(f"Joint limits hi:  {np.round(hi, 3)}")
    print(f"EEF site: {eef_sid} ({env.sim.model.site_id2name(eef_sid)})")
    print(f"Cube body: {cube_bid} ({env.sim.model.body_id2name(cube_bid)})")
    print(f"Action dim: {env.action_dim}")

    # --- Start and goal ---
    q_start    = env.sim.data.qpos[arm_idx].copy()
    cube_pos   = env.sim.data.body_xpos[cube_bid].copy()
    target_pos = cube_pos + np.array([0.0, 0.0, 0.08])

    print(f"\nq_start:    {np.round(q_start, 3)}")
    print(f"Cube pos:   {np.round(cube_pos, 3)}")
    print(f"Target pos: {np.round(target_pos, 3)}")

    # --- IK goal seed ---
    print("\nComputing IK goal seed...")
    q_goal_seed = ik_goal_seed(
        env=env, target_pos=target_pos, arm_idx=arm_idx,
        eef_sid=eef_sid, lo=lo, hi=hi, q_init=q_start.copy(),
    )

    # --- Plan ---
    t_plan = time.perf_counter()
    path, tree, iters_used = rrt(
        env=env, q_start=q_start, target_pos=target_pos,
        arm_idx=arm_idx, eef_sid=eef_sid, cfg=cfg,
        rng=rng, lo=lo, hi=hi, q_goal_seed=q_goal_seed,
    )
    plan_time = time.perf_counter() - t_plan

    # --- Visualize RRT tree ---
    visualize(
        tree=tree, path=path, target_pos=target_pos, cube_pos=cube_pos,
        arm_idx=arm_idx, eef_sid=eef_sid, env=env,
        show=args.show, out_path=out_tree,
    )

    # --- Save original plan ---
    if path is not None:
        np.savez(out_plan, qpos=np.array(path))
        print(f"Original plan saved to {out_plan} ({len(path)} waypoints)")

    # --- Smooth path ---
    smoothed_path: Optional[List[np.ndarray]] = None
    smooth_info: Dict[str, Any] = {}

    if path is not None:
        print(f"\n[smoothing] Method: {args.smoother}  n_shortcut={args.n_shortcut}  "
              f"spline_s={args.spline_s}  n_samples={args.n_samples}")
        t_smooth = time.perf_counter()
        smoothed_path, shortcut_anchors, smooth_info = smooth_path(
            path=path,
            method=args.smoother,
            env=env,
            arm_idx=arm_idx,
            lo=lo,
            hi=hi,
            interp_step=cfg.interp_step,
            n_shortcut=args.n_shortcut,
            spline_s=args.spline_s,
            n_samples=args.n_samples,
            rng=rng,
        )
        smooth_time = time.perf_counter() - t_smooth
        print(f"[smoothing] Done in {smooth_time:.2f}s  "
              f"{'(fallback: shortcut path used)' if smooth_info.get('fallback') else ''}")

        np.savez(out_plan_smoothed, qpos=np.array(smoothed_path))
        print(f"Smoothed plan saved to {out_plan_smoothed} ({len(smoothed_path)} waypoints)")

        # --- Path comparison visualization ---
        visualize_comparison(
            original_path=path,
            smoothed_path=smoothed_path,
            target_pos=target_pos,
            cube_pos=cube_pos,
            arm_idx=arm_idx,
            eef_sid=eef_sid,
            env=env,
            show=args.show,
            out_path=out_comparison,
            task_name="Panda Lift",
        )

        if args.plot_per_joint:
            visualize_joints(
                original_path=path,
                smoothed_path=smoothed_path,
                show=args.show,
                out_path=out_joints,
            )

    # --- Execute ---
    exec_orig_success    = False
    exec_smooth_success  = False
    dist_final_orig      = None
    dist_final_smooth    = None

    def _run_execution(exec_path: List[np.ndarray], label: str) -> Tuple[bool, float]:
        env.reset()
        frames: list[np.ndarray] = []
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
            frames.append(_capture_frame(env, elapsed, f"Stabilising [{label}]", err, dist))

        eef_final = execute_path(
            env=env, path=exec_path, arm_idx=arm_idx, eef_sid=eef_sid,
            target_pos=target_pos, cfg=cfg, render=args.render, frames=frames,
        )
        dist_f = float(np.linalg.norm(eef_final - target_pos))
        success = dist_f < cfg.goal_threshold * 2.0
        print(f"[{label}] Final EEF dist={dist_f:.4f}m  success={success}")

        if HAS_CV2:
            vid_path = out_video.replace(".mp4", f"_{label.lower()}.mp4")
            save_video(frames, vid_path, args.video_fps)
        return success, dist_f

    if path is not None and not args.no_exec:
        if args.execute in ("original", "both"):
            print("\n=== Executing ORIGINAL path ===")
            exec_orig_success, dist_final_orig = _run_execution(path, "original")

        if smoothed_path is not None and args.execute in ("smoothed", "both"):
            print("\n=== Executing SMOOTHED path ===")
            exec_smooth_success, dist_final_smooth = _run_execution(smoothed_path, "smoothed")

    # --- Collision count on original path (informational) ---
    original_collisions = 0
    if path is not None:
        robot_b, obs_b = build_contact_sets(env)
        for q in path:
            if is_collision(env, q, arm_idx, robot_b, obs_b):
                original_collisions += 1

    # --- Summary table ---
    orig_len   = path_length(path)   if path           else 0.0
    smooth_len = path_length(smoothed_path) if smoothed_path else 0.0

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Run timestamp          : {run_ts}")
    print(f"  RRT iterations         : {iters_used} / {cfg.max_iters}")
    print(f"  Tree size              : {len(tree)} nodes")
    print(f"  Plan time              : {plan_time:.2f} s")
    print(f"  Path found             : {path is not None}")
    if path is not None:
        print(f"  original_waypoints     : {len(path)}")
        print(f"  shortcut_waypoints     : {smooth_info.get('shortcut_len', 'N/A')}")
        print(f"  smoothed_samples       : {len(smoothed_path) if smoothed_path else 'N/A'}")
        print(f"  original_path_length   : {orig_len:.4f} rad")
        print(f"  smoothed_path_length   : {smooth_len:.4f} rad")
        print(f"  original_collisions    : {original_collisions}")
        print(f"  smoothed_collisions    : {smooth_info.get('n_collisions_smoothed', 'N/A')}")
        print(f"  smoother               : {args.smoother}")
        print(f"  spline_fitted          : {smooth_info.get('spline_fitted', 'N/A')}")
        print(f"  fallback_used          : {smooth_info.get('fallback', 'N/A')}")
    if not args.no_exec and path is not None:
        if args.execute in ("original", "both"):
            print(f"  exec_original_success  : {exec_orig_success}  "
                  f"(dist={dist_final_orig:.4f}m)" if dist_final_orig is not None else
                  f"  exec_original_success  : {exec_orig_success}")
        if args.execute in ("smoothed", "both"):
            print(f"  exec_smoothed_success  : {exec_smooth_success}  "
                  f"(dist={dist_final_smooth:.4f}m)" if dist_final_smooth is not None else
                  f"  exec_smoothed_success  : {exec_smooth_success}")
    print(f"  Tree image             : {out_tree}")
    if path is not None:
        print(f"  Original plan          : {out_plan}")
        print(f"  Smoothed plan          : {out_plan_smoothed}")
        print(f"  Path comparison        : {out_comparison}")
        if args.plot_per_joint:
            print(f"  Per-joint plot         : {out_joints}")
    print("=" * 70)

    env.close()


if __name__ == "__main__":
    main()
