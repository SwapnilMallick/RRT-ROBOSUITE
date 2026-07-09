"""
RRT in end-effector (Cartesian) space with an off-the-shelf encoder (DINOv2) as
the termination checker, for the Franka cube-REACH task in robosuite.

Per the spec (rrt_encoder.pdf):
  * Configuration space = end-effector (x, y, z), NOT the 7 joint angles.
  * Each iteration: sample q_rand uniformly in the table-workspace box, find the
    nearest tree node q_near (Euclidean xyz), and extend a DELTA STEP toward
    q_rand (not fully greedy).
  * The Cartesian target is solved with a self-contained MuJoCo damped-least-
    squares (DLS) Jacobian IK -> joint angles.
  * Reset-and-replay: to position the arm at q_near, reset the env and replay the
    stored JOINT configs along the path (the sim can't teleport). Nodes therefore
    store joint angles (required for replay), plus EEF pos, parent, action.
  * Collision: set qpos + mj_forward, check data.ncon; intermediate-config checks
    (anti-tunnel). Forbidden: table<->arm, arm-self. Gripper<->cube allowed.
  * Termination: render the new state -> DINOv2 embedding z_state -> cosine sim to
    z_goal; if sim > threshold, terminate and return the path. THRESHOLD IS A
    PLAIN TWEAKABLE PARAMETER (no calibration).
  * Goal frame: NOT the demo's last frame. Pick the first frame whose true
    gripper-cube distance crosses eps (the reach moment), embed that as z_goal.

Encoder is pluggable (DINOv2 default). Cube is frozen at the demo's cube pose so
reset-and-replay is coherent. A --synthetic mode verifies RRT + IK mechanics
without robosuite/MuJoCo/weights.
"""

import argparse
import numpy as np


# =========================================================================== #
# Encoders (DINOv2 default; I-JEPA available) -- single-image interface
# =========================================================================== #
class DINOv2Encoder:
    name = "DINOv2"
    backbone = "ViT-S/14 (DINOv2)"
    def __init__(self, device, repo="dinov2_vits14", pooling="cls"):
        import torch
        import torchvision.transforms as T
        self.torch = torch; self.device = device; self.pooling = pooling
        self.model = torch.hub.load("facebookresearch/dinov2", repo).to(device).eval()
        self.tf = T.Compose([
            T.ToPILImage(), T.Resize(224), T.CenterCrop(224), T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    def embed(self, img_uint8):
        import torch
        with torch.no_grad():
            x = self.tf(img_uint8).unsqueeze(0).to(self.device)
            if self.pooling == "cls":
                z = self.model(x)
            else:
                z = self.model.forward_features(x)["x_norm_patchtokens"].mean(1)
            z = torch.nn.functional.normalize(z, dim=-1)
        return z[0].cpu().numpy()


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


# =========================================================================== #
# Goal-frame selection (REACH task): first frame within eps of the cube.
# =========================================================================== #
def select_reach_index(eef_pos, cube_pos, eps):
    dist = np.linalg.norm(np.asarray(eef_pos) - np.asarray(cube_pos), axis=1)
    below = np.where(dist < eps)[0]
    if len(below) > 0:
        return int(below[0]), dist, f"reach frame = first within eps at t={int(below[0])}"
    idx = int(np.argmin(dist))
    return idx, dist, (f"WARNING: no frame within eps; using closest approach t={idx} "
                       f"(dist {dist[idx]:.4f})")


# =========================================================================== #
# Robosuite backend with DLS Jacobian IK (kinematic; set qpos + mj_forward)
# =========================================================================== #
class RobosuiteBackend:
    def __init__(self, env_name="Lift", robot="Panda", camera="agentview",
                 img_hw=224, n_collision_substeps=8, cube_pos=None, cube_quat_wxyz=None):
        import copy
        import robosuite as suite
        from robosuite import load_part_controller_config
        self.suite = suite
        base_cfg = suite.load_composite_controller_config(robot=robot)
        cfg = copy.deepcopy(base_cfg)
        jp = load_part_controller_config(default_controller="JOINT_POSITION")
        jp["input_type"] = "absolute"; jp["output_max"] = 5.0; jp["output_min"] = -5.0
        jp["kp"] = 50; jp["gripper"] = {"type": "GRIP"}
        cfg["body_parts"]["right"] = jp
        self.cube_target = None if cube_pos is None else np.asarray(cube_pos, float)
        self.cube_quat_target = (None if cube_quat_wxyz is None
                                  else np.asarray(cube_quat_wxyz, float))
        mk = dict(robots=robot, controller_configs=cfg, has_renderer=False,
                  has_offscreen_renderer=True, use_camera_obs=True,
                  camera_names=camera, camera_heights=img_hw, camera_widths=img_hw,
                  ignore_done=True)
        if self.cube_target is not None:
            mk["placement_initializer"] = self._fixed_initializer(
                self.cube_target, self.cube_quat_target)
        self.env = suite.make(env_name, **mk)
        self.camera = camera; self.img_hw = img_hw; self.n_sub = n_collision_substeps
        self.env.reset()
        self.sim = self.env.sim
        self.joint_names = [f"robot0_joint{i}" for i in range(1, 8)]
        self.qpos_idx = np.array([self.sim.model.get_joint_qpos_addr(n)
                                  for n in self.joint_names])
        jr = self.sim.model.jnt_range
        jids = [self.sim.model.joint_name2id(n) for n in self.joint_names]
        self.jnt_low = jr[jids, 0].copy(); self.jnt_high = jr[jids, 1].copy()
        self._dof_idx = np.array([self.sim.model.jnt_dofadr[j] for j in jids])
        self._build_geom_sets()
        grip = [n for n in self.sim.model.site_names if n.endswith("grip_site")]
        if not grip:
            raise RuntimeError(f"no grip site; sites={self.sim.model.site_names}")
        self.grip_site_name = grip[0]
        self.grip_site_id = self.sim.model.site_name2id(self.grip_site_name)
        self._cube_qpos_addr = self._cube_dof_addr = None
        if self.cube_target is not None:
            self._resolve_cube_qpos_addr()
            self._freeze_cube()
            got = self.cube_pos()
            print(f"[cube frozen] target xy={np.round(self.cube_target[:2],4)} | "
                  f"spawned={np.round(got,4)} | "
                  f"xy-err {np.linalg.norm(got[:2]-self.cube_target[:2]):.4f}")
            if self.cube_quat_target is not None:
                got_quat = self.sim.data.body_xquat[
                    self.sim.model.body_name2id("cube_main")].copy()
                print(f"[cube frozen] target quat(wxyz)={np.round(self.cube_quat_target,4)} | "
                      f"spawned quat(wxyz)={np.round(got_quat,4)}")

    def _resolve_cube_qpos_addr(self):
        """Locate the cube's free-joint qpos/qvel offsets once (stable across
        env.reset() calls since the MJCF doesn't change), so _freeze_cube can
        overwrite them directly instead of relying on the placement sampler +
        physics settling, which drifts by ~1mm reset-to-reset from contact-
        solver warm-start residue left over from the RRT's replayed states."""
        bid = self.sim.model.body_name2id("cube_main")
        jid = self.sim.model.body_jntadr[bid]
        self._cube_qpos_addr = self.sim.model.jnt_qposadr[jid]
        self._cube_dof_addr = self.sim.model.jnt_dofadr[jid]

    def _freeze_cube(self):
        """Force the cube's free-joint qpos to the exact demo pose (bypassing
        the placement sampler's post-spawn settling) and zero its velocity, so
        cube_pos()/cube orientation are bit-identical to the demo after every
        reset -- no z (or x/y/quat) jitter."""
        if self._cube_qpos_addr is None:
            return
        a = self._cube_qpos_addr
        self.sim.data.qpos[a:a + 3] = self.cube_target
        quat = (self.cube_quat_target if self.cube_quat_target is not None
                else np.array([1.0, 0.0, 0.0, 0.0]))
        self.sim.data.qpos[a + 3:a + 7] = quat
        d = self._cube_dof_addr
        self.sim.data.qvel[d:d + 6] = 0.0
        self.sim.forward()

    def _fixed_initializer(self, cube_pos, cube_quat_wxyz=None):
        """Deterministic placement at the demo's (x, y) with the demo's actual
        orientation. UniformRandomSampler._sample_quat only supports a rotation
        about a single fixed axis (rotation=<float>), which forces identity for
        any demo quat that isn't a pure z-rotation -- so we subclass it and
        override _sample_quat to return the exact demo quaternion (wxyz) every
        call, leaving the rest of the (retry / z-offset / init_quat-multiply)
        sampling logic untouched."""
        from robosuite.utils.placement_samplers import UniformRandomSampler
        cx, cy = float(cube_pos[0]), float(cube_pos[1])
        quat_fixed = (np.array([1.0, 0.0, 0.0, 0.0]) if cube_quat_wxyz is None
                      else np.asarray(cube_quat_wxyz, float).copy())

        class _FixedPoseSampler(UniformRandomSampler):
            def _sample_quat(self):
                return quat_fixed.copy()

        return _FixedPoseSampler(
            name="ObjectSampler", x_range=[cx, cx], y_range=[cy, cy],
            rotation=0.0, rotation_axis="z", ensure_object_boundary_in_range=False,
            ensure_valid_placement=False, reference_pos=(0.0, 0.0, 0.8), z_offset=0.01)

    def _build_geom_sets(self):
        m = self.sim.model
        self.arm_geoms, self.table_geoms, self.cube_geoms = set(), set(), set()
        for gid in range(m.ngeom):
            name = m.geom_id2name(gid) or ""
            if name.startswith("robot0") or "gripper" in name:
                self.arm_geoms.add(gid)
            elif "table" in name:
                self.table_geoms.add(gid)
            elif "cube" in name:
                self.cube_geoms.add(gid)

    def reset(self):
        self.env.reset()
        self.sim = self.env.sim
        self._freeze_cube()
        return self.get_q()

    def get_q(self):
        return self.sim.data.qpos[self.qpos_idx].copy()

    def set_q(self, q):
        self.sim.data.qpos[self.qpos_idx] = q
        self.sim.forward()

    def eef_pos(self):
        return self.sim.data.site_xpos[self.grip_site_id].copy()

    def cube_pos(self):
        return self.sim.data.get_body_xpos("cube_main").copy()

    def joint_limits_ok(self, q):
        return bool(np.all(q >= self.jnt_low) and np.all(q <= self.jnt_high))

    # ---- self-contained damped-least-squares Jacobian IK ------------------- #
    def solve_ik(self, target_xyz, q_init, iters=100, tol=1e-3, damping=1e-2,
                 max_step=0.1, q_bias=None, null_space_gain=0.0):
        """Damped-least-squares IK for the grip site. Returns (q, ok). Kinematic:
        sets qpos, mj_forward, reads site Jacobian, DLS update, clip to limits.
        q_bias/null_space_gain pull the redundant DOF toward a reference posture
        (e.g. the demo's reach-frame joint config) using the standard null-space
        projection, without disturbing the primary position task."""
        import mujoco
        q = np.array(q_init, float).copy()
        m, d = self.sim.model, self.sim.data
        n_arm = len(self._dof_idx)
        jacp = np.zeros((3, m.nv))
        for _ in range(iters):
            self.set_q(q)
            cur = self.sim.data.site_xpos[self.grip_site_id]
            err = target_xyz - cur
            if np.linalg.norm(err) < tol:
                return q, True
            # position Jacobian of the grip site (3 x nv), then select arm dofs
            try:
                mujoco.mj_jacSite(m._model if hasattr(m, "_model") else m,
                                  d._data if hasattr(d, "_data") else d,
                                  jacp, None, self.grip_site_id)
            except Exception:
                # fallback for mujoco-py style bindings
                self.sim.data.get_site_jacp(self.grip_site_name, jacp=jacp.reshape(-1))
            J = jacp[:, self._dof_idx]                    # 3 x 7
            # DLS: dq = J^T (J J^T + lambda^2 I)^-1 err
            JJt = J @ J.T + (damping ** 2) * np.eye(3)
            J_dls = J.T @ np.linalg.inv(JJt)               # 7 x 3 damped pseudo-inverse
            dq = J_dls @ err
            if q_bias is not None and null_space_gain > 0:
                null_proj = np.eye(n_arm) - J_dls @ J
                dq = dq + null_proj @ (null_space_gain * (np.asarray(q_bias, float) - q))
            n = np.linalg.norm(dq)
            if n > max_step:
                dq *= max_step / n
            q = np.clip(q + dq, self.jnt_low, self.jnt_high)
        self.set_q(q)
        final_err = np.linalg.norm(target_xyz - self.sim.data.site_xpos[self.grip_site_id])
        return q, bool(final_err < tol)

    def _forbidden_contact(self):
        d = self.sim.data
        for i in range(d.ncon):
            c = d.contact[i]; g1, g2 = c.geom1, c.geom2
            a1, a2 = g1 in self.arm_geoms, g2 in self.arm_geoms
            t1, t2 = g1 in self.table_geoms, g2 in self.table_geoms
            if (a1 and t2) or (a2 and t1) or (a1 and a2):
                return True
        return False

    def segment_collision_free(self, q_from, q_to):
        for s in np.linspace(0, 1, self.n_sub + 1)[1:]:
            q = q_from + s * (q_to - q_from)
            if not self.joint_limits_ok(q):
                return False
            self.set_q(q)
            if self._forbidden_contact():
                return False
        return True

    def render(self):
        img = self.env.sim.render(camera_name=self.camera,
                                  height=self.img_hw, width=self.img_hw)[::-1]
        return np.ascontiguousarray(img)


# =========================================================================== #
# Cartesian RRT with IK extension
# =========================================================================== #
class Node:
    __slots__ = ("id", "eef", "q", "parent", "action")
    def __init__(self, nid, eef, q, parent, action):
        self.id, self.eef, self.q, self.parent, self.action = nid, eef, q, parent, action


class IKRRT:
    def __init__(self, backend, encoder, goal_img, sim_threshold,
                 x_range, y_range, z_range, delta=0.05, m_iters=4000,
                 eps_dist=0.03, ik_tol=1e-3, rng_seed=0, log_every=200,
                 save_dir=None, save_every=200, video_path=None, video_fps=10,
                 goal_bias=0.15, q_bias=None, null_space_gain=0.0):
        self.b = backend; self.enc = encoder
        self.z_goal = encoder.embed(goal_img)
        self.thresh = sim_threshold
        self.xr, self.yr, self.zr = x_range, y_range, z_range
        self.delta = delta; self.m = m_iters; self.eps = eps_dist
        self.goal_bias = goal_bias
        self.q_bias = q_bias; self.null_space_gain = null_space_gain
        self.ik_tol = ik_tol
        self.rng = np.random.default_rng(rng_seed)
        self.log_every = log_every
        self.save_dir = save_dir; self.save_every = save_every
        self.video_path = video_path; self.video_fps = video_fps
        self._video_writer = None
        self.gc_steps = 0; self.max_sim = -1.0
        if save_dir:
            import os; os.makedirs(save_dir, exist_ok=True)
            self._save_img(goal_img, "goal_frame.png")
        if video_path:
            import os
            d = os.path.dirname(video_path)
            if d:
                os.makedirs(d, exist_ok=True)
        q0 = self.b.get_q()
        self.nodes = [Node(0, self.b.eef_pos(), q0, None, None)]
        self.reached = None; self.outcome = None
        self.collisions = self.ik_fail = self.resets = self.replay_steps = 0
        if self.video_path:
            self._write_video_frame(self.b.render())

    def _save_img(self, img, fname):
        try:
            import os
            from PIL import Image
            Image.fromarray(np.asarray(img, np.uint8)).save(
                os.path.join(self.save_dir, fname))
        except Exception as ex:
            print(f"(save failed {fname}: {ex})")

    def _sample(self):
        if self.goal_bias > 0 and self.rng.random() < self.goal_bias:
            return np.asarray(self.b.cube_pos(), float).copy()
        return np.array([self.rng.uniform(*self.xr),
                         self.rng.uniform(*self.yr),
                         self.rng.uniform(*self.zr)])

    def _nearest(self, q_rand):
        eefs = np.array([n.eef for n in self.nodes])
        return self.nodes[int(np.argmin(np.linalg.norm(eefs - q_rand, axis=1)))]

    def _predecessor_qs(self, node):
        chain = []
        while node.parent is not None:
            chain.append(node.q); node = node.parent
        return chain[::-1]

    def _reset_and_replay(self, node):
        self.resets += 1
        self.b.reset()
        for q in self._predecessor_qs(node):
            self.b.set_q(q); self.replay_steps += 1

    def _write_video_frame(self, img):
        if not self.video_path:
            return
        import cv2
        frame = cv2.cvtColor(np.asarray(img, np.uint8), cv2.COLOR_RGB2BGR)
        if self._video_writer is None:
            h, w = frame.shape[:2]
            # Prefer H.264 (avc1) — natively supported by QuickTime/browsers/VLC;
            # bare mp4v often fails to open on macOS. Fall back if unavailable.
            for fourcc_str in ("avc1", "mp4v"):
                fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
                writer = cv2.VideoWriter(
                    self.video_path, fourcc, self.video_fps, (w, h))
                if writer.isOpened():
                    self._video_writer = writer
                    break
                writer.release()
            else:
                print(f"Could not open a VideoWriter for {self.video_path} "
                      "— skipping exploration video.")
                self.video_path = None
                return
        self._video_writer.write(frame)

    def _goal_check(self, eef, img=None):
        if img is None:
            img = self.b.render()
        sim = cosine(self.enc.embed(img), self.z_goal)
        self.gc_steps += 1
        if sim > self.max_sim:
            self.max_sim = sim
        if self.save_dir and self.save_every and self.gc_steps % self.save_every == 0:
            td = float(np.linalg.norm(eef - self.b.cube_pos()))
            self._save_img(img, f"step_{self.gc_steps:07d}_sim{sim:.3f}_d{td:.3f}.png")
        if sim > self.thresh:
            td = float(np.linalg.norm(eef - self.b.cube_pos()))
            tp = td < self.eps
            self.outcome = (sim, td, tp)
            if self.save_dir:
                self._save_img(img, f"REACHED_{'TP' if tp else 'FP'}_"
                               f"step{self.gc_steps:07d}_sim{sim:.3f}_d{td:.3f}.png")
            return True
        return False

    def run(self):
        for t in range(self.m):
            q_rand = self._sample()
            near = self._nearest(q_rand)
            # delta step toward q_rand in Cartesian space
            direction = q_rand - near.eef
            dist = np.linalg.norm(direction)
            if dist < 1e-9:
                continue
            target = near.eef + (direction / dist) * min(self.delta, dist)
            # position the arm at q_near (reset + replay), then IK to target
            self._reset_and_replay(near)
            q_new, ok = self.b.solve_ik(target, near.q, tol=self.ik_tol,
                                        q_bias=self.q_bias,
                                        null_space_gain=self.null_space_gain)
            if not ok:
                self.ik_fail += 1
                if self._log(t): pass
                continue
            if not self.b.segment_collision_free(near.q, q_new):
                self.collisions += 1
                if self._log(t): pass
                continue
            self.b.set_q(q_new)
            eef = self.b.eef_pos()
            child = Node(len(self.nodes), eef, q_new, near, q_new - near.q)
            self.nodes.append(child)
            img = self.b.render()
            self._write_video_frame(img)
            if self._goal_check(eef, img=img):
                self.reached = child
                self._close_video()
                return self._result()
            self._log(t)
        self._close_video()
        return self._result()

    def _close_video(self):
        if self._video_writer is not None:
            self._video_writer.release()
            print(f"saved exploration video -> {self.video_path}")
            self._video_writer = None

    def _log(self, t):
        if self.log_every and (t + 1) % self.log_every == 0:
            print(f"iter {t+1}/{self.m} | nodes {len(self.nodes)} | "
                  f"ik_fail {self.ik_fail} | collisions {self.collisions} | "
                  f"max-sim {self.max_sim:.3f}/thr {self.thresh:.3f}")
            return True
        return False

    def _result(self):
        term = self.outcome is not None
        sim, td, tp = self.outcome if term else (None, None, None)
        return dict(
            terminated=term,
            verdict=(None if not term else ("TRUE_POSITIVE" if tp else "FALSE_POSITIVE")),
            crossing_sim=sim, crossing_true_dist=td,
            nodes=len(self.nodes), ik_fail=self.ik_fail, collisions=self.collisions,
            resets=self.resets, replay_steps=self.replay_steps,
            threshold=float(self.thresh), max_sim_seen=float(self.max_sim))

    def save_tree(self, path):
        n = len(self.nodes)
        eef = np.array([nd.eef for nd in self.nodes])
        q = np.array([nd.q for nd in self.nodes])
        parent_id = np.array([-1 if nd.parent is None else nd.parent.id
                              for nd in self.nodes], dtype=np.int64)
        depth = np.zeros(n, dtype=np.int64)
        for nd in self.nodes:
            if nd.parent is not None:
                depth[nd.id] = depth[nd.parent.id] + 1
        try:
            cube = np.asarray(self.b.cube_pos(), float)
        except Exception:
            cube = np.full(3, np.nan)
        np.savez_compressed(
            path, id=np.arange(n), parent_id=parent_id, q=q, eef=eef, depth=depth,
            cube=cube, start_eef=eef[0],
            reached_id=(-1 if self.reached is None else int(self.reached.id)),
            threshold=float(self.thresh), max_sim_seen=float(self.max_sim))
        print(f"saved tree ({n} nodes) -> {path}")


# =========================================================================== #
# Synthetic backend for verifying RRT + IK mechanics (no robosuite/MuJoCo)
# =========================================================================== #
class SyntheticBackend:
    """Toy 3-DOF-ish planar-ish arm: a simple analytic forward map from a 7-vector
    to an EEF xyz, and a numeric IK by gradient on that map -> exercises the same
    RRT code paths (sample, nearest, IK-extend, replay, collision, render)."""
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.jnt_low = -np.ones(7) * 3; self.jnt_high = np.ones(7) * 3
        self._cube = np.array([0.4, 0.0, 0.85]); self._q = np.zeros(7)
        self.img_hw = 16
    def _fk(self, q):
        # smooth nonlinear map R^7 -> R^3 (enough to make IK nontrivial)
        return np.array([0.1 + 0.15*np.sin(q[0]) + 0.1*q[1],
                         0.0 + 0.15*np.sin(q[2]) + 0.1*q[3],
                         1.0 - 0.1*np.cos(q[4]) - 0.05*q[5]])
    def reset(self): self._q = np.zeros(7); return self._q.copy()
    def get_q(self): return self._q.copy()
    def set_q(self, q): self._q = np.array(q, float).copy()
    def eef_pos(self): return self._fk(self._q)
    def cube_pos(self): return self._cube.copy()
    def joint_limits_ok(self, q):
        return bool(np.all(q >= self.jnt_low) and np.all(q <= self.jnt_high))
    def segment_collision_free(self, a, b): return self._fk(b)[2] > 0.80
    def solve_ik(self, target, q_init, iters=200, tol=1e-3, **kw):
        q = np.array(q_init, float).copy()
        for _ in range(iters):
            e = target - self._fk(q)
            if np.linalg.norm(e) < tol:
                return q, True
            J = np.zeros((3, 7)); h = 1e-4
            for i in range(7):
                dq = q.copy(); dq[i] += h
                J[:, i] = (self._fk(dq) - self._fk(q)) / h
            JJt = J @ J.T + 1e-4*np.eye(3)
            step = J.T @ np.linalg.solve(JJt, e)
            q = np.clip(q + 0.5*step, self.jnt_low, self.jnt_high)
        return q, bool(np.linalg.norm(target - self._fk(q)) < tol)
    def render(self):
        d = np.linalg.norm(self._fk(self._q) - self._cube)
        return np.full((16, 16, 3), int(np.clip(255*(1-d), 0, 255)), np.uint8)


class SyntheticEncoder:
    name = "synthetic"; backbone = "synthetic"
    def embed(self, img): v = img.astype(float).mean()/255.0; return np.array([v, 1-v])


def run_synthetic():
    print("=== SYNTHETIC verification (RRT + IK mechanics) ===")
    b = SyntheticBackend(); enc = SyntheticEncoder()
    rrt = IKRRT(b, enc, b.render(), sim_threshold=0.985,
                x_range=(0.0, 0.5), y_range=(-0.25, 0.25), z_range=(0.80, 1.05),
                delta=0.05, m_iters=600, eps_dist=0.05, log_every=200,
                goal_bias=0.15)
    res = rrt.run()
    print("result:", {k: (round(v, 3) if isinstance(v, float) else v)
                      for k, v in res.items()})
    assert res["nodes"] > 1
    print("OK: sample -> nearest -> IK delta-extend -> replay -> collision -> "
          "encoder termination all run.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--goal_demo_hdf5", default="./datasets/lift/ph/image224.hdf5")
    p.add_argument("--goal_demo", default="demo_0")
    p.add_argument("--image_key", default="agentview_image")
    p.add_argument("--camera", default="agentview")
    p.add_argument("--img_hw", type=int, default=224)
    p.add_argument("--encoder", default="dinov2", choices=["dinov2"])
    p.add_argument("--dino_pool", default="cls", choices=["cls", "mean"])
    p.add_argument("--sim_threshold", type=float, required=False, default=0.6,
                   help="cosine-sim termination threshold (plain tweakable parameter)")
    p.add_argument("--eps_dist", type=float, default=0.03)
    p.add_argument("--goal_eps", type=float, default=None,
                   help="reach-goal frame distance (defaults to --eps_dist)")
    p.add_argument("--delta", type=float, default=0.05, help="Cartesian step size (m)")
    p.add_argument("--goal_bias", type=float, default=0.15,
                   help="probability of sampling the (frozen) cube position directly "
                        "(set to 0 to disable)")
    p.add_argument("--null_space_gain", type=float, default=0.3,
                   help="pulls IK's redundant DOF toward the demo's reach-frame "
                        "joint config (0 disables the null-space bias)")
    p.add_argument("--x_range", type=float, nargs=2, default=[-0.2, 0.3])
    p.add_argument("--y_range", type=float, nargs=2, default=[-0.3, 0.3])
    p.add_argument("--z_range", type=float, nargs=2, default=[0.80, 1.05])
    p.add_argument("--m_iters", type=int, default=4000)
    p.add_argument("--ik_tol", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--save_dir", default=None)
    p.add_argument("--save_every", type=int, default=200)
    p.add_argument("--save_video", default=None,
                   help="mp4 path to record every accepted extension (exploration video)")
    p.add_argument("--video_fps", type=int, default=1)
    p.add_argument("--log_dir", default="logs",
                   help="directory for the timestamped run log (a .txt mirror of stdout)")
    a = p.parse_args()

    import atexit
    import datetime
    import os
    import sys

    os.makedirs(a.log_dir, exist_ok=True)
    log_path = os.path.join(
        a.log_dir, f"run_{datetime.datetime.now():%Y%m%d_%H%M%S}.txt")
    log_file = open(log_path, "w")

    def _restore_and_close():
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        log_file.close()

    atexit.register(_restore_and_close)

    class _Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, data):
            for s in self.streams:
                s.write(data)

        def flush(self):
            for s in self.streams:
                s.flush()

    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)
    print(f"logging to {log_path}")

    if a.synthetic:
        run_synthetic()
    else:
        import torch, h5py
        device = torch.device("mps" if torch.backends.mps.is_available()
                              else ("cuda" if torch.cuda.is_available() else "cpu"))
        print("device:", device)
        enc = DINOv2Encoder(device, pooling=a.dino_pool)
        with h5py.File(a.goal_demo_hdf5, "r") as f:
            root = f["data"] if "data" in f else f
            og = root[a.goal_demo]["obs"]
            frames = [np.asarray(og[a.image_key][i])
                      for i in range(og[a.image_key].shape[0])]
            eef = np.asarray(og["robot0_eef_pos"][:], float)
            obj_full = np.asarray(og["object"][:], float)
            obj = obj_full[:, :3]
            obj_quat_xyzw = obj_full[:, 3:7]
            joint_pos = np.asarray(og["robot0_joint_pos"][:], float)
        goal_eps = a.goal_eps if a.goal_eps is not None else a.eps_dist
        ridx, _, note = select_reach_index(eef, obj, goal_eps)
        print(note)
        goal_img = frames[ridx]
        cube_xyz = obj[ridx]
        from robosuite.utils.transform_utils import convert_quat
        cube_quat_wxyz = convert_quat(obj_quat_xyzw[ridx], to="wxyz")
        q_bias = joint_pos[ridx]
        print(f"goal frame t={ridx} | threshold {a.sim_threshold} (tweakable) | "
              f"null_space_gain {a.null_space_gain} | goal_bias {a.goal_bias}")
        backend = RobosuiteBackend(camera=a.camera, img_hw=a.img_hw, cube_pos=cube_xyz,
                                    cube_quat_wxyz=cube_quat_wxyz)
        rrt = IKRRT(backend, enc, goal_img, sim_threshold=a.sim_threshold,
                    x_range=tuple(a.x_range), y_range=tuple(a.y_range),
                    z_range=tuple(a.z_range), delta=a.delta, m_iters=a.m_iters,
                    eps_dist=a.eps_dist, ik_tol=a.ik_tol, rng_seed=a.seed,
                    save_dir=a.save_dir, save_every=a.save_every,
                    video_path=a.save_video, video_fps=a.video_fps,
                    goal_bias=a.goal_bias, q_bias=q_bias,
                    null_space_gain=a.null_space_gain)
        res = rrt.run()
        if a.save_dir:
            import os; rrt.save_tree(os.path.join(a.save_dir, "tree.npz"))
        print("\n=== RESULT ===")
        for k, v in res.items():
            print(f"  {k}: {v}")
        if res["terminated"]:
            print(f"\nENCODER TERMINATED: sim {res['crossing_sim']:.4f} > "
                  f"{res['threshold']:.4f} -> {res['verdict']} "
                  f"(true gripper-cube dist {res['crossing_true_dist']:.4f} m vs "
                  f"eps {a.eps_dist} m)")
        else:
            print(f"\nNo termination. max-sim {res['max_sim_seen']:.4f} vs "
                  f"threshold {res['threshold']:.4f}. ik_fail {res['ik_fail']} "
                  f"(if high, IK/workspace box needs tuning).")
