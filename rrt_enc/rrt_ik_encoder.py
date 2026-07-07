"""
RRT in end-effector (Cartesian) space with an off-the-shelf encoder (DINOv2) as
the termination checker, for the Franka cube-REACH task in robosuite.

Per the spec (rrt_encoder.pdf):
  * Configuration space = end-effector (x, y, z), NOT the 7 joint angles.
  * Each iteration: sample q_rand uniformly in the table-workspace box, find the
    nearest tree node q_near (Euclidean xyz), and extend a DELTA STEP toward
    q_rand (not fully greedy).
  * The delta-stepped Cartesian target is solved with a self-contained MuJoCo
    damped-least-squares (DLS) Jacobian IK -> joint angles.
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
                 img_hw=224, n_collision_substeps=8, cube_pos=None):
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
        mk = dict(robots=robot, controller_configs=cfg, has_renderer=False,
                  has_offscreen_renderer=True, use_camera_obs=True,
                  camera_names=camera, camera_heights=img_hw, camera_widths=img_hw,
                  ignore_done=True)
        if self.cube_target is not None:
            mk["placement_initializer"] = self._fixed_initializer(self.cube_target)
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
        if self.cube_target is not None:
            got = self.cube_pos()
            print(f"[cube frozen] target xy={np.round(self.cube_target[:2],4)} | "
                  f"spawned={np.round(got,4)} | "
                  f"xy-err {np.linalg.norm(got[:2]-self.cube_target[:2]):.4f}")

    def _fixed_initializer(self, cube_pos):
        from robosuite.utils.placement_samplers import UniformRandomSampler
        cx, cy = float(cube_pos[0]), float(cube_pos[1])
        return UniformRandomSampler(
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
        self.env.reset(); return self.get_q()

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
                 max_step=0.1):
        """Damped-least-squares IK for the grip site. Returns (q, ok). Kinematic:
        sets qpos, mj_forward, reads site Jacobian, DLS update, clip to limits."""
        import mujoco
        q = np.array(q_init, float).copy()
        m, d = self.sim.model, self.sim.data
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
            dq = J.T @ np.linalg.solve(JJt, err)
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
                 save_dir=None, save_every=200):
        self.b = backend; self.enc = encoder
        self.z_goal = encoder.embed(goal_img)
        self.thresh = sim_threshold
        self.xr, self.yr, self.zr = x_range, y_range, z_range
        self.delta = delta; self.m = m_iters; self.eps = eps_dist
        self.ik_tol = ik_tol
        self.rng = np.random.default_rng(rng_seed)
        self.log_every = log_every
        self.save_dir = save_dir; self.save_every = save_every
        self.gc_steps = 0; self.max_sim = -1.0
        if save_dir:
            import os; os.makedirs(save_dir, exist_ok=True)
            self._save_img(goal_img, "goal_frame.png")
        q0 = self.b.get_q()
        self.nodes = [Node(0, self.b.eef_pos(), q0, None, None)]
        self.reached = None; self.outcome = None
        self.collisions = self.ik_fail = self.resets = self.replay_steps = 0

    def _save_img(self, img, fname):
        try:
            import os
            from PIL import Image
            Image.fromarray(np.asarray(img, np.uint8)).save(
                os.path.join(self.save_dir, fname))
        except Exception as ex:
            print(f"(save failed {fname}: {ex})")

    def _sample(self):
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

    def _goal_check(self, eef):
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
            q_new, ok = self.b.solve_ik(target, near.q, tol=self.ik_tol)
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
            if self._goal_check(eef):
                self.reached = child
                return self._result()
            self._log(t)
        return self._result()

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
                delta=0.05, m_iters=600, eps_dist=0.05, log_every=200)
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
    p.add_argument("--x_range", type=float, nargs=2, default=[-0.2, 0.3])
    p.add_argument("--y_range", type=float, nargs=2, default=[-0.3, 0.3])
    p.add_argument("--z_range", type=float, nargs=2, default=[0.80, 1.05])
    p.add_argument("--m_iters", type=int, default=4000)
    p.add_argument("--ik_tol", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_dir", default=None)
    p.add_argument("--save_every", type=int, default=200)
    a = p.parse_args()

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
            obj = np.asarray(og["object"][:], float)[:, :3]
        goal_eps = a.goal_eps if a.goal_eps is not None else a.eps_dist
        ridx, _, note = select_reach_index(eef, obj, goal_eps)
        print(note)
        goal_img = frames[ridx]
        cube_xyz = obj[ridx]
        print(f"goal frame t={ridx} | threshold {a.sim_threshold} (tweakable)")
        backend = RobosuiteBackend(camera=a.camera, img_hw=a.img_hw, cube_pos=cube_xyz)
        rrt = IKRRT(backend, enc, goal_img, sim_threshold=a.sim_threshold,
                    x_range=tuple(a.x_range), y_range=tuple(a.y_range),
                    z_range=tuple(a.z_range), delta=a.delta, m_iters=a.m_iters,
                    eps_dist=a.eps_dist, ik_tol=a.ik_tol, rng_seed=a.seed,
                    save_dir=a.save_dir, save_every=a.save_every)
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
