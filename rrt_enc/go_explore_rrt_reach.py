"""
Go-Explore style reset-and-replay RRT for the Franka cube-REACH task in robosuite,
with an off-the-shelf encoder (DINOv2 by default; I-JEPA available) providing the goal signal.

Design (per the agreed decisions):
  * State = 7-D joint config q. Kinematic RRT: candidates are set via qpos +
    mj_forward (no physics stepping); "execute" = adopt the configuration.
  * Selection = EEF (x,y,z) VOXEL binning (NOT raw 7-D nodes, which blob): pick
    the least-occupied non-empty voxel, then the least-visited node in it.
  * Termination = cosine similarity(candidate_img, goal_img) > threshold  (>, not
    the PDF's inverted <). Threshold is CALIBRATED from demos, not guessed.
  * Encoder validation = log true gripper-cube distance at every termination and
    label it TP/FP (did the encoder's "reached" agree with distance < eps). The
    encoder NEVER decides reaching via distance; distance is only the audit.
  * Collisions = check INTERMEDIATE configs along parent->candidate (anti-
    tunnelling). Forbidden contacts: table<->arm, arm<->arm(self). Gripper<->cube
    contact is ALLOWED (target).
  * Candidate images rendered with the GOAL frame's exact camera config.

Encoder is a pluggable interface (swap I-JEPA for OpenCLIP/DINOv2 trivially).
Runs on MPS. A --synthetic mode verifies tree / selection / replay / calibration
/ TP-FP logging WITHOUT robosuite, MuJoCo, or encoder weights.
"""

import argparse
import numpy as np


# =========================================================================== #
# Encoder interface (real I-JEPA loader + a synthetic stand-in)
# =========================================================================== #
class IJEPAEncoder:
    """Frozen I-JEPA via HuggingFace transformers; mean-pooled patch tokens
    (I-JEPA has no CLS), L2-normalized."""
    name = "I-JEPA"
    def __init__(self, device, hf_repo="facebook/ijepa_vith14_1k"):
        import torch
        from transformers import AutoModel, AutoProcessor
        self.torch = torch
        self.device = device
        self.proc = AutoProcessor.from_pretrained(hf_repo)
        self.model = AutoModel.from_pretrained(hf_repo).to(device).eval()

    def embed(self, img_uint8):
        import torch
        from PIL import Image
        with torch.no_grad():
            inp = self.proc(images=[Image.fromarray(img_uint8)],
                            return_tensors="pt").to(self.device)
            z = self.model(**inp).last_hidden_state.mean(1)      # (1, D) mean-pool
            z = torch.nn.functional.normalize(z, dim=-1)
        return z[0].cpu().numpy()


class DINOv2Encoder:
    """Frozen DINOv2 (ViT-S/14) via torch.hub; CLS token (default) or mean-pooled
    patch tokens, L2-normalized. Single-image interface to match the RRT loop."""
    name = "DINOv2"
    def __init__(self, device, repo="dinov2_vits14", pooling="cls"):
        import torch
        import torchvision.transforms as T
        self.torch = torch
        self.device = device
        self.pooling = pooling                               # "cls" or "mean"
        self.model = torch.hub.load("facebookresearch/dinov2", repo).to(device).eval()
        self.tf = T.Compose([
            T.ToPILImage(), T.Resize(224), T.CenterCrop(224), T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def embed(self, img_uint8):
        import torch
        with torch.no_grad():
            x = self.tf(img_uint8).unsqueeze(0).to(self.device)   # (1,3,224,224)
            if self.pooling == "cls":
                z = self.model(x)                                 # (1, D) CLS
            else:
                z = self.model.forward_features(x)["x_norm_patchtokens"].mean(1)
            z = torch.nn.functional.normalize(z, dim=-1)
        return z[0].cpu().numpy()


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def obs_quat_to_mujoco(q_obs):
    """robosuite obs quaternions are scalar-LAST [qx,qy,qz,qw]; MuJoCo qpos wants
    scalar-FIRST [qw,qx,qy,qz]."""
    q = np.asarray(q_obs, float)
    return np.array([q[3], q[0], q[1], q[2]])


# =========================================================================== #
# Robosuite/MuJoCo backend interface. The RRT only talks to THIS class, so the
# synthetic backend can stand in for verification.
# =========================================================================== #
class RobosuiteBackend:
    """Wraps the env so the planner is sim-agnostic. Implements kinematic moves
    (set qpos + mj_forward), collision checks, EEF position, and rendering."""
    def __init__(self, env_name="Lift", robot="Panda", camera="agentview",
                 img_hw=224, n_collision_substeps=8, cube_pos=None, cube_quat=None):
        import copy
        import robosuite as suite
        from robosuite import load_part_controller_config
        self.suite = suite
        base_cfg = suite.load_composite_controller_config(robot=robot)
        cfg = copy.deepcopy(base_cfg)
        jp_cfg = load_part_controller_config(default_controller="JOINT_POSITION")
        jp_cfg["input_type"] = "absolute"
        jp_cfg["output_max"] = 5.0
        jp_cfg["output_min"] = -5.0
        jp_cfg["kp"] = 50
        jp_cfg["gripper"] = {"type": "GRIP"}
        cfg["body_parts"]["right"] = jp_cfg
        # Freeze the cube: build a deterministic placement initializer pinned to a
        # fixed (x, y, rotation) so EVERY reset (incl. reset-and-replay) spawns the
        # cube at the SAME pose. Without this, robosuite's default
        # UniformRandomSampler re-randomizes the cube each reset, making "reach
        # the cube" a moving target and corrupting both the goal-similarity signal
        # and the audit. cube_quat (MuJoCo wxyz, z-axis rotation only -- matches
        # how Lift's cube is actually dropped) is optional; omitting it falls back
        # to zero rotation, which will NOT match a demo whose cube was rotated.
        self.cube_target = None if cube_pos is None else np.asarray(cube_pos, float)
        self.cube_target_quat = None if cube_quat is None else np.asarray(cube_quat, float)
        make_kwargs = dict(
            robots=robot, controller_configs=cfg,
            has_renderer=False, has_offscreen_renderer=True,
            use_camera_obs=True, camera_names=camera,
            camera_heights=img_hw, camera_widths=img_hw, ignore_done=True)
        if self.cube_target is not None:
            rotation = 0.0
            if self.cube_target_quat is not None:
                qw, qz = self.cube_target_quat[0], self.cube_target_quat[3]
                rotation = float(2 * np.arctan2(qz, qw))
            make_kwargs["placement_initializer"] = self._fixed_initializer(
                self.cube_target, rotation)
        self.env = suite.make(env_name, **make_kwargs)
        self.camera = camera
        self.img_hw = img_hw
        self.n_sub = n_collision_substeps
        self.env.reset()
        self.sim = self.env.sim
        # 7 arm joint indices in qpos
        self.joint_names = [f"robot0_joint{i}" for i in range(1, 8)]
        self.qpos_idx = np.array([
            self.sim.model.get_joint_qpos_addr(n) for n in self.joint_names])
        jr = self.sim.model.jnt_range
        jids = [self.sim.model.joint_name2id(n) for n in self.joint_names]
        self.jnt_low = jr[jids, 0].copy()
        self.jnt_high = jr[jids, 1].copy()
        # geom groupings for contact filtering
        self._build_geom_sets()
        # grip site name varies by robosuite version (e.g. "gripper0_grip_site"
        # vs. the newer arm-suffixed "gripper0_right_grip_site")
        grip_candidates = [n for n in self.sim.model.site_names
                            if n.endswith("grip_site")]
        if not grip_candidates:
            raise RuntimeError(
                f"No grip site found; available sites = {self.sim.model.site_names}")
        self.grip_site_name = grip_candidates[0]
        if self.cube_target is not None:
            got = self.cube_pos()
            rot_note = ""
            if self.cube_target_quat is not None:
                qw, qz = self.cube_target_quat[0], self.cube_target_quat[3]
                rot_note = f" | target z-rot {np.degrees(2*np.arctan2(qz,qw)):.1f}deg"
            print(f"[cube frozen] target xy={np.round(self.cube_target[:2],4)} | "
                  f"spawned cube={np.round(got,4)} | "
                  f"xy-err {np.linalg.norm(got[:2]-self.cube_target[:2]):.4f}"
                  f"{rot_note}")

    def _fixed_initializer(self, cube_pos, rotation=0.0):
        """A zero-width UniformRandomSampler that always places the object at the
        given (x, y, rotation) -> deterministic cube every reset. Z is left to
        settle physically (matches how the demo's cube came to rest), so only
        x/y/rotation are pinned here."""
        from robosuite.utils.placement_samplers import UniformRandomSampler
        cx, cy = float(cube_pos[0]), float(cube_pos[1])
        # robosuite samples x/y as offsets from the table center; we pin the range
        # to a single point. reference_pos defaults to table top; using x/y_range
        # of zero width makes the draw deterministic. A scalar (non-iterable)
        # `rotation` is likewise used as-is rather than sampled from a range.
        return UniformRandomSampler(
            name="ObjectSampler",
            x_range=[cx, cx],
            y_range=[cy, cy],
            rotation=rotation,
            rotation_axis="z",
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=False,
            reference_pos=(0.0, 0.0, 0.8),
            z_offset=0.01,
        )

    def _build_geom_sets(self):
        m = self.sim.model
        self.arm_geoms, self.table_geoms, self.cube_geoms = set(), set(), set()
        for gid in range(m.ngeom):
            name = m.geom_id2name(gid) or ""
            if name.startswith("robot0") or "gripper" in name:
                self.arm_geoms.add(gid)
            elif "table" in name:
                self.table_geoms.add(gid)
            elif "cube" in name or name.startswith("cube"):
                self.cube_geoms.add(gid)

    def reset(self):
        self.env.reset()
        return self.get_q()

    def get_q(self):
        return self.sim.data.qpos[self.qpos_idx].copy()

    def set_q(self, q):
        self.sim.data.qpos[self.qpos_idx] = q
        self.sim.forward()                 # mj_forward: kinematics + contacts

    def eef_pos(self):
        return self.sim.data.get_site_xpos(self.grip_site_name).copy()

    def cube_pos(self):
        return self.sim.data.get_body_xpos("cube_main").copy()

    def joint_limits_ok(self, q):
        return bool(np.all(q >= self.jnt_low) and np.all(q <= self.jnt_high))

    def _forbidden_contact(self):
        """True if any CURRENT contact is forbidden (table<->arm or arm self).
        Gripper<->cube is allowed."""
        d = self.sim.data
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = c.geom1, c.geom2
            a1, a2 = g1 in self.arm_geoms, g2 in self.arm_geoms
            t1, t2 = g1 in self.table_geoms, g2 in self.table_geoms
            if (a1 and t2) or (a2 and t1):      # arm <-> table
                return True
            if a1 and a2:                        # arm self-collision
                return True
            # arm <-> cube intentionally allowed
        return False

    def segment_collision_free(self, q_from, q_to):
        """Check intermediate configs along the joint-space segment (anti-tunnel)."""
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
        return np.ascontiguousarray(img)     # uint8 HWC, un-flipped


# =========================================================================== #
# Go-Explore reset-and-replay RRT
# =========================================================================== #
class Node:
    __slots__ = ("id", "q", "parent", "action", "visits", "eef")
    def __init__(self, nid, q, parent, action, eef):
        self.id, self.q, self.parent, self.action = nid, q, parent, action
        self.visits, self.eef = 1, eef


class GoExploreRRT:
    def __init__(self, backend, encoder, goal_img, sim_threshold,
                 step_scale=0.15, voxel=0.05, k_seed=30, j_roll=20, m_iters=4000,
                 eps_dist=0.03, rng_seed=0, log_every=200, save_dir=None,
                 save_every=200):
        self.b = backend
        self.enc = encoder
        self.z_goal = encoder.embed(goal_img)
        self.thresh = sim_threshold
        self.step = step_scale            # max per-joint random step (radians)
        self.voxel = voxel                # EEF voxel size (meters)
        self.k, self.j, self.m = k_seed, j_roll, m_iters
        self.eps = eps_dist
        self.rng = np.random.default_rng(rng_seed)
        self.log_every = log_every
        self.save_dir = save_dir          # if set, dump frames here
        self.save_every = save_every      # save a candidate frame every N goal-checks
        self.gc_steps = 0                 # global goal-check counter
        self.max_sim = -1.0               # best similarity seen (diagnostic)
        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)
            self._save_img(goal_img, "goal_frame.png")

        q0 = self.b.get_q()
        self.nodes = [Node(0, q0, None, None, self.b.eef_pos())]
        self.bins = {}                    # voxel -> list[node_id]
        self._bin_add(self.nodes[0])
        self.reached = None
        self.outcome = None               # single crossing: (sim, true_dist, tp)
        self.collisions = self.resets = self.replay_steps = 0

    def _save_img(self, img, fname):
        try:
            import os
            from PIL import Image
            Image.fromarray(np.asarray(img, dtype=np.uint8)).save(
                os.path.join(self.save_dir, fname))
        except Exception as ex:
            print(f"(image save failed for {fname}: {ex})")

    # --- EEF voxel binning for selection ---
    def _voxel_of(self, eef):
        return tuple(np.floor(eef / self.voxel).astype(int))

    def _bin_add(self, node):
        v = self._voxel_of(node.eef)
        self.bins.setdefault(v, []).append(node.id)

    def _select(self):
        # least-OCCUPIED non-empty voxel, then least-visited node in it
        vmin = min(len(ids) for ids in self.bins.values())
        cand_v = [v for v, ids in self.bins.items() if len(ids) == vmin]
        v = cand_v[self.rng.integers(len(cand_v))]
        ids = self.bins[v]
        nvmin = min(self.nodes[i].visits for i in ids)
        pool = [i for i in ids if self.nodes[i].visits == nvmin]
        return self.nodes[pool[self.rng.integers(len(pool))]]

    def _random_q(self, q):
        dq = self.rng.uniform(-self.step, self.step, size=7)
        return np.clip(q + dq, self.b.jnt_low, self.b.jnt_high)

    def _add(self, q, parent, action, eef):
        n = Node(len(self.nodes), q, parent, action, eef)
        self.nodes.append(n)
        self._bin_add(n)
        return n

    def _predecessor_actions(self, node):
        chain = []
        while node.parent is not None:
            chain.append(node.action); node = node.parent
        return chain[::-1]

    def _reset_and_replay(self, node):
        self.resets += 1
        self.b.reset()
        q = self.b.get_q()
        for a in self._predecessor_actions(node):
            q = np.clip(q + a, self.b.jnt_low, self.b.jnt_high)
            self.b.set_q(q)
            self.replay_steps += 1
        return q

    def _goal_check(self, eef):
        """Termination by ENCODER ONLY; true distance logged for audit.
        Also: track max similarity, save a frame every save_every goal-checks,
        and always save the frame when the threshold is crossed."""
        img = self.b.render()
        sim = cosine(self.enc.embed(img), self.z_goal)
        self.gc_steps += 1
        if sim > self.max_sim:
            self.max_sim = sim
        # periodic snapshot (reuses the render already taken -> no extra cost)
        if self.save_dir and self.save_every and self.gc_steps % self.save_every == 0:
            true_d = float(np.linalg.norm(eef - self.b.cube_pos()))
            self._save_img(img, f"step_{self.gc_steps:07d}_sim{sim:.3f}_d{true_d:.3f}.png")
        if sim > self.thresh:
            true_d = float(np.linalg.norm(eef - self.b.cube_pos()))
            tp = true_d < self.eps
            self.outcome = (sim, true_d, tp)      # the one terminating crossing
            if self.save_dir:
                tag = "TP" if tp else "FP"
                self._save_img(
                    img,
                    f"REACHED_{tag}_step{self.gc_steps:07d}_sim{sim:.3f}_d{true_d:.3f}.png")
            return True
        return False

    def _walk(self, start_node, start_q, steps):
        cur, q = start_node, start_q.copy()
        for _ in range(steps):
            q_new = self._random_q(q)
            if not self.b.segment_collision_free(q, q_new):
                self.collisions += 1
                break
            self.b.set_q(q_new)
            eef = self.b.eef_pos()
            child = self._add(q_new, cur, q_new - q, eef)
            cur, q = child, q_new
            if self._goal_check(eef):
                self.reached = child
                return cur
        return cur

    def run(self):
        # phase 1: seed walk from start
        self.b.reset()
        cur = self._walk(self.nodes[0], self.nodes[0].q, self.k)
        if self.reached is not None:
            return self._result()
        # phase 2: least-visited-voxel selection + reset-replay + rollout
        for t in range(self.m):
            v = self._select()
            v.visits += 1
            q = self._reset_and_replay(v)
            self._walk(v, q, self.j)
            if self.reached is not None:
                return self._result()
            if self.log_every and (t + 1) % self.log_every == 0:
                print(f"iter {t+1}/{self.m} | nodes {len(self.nodes)} | "
                      f"voxels {len(self.bins)} | collisions {self.collisions} | "
                      f"max-sim {self.max_sim:.3f}/thr {self.thresh:.3f}")
        return self._result()

    def _result(self):
        terminated = self.outcome is not None
        sim, true_d, tp = self.outcome if terminated else (None, None, None)
        return dict(
            terminated=terminated,                 # did the encoder cross threshold?
            verdict=(None if not terminated else ("TRUE_POSITIVE" if tp
                                                  else "FALSE_POSITIVE")),
            crossing_sim=sim,
            crossing_true_dist=true_d,
            nodes=len(self.nodes), voxels=len(self.bins),
            collisions=self.collisions, resets=self.resets,
            replay_steps=self.replay_steps,
            threshold=float(self.thresh),
            max_sim_seen=float(self.max_sim),
        )

    def save_tree(self, path):
        """Dump the full tree to a .npz: per-node id, parent_id (-1 for root),
        joint config q (7), EEF position (3), visit count, and depth. Also stores
        the cube position and reached-node id as top-level arrays for the
        visualizer."""
        n = len(self.nodes)
        ids = np.arange(n, dtype=np.int64)
        parent_id = np.full(n, -1, dtype=np.int64)
        q = np.zeros((n, 7), dtype=np.float64)
        eef = np.zeros((n, 3), dtype=np.float64)
        visits = np.zeros(n, dtype=np.int64)
        depth = np.zeros(n, dtype=np.int64)
        for nd in self.nodes:
            q[nd.id] = nd.q
            eef[nd.id] = nd.eef
            visits[nd.id] = nd.visits
            if nd.parent is not None:
                parent_id[nd.id] = nd.parent.id
                depth[nd.id] = depth[nd.parent.id] + 1     # parents precede children
        try:
            cube = np.asarray(self.b.cube_pos(), float)
        except Exception:
            cube = np.full(3, np.nan)
        reached_id = -1 if self.reached is None else int(self.reached.id)
        np.savez_compressed(
            path, id=ids, parent_id=parent_id, q=q, eef=eef, visits=visits,
            depth=depth, cube=cube, start_eef=eef[0], reached_id=reached_id,
            threshold=float(self.thresh), max_sim_seen=float(self.max_sim))
        print(f"saved tree ({n} nodes) -> {path}")


# =========================================================================== #
# Reach-goal frame selection (REACH task, not lift): the goal is the first frame
# where the gripper comes within eps of the cube (approach), NOT the demo's last
# frame (which shows the cube grasped-and-lifted). We also truncate the demo at
# that moment so the grasp/lift tail doesn't contaminate calibration.
# =========================================================================== #
def select_reach_index(eef_pos, cube_pos, eps):
    """First frame whose true gripper-cube distance < eps (scanning from start ->
    the approach moment, before the grasp closes). Falls back to closest approach
    (argmin) with a warning if no frame crosses eps."""
    dist = np.linalg.norm(np.asarray(eef_pos) - np.asarray(cube_pos), axis=1)
    below = np.where(dist < eps)[0]
    if len(below) > 0:
        idx = int(below[0])
        note = f"reach frame = first within eps={eps} at t={idx} (dist {dist[idx]:.4f})"
    else:
        idx = int(np.argmin(dist))
        note = (f"WARNING: no frame within eps={eps}; fell back to closest approach "
                f"at t={idx} (dist {dist[idx]:.4f}) -- consider loosening --goal_eps")
    return idx, dist, note


# =========================================================================== #
# Threshold calibration from demos (so the threshold is physical, not guessed)
# =========================================================================== #
def calibrate_threshold(encoder, demo_frames, eef_pos, cube_pos, eps):
    """Return (threshold, margin, report). threshold = similarity that best
    separates 'within eps of cube' from 'farther'. Expects demo already TRUNCATED
    at the reach moment, so the goal = last (reach) frame."""
    z_goal = encoder.embed(demo_frames[-1])
    sims = np.array([cosine(encoder.embed(f), z_goal) for f in demo_frames])
    dist = np.linalg.norm(eef_pos - cube_pos, axis=1)
    near = dist < eps
    if near.sum() == 0 or (~near).sum() == 0:
        return float(sims.max()), 0.0, "insufficient near/far split"
    # threshold midway between max far-sim and min near-sim (separation margin)
    far_hi = sims[~near].max()
    near_lo = sims[near].min()
    thr = 0.5 * (far_hi + near_lo)
    margin = near_lo - far_hi          # >0 => cleanly separable; <=0 => overlap
    report = (f"near-min-sim {near_lo:.4f} | far-max-sim {far_hi:.4f} | "
              f"margin {margin:+.4f} "
              f"({'separable' if margin > 0 else 'OVERLAP - threshold unreliable'})")
    return float(thr), float(margin), report


# =========================================================================== #
# Synthetic verification backend (no robosuite / MuJoCo / weights)
# =========================================================================== #
class SyntheticBackend:
    """A toy 2-link-ish kinematics so the RRT machinery can be exercised: q in
    R^7 maps to a fake EEF position; cube at a fixed point; collisions when a
    'height' coord dips below the table."""
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.jnt_low = -np.ones(7); self.jnt_high = np.ones(7)
        self._cube = np.array([0.4, 0.0, 0.85])
        self._q = np.zeros(7)
        self.img_hw = 16
    def reset(self):
        self._q = np.zeros(7); return self._q.copy()
    def get_q(self): return self._q.copy()
    def set_q(self, q): self._q = q.copy()
    def eef_pos(self):
        # smooth map: more "reach" in +q drives EEF toward the cube
        base = np.array([0.1, 0.0, 1.1])
        reach = np.array([0.05, 0.03, -0.04]) * self._q.sum()
        return base + reach
    def cube_pos(self): return self._cube.copy()
    def joint_limits_ok(self, q):
        return bool(np.all(q >= self.jnt_low) and np.all(q <= self.jnt_high))
    def segment_collision_free(self, a, b):
        return self.eef_pos()[2] > 0.80     # crude table floor on current q
    def render(self):
        # fake image whose content depends on EEF-cube distance, so a synthetic
        # encoder can produce a distance-correlated similarity
        d = np.linalg.norm(self.eef_pos() - self._cube)
        img = np.full((16, 16, 3), int(np.clip(255 * (1 - d), 0, 255)), np.uint8)
        return img


class SyntheticEncoder:
    name = "synthetic"
    def __init__(self, goal_d=0.0): self.goal_d = goal_d
    def embed(self, img_uint8):
        v = img_uint8.mean() / 255.0       # ~ (1 - dist)
        return np.array([v, 1 - v], dtype=float)


def run_synthetic():
    print("=== SYNTHETIC verification (no robosuite/MuJoCo/weights) ===")
    b = SyntheticBackend()
    enc = SyntheticEncoder()
    # calibrate on a fake straight-in demo
    demo_q = np.linspace(np.zeros(7), np.ones(7), 30)
    frames, eefs = [], []
    for q in demo_q:
        b.set_q(q); frames.append(b.render()); eefs.append(b.eef_pos())
    eefs = np.array(eefs); cubes = np.repeat(b.cube_pos()[None], len(eefs), 0)
    thr, margin, rep = calibrate_threshold(enc, frames, eefs, cubes, eps=0.05)
    print(f"calibration: threshold {thr:.4f} | {rep}")
    b.reset()
    goal_img = frames[-1]
    rrt = GoExploreRRT(b, enc, goal_img, sim_threshold=thr, step_scale=0.3,
                       voxel=0.05, k_seed=20, j_roll=15, m_iters=500, eps_dist=0.05,
                       log_every=100)
    res = rrt.run()
    print("result:", {k: (round(v, 3) if isinstance(v, float) else v)
                       for k, v in res.items()})
    assert res["nodes"] > 1 and res["voxels"] >= 1
    print("OK: tree growth, EEF-voxel selection, reset-replay, calibration, "
          "and TP/FP logging all run.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic", action="store_true")
    # real-run args
    p.add_argument("--goal_demo_hdf5", default="/path/to/image224.hdf5")
    p.add_argument("--goal_demo", default="demo_0")
    p.add_argument("--image_key", default="agentview_image")
    p.add_argument("--camera", default="agentview")
    p.add_argument("--img_hw", type=int, default=224)
    p.add_argument("--eps_dist", type=float, default=0.03)
    p.add_argument("--goal_eps", type=float, default=None,
                   help="distance defining the REACH goal frame (first frame within "
                        "this of the cube). Defaults to --eps_dist if unset.")
    p.add_argument("--voxel", type=float, default=0.05)
    p.add_argument("--k_seed", type=int, default=30)
    p.add_argument("--j_roll", type=int, default=20)
    p.add_argument("--m_iters", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--encoder", default="dinov2", choices=["dinov2", "ijepa"],
                   help="goal-signal encoder (DINOv2 default; swappable)")
    p.add_argument("--dino_pool", default="cls", choices=["cls", "mean"],
                   help="DINOv2 pooling; try 'mean' for a more local cube signal")
    p.add_argument("--save_dir", default=None,
                   help="if set, save the goal frame, a candidate frame every "
                        "--save_every goal-checks, and every threshold-crossing frame")
    p.add_argument("--save_every", type=int, default=200,
                   help="save a candidate frame every N goal-checks (default 200)")
    a = p.parse_args()

    if a.synthetic:
        run_synthetic()
    else:
        import torch, h5py
        device = torch.device("mps" if torch.backends.mps.is_available()
                              else ("cuda" if torch.cuda.is_available() else "cpu"))
        print("device:", device)
        enc = (DINOv2Encoder(device, pooling=a.dino_pool)
               if a.encoder == "dinov2" else IJEPAEncoder(device))
        print(f"encoder: {enc.name}" +
              (f" ({a.dino_pool}-pool)" if a.encoder == "dinov2" else ""))
        # load the goal demo frames + true positions
        with h5py.File(a.goal_demo_hdf5, "r") as f:
            root = f["data"] if "data" in f else f
            og = root[a.goal_demo]["obs"]
            frames = [np.asarray(og[a.image_key][i]) for i in range(og[a.image_key].shape[0])]
            eef = np.asarray(og["robot0_eef_pos"][:], float)
            obj_full = np.asarray(og["object"][:], float)      # [pos(3), quat_xyzw(4), ...]
            obj = obj_full[:, :3]                               # cube pos
        # REACH goal: first frame within goal_eps of the cube (approach), NOT the
        # grasped-and-lifted last frame. Truncate the demo there so the lift tail
        # doesn't contaminate calibration.
        goal_eps = a.goal_eps if a.goal_eps is not None else a.eps_dist
        reach_idx, _, note = select_reach_index(eef, obj, goal_eps)
        print(note)
        frames = frames[:reach_idx + 1]
        eef = eef[:reach_idx + 1]
        obj = obj[:reach_idx + 1]
        goal_img = frames[-1]                                  # = reach frame
        cube_xyz = obj[reach_idx]                              # demo cube world pos
        cube_quat = obs_quat_to_mujoco(obj_full[reach_idx, 3:7])  # demo cube orientation
        thr, margin, rep = calibrate_threshold(enc, frames, eef, obj, a.eps_dist)
        print(f"CALIBRATION: threshold {thr:.4f} | {rep}")
        if margin <= 0:
            print("WARNING: similarity does not separate near/far cleanly "
                  "(encoder may be too flat for a reliable threshold).")
        # Freeze the cube at the demo's cube pose (position AND rotation) so
        # reset-and-replay is coherent and the goal frame's cube matches the live
        # cube every reset.
        backend = RobosuiteBackend(camera=a.camera, img_hw=a.img_hw,
                                   cube_pos=cube_xyz, cube_quat=cube_quat)
        rrt = GoExploreRRT(backend, enc, goal_img, sim_threshold=thr,
                           voxel=a.voxel, k_seed=a.k_seed, j_roll=a.j_roll,
                           m_iters=a.m_iters, eps_dist=a.eps_dist, rng_seed=a.seed,
                           save_dir=a.save_dir, save_every=a.save_every)
        res = rrt.run()
        if a.save_dir:
            import os
            rrt.save_tree(os.path.join(a.save_dir, "tree.npz"))
        print("\n=== RESULT ===")
        for k, v in res.items():
            print(f"  {k}: {v}")
        if res["terminated"]:
            print(f"\nENCODER TERMINATED the search: similarity crossed the "
                  f"threshold (sim {res['crossing_sim']:.4f} > {res['threshold']:.4f}).")
            print(f"AUDIT of that state: true gripper-cube distance "
                  f"{res['crossing_true_dist']:.4f} m vs eps {a.eps_dist} m -> "
                  f"{res['verdict']} "
                  + ("(gripper really was within eps -- encoder correct)"
                     if res['verdict'] == 'TRUE_POSITIVE'
                     else "(encoder said reached, but gripper was NOT within eps)"))
        else:
            print(f"\nEncoder NEVER crossed the threshold in the whole run "
                  f"(no termination). max similarity seen {res['max_sim_seen']:.4f} "
                  f"vs threshold {res['threshold']:.4f} -> "
                  + ("threshold was approached but never crossed (threshold may be "
                     "too strict)." if res['max_sim_seen'] >= res['threshold'] - 0.02
                     else "threshold never approached -- exploration likely never got "
                          "the gripper near the cube (reachability), or calibration "
                          "set the bar too high."))
