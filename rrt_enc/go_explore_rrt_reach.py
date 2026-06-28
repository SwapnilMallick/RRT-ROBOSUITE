"""
Go-Explore style reset-and-replay RRT for the Franka cube-REACH task in robosuite,
with an off-the-shelf encoder (I-JEPA by default) providing the goal signal.

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


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


# =========================================================================== #
# Robosuite/MuJoCo backend interface. The RRT only talks to THIS class, so the
# synthetic backend can stand in for verification.
# =========================================================================== #
class RobosuiteBackend:
    """Wraps the env so the planner is sim-agnostic. Implements kinematic moves
    (set qpos + mj_forward), collision checks, EEF position, and rendering."""
    def __init__(self, env_name="Lift", robot="Panda", camera="agentview",
                 img_hw=224, n_collision_substeps=8):
        import robosuite as suite
        from robosuite.controllers import load_composite_controller_config
        self.suite = suite
        cfg = load_composite_controller_config(controller="BASIC", robot=robot)
        self.env = suite.make(
            env_name, robots=robot, controller_configs=cfg,
            has_renderer=False, has_offscreen_renderer=True,
            use_camera_obs=True, camera_names=camera,
            camera_heights=img_hw, camera_widths=img_hw, ignore_done=True)
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
        return self.sim.data.get_site_xpos("gripper0_right_grip_site").copy()

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
                 eps_dist=0.03, rng_seed=0, log_every=200):
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

        q0 = self.b.get_q()
        self.nodes = [Node(0, q0, None, None, self.b.eef_pos())]
        self.bins = {}                    # voxel -> list[node_id]
        self._bin_add(self.nodes[0])
        self.reached = None
        self.terminations = []            # (sim, true_dist, TP/FP) audit log
        self.collisions = self.resets = self.replay_steps = 0

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
        """Termination by ENCODER ONLY; true distance logged for audit."""
        img = self.b.render()
        sim = cosine(self.enc.embed(img), self.z_goal)
        if sim > self.thresh:
            true_d = float(np.linalg.norm(eef - self.b.cube_pos()))
            tp = true_d < self.eps
            self.terminations.append((sim, true_d, tp))
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
                      f"false-term {sum(1 for _,_,tp in self.terminations if not tp)}")
        return self._result()

    def _result(self):
        tps = [t for t in self.terminations if t[2]]
        fps = [t for t in self.terminations if not t[2]]
        return dict(
            reached=self.reached is not None,
            nodes=len(self.nodes), voxels=len(self.bins),
            collisions=self.collisions, resets=self.resets,
            replay_steps=self.replay_steps,
            terminations=len(self.terminations),
            true_positive=len(tps), false_positive=len(fps),
            encoder_precision=(len(tps) / max(1, len(self.terminations))),
        )


# =========================================================================== #
# Threshold calibration from demos (so the threshold is physical, not guessed)
# =========================================================================== #
def calibrate_threshold(encoder, demo_frames, eef_pos, cube_pos, eps):
    """Return (threshold, margin, report). threshold = similarity that best
    separates 'within eps of cube' from 'farther', using the goal=last frame."""
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
    p.add_argument("--voxel", type=float, default=0.05)
    p.add_argument("--k_seed", type=int, default=30)
    p.add_argument("--j_roll", type=int, default=20)
    p.add_argument("--m_iters", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    if a.synthetic:
        run_synthetic()
    else:
        import torch, h5py
        device = torch.device("mps" if torch.backends.mps.is_available()
                              else ("cuda" if torch.cuda.is_available() else "cpu"))
        print("device:", device)
        enc = IJEPAEncoder(device)
        # load the goal demo frames + true positions for threshold calibration
        with h5py.File(a.goal_demo_hdf5, "r") as f:
            root = f["data"] if "data" in f else f
            og = root[a.goal_demo]["obs"]
            frames = [np.asarray(og[a.image_key][i]) for i in range(og[a.image_key].shape[0])]
            eef = np.asarray(og["robot0_eef_pos"][:], float)
            obj = np.asarray(og["object"][:], float)[:, :3]   # cube pos
        thr, margin, rep = calibrate_threshold(enc, frames, eef, obj, a.eps_dist)
        print(f"CALIBRATION: threshold {thr:.4f} | {rep}")
        if margin <= 0:
            print("WARNING: similarity does not separate near/far cleanly "
                  "(encoder may be too flat for a reliable threshold).")
        goal_img = frames[-1]
        backend = RobosuiteBackend(camera=a.camera, img_hw=a.img_hw)
        rrt = GoExploreRRT(backend, enc, goal_img, sim_threshold=thr,
                           voxel=a.voxel, k_seed=a.k_seed, j_roll=a.j_roll,
                           m_iters=a.m_iters, eps_dist=a.eps_dist, rng_seed=a.seed)
        res = rrt.run()
        print("\n=== RESULT ===")
        for k, v in res.items():
            print(f"  {k}: {v}")
        print(f"\nENCODER AUDIT: of {res['terminations']} encoder-'reached' events, "
              f"{res['true_positive']} were truly within {a.eps_dist}m of the cube "
              f"({res['encoder_precision']*100:.0f}% precision). "
              "Low precision => encoder declares 'reached' when it isn't.")
