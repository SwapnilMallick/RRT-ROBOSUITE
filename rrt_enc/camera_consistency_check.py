"""
Camera-consistency check for the Go-Explore RRT.

The RRT decides "reached" by cosine-similarity between a freshly-rendered
candidate frame (from RobosuiteBackend) and the GOAL frame (stored in the demo
hdf5, produced by dataset_states_to_obs.py). Those are two DIFFERENT render
pipelines/env instantiations. If their camera name / pose / resolution / vertical
flip don't match, every similarity score carries a constant viewpoint offset
unrelated to gripper-cube proximity -- silently corrupting the whole experiment.

This script verifies the two pipelines agree by rendering the SAME joint
configuration through the RRT's backend and comparing to the demo's stored frame
for that timestep. If the cameras match, the two images should be nearly
identical (pixel MAE small, encoder cosine-sim ~1). It saves a side-by-side and
prints pixel + encoder metrics.

Note: robosuite randomizes cube position on reset, so we construct the backend
with the SAME cube_pos/cube_quat freezing mechanism go_explore_rrt_reach.py uses
in production (RobosuiteBackend's deterministic placement initializer), rather
than a separate ad-hoc override -- otherwise this check would validate a
different code path than the one the actual RRT run depends on, and could pass
while the real run's cube pose is still wrong.

Can be run from any directory -- --rrt_script and --out default to paths
relative to this script's own location, not the caller's cwd.
"""

import argparse
import os
import numpy as np
import h5py

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_go_explore(path="go_explore_rrt_reach.py"):
    import importlib.util
    spec = importlib.util.spec_from_file_location("gerrt", path)
    m = importlib.util.module_from_spec(spec)
    import sys; sys.argv = ["gerrt"]        # avoid argparse on import
    spec.loader.exec_module(m)
    return m


def pixel_stats(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    mae = float(np.abs(a - b).mean())
    # also check for a vertical-flip mismatch (common robosuite gotcha)
    mae_flip = float(np.abs(a - b[::-1]).mean())
    return mae, mae_flip


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--goal_demo_hdf5", required=True)
    p.add_argument("--goal_demo", default="demo_0")
    p.add_argument("--image_key", default="agentview_image")
    p.add_argument("--camera", default="agentview")
    p.add_argument("--img_hw", type=int, default=224)
    p.add_argument("--frame", type=int, default=0,
                   help="which demo timestep to reproduce (default 0 = start state)")
    p.add_argument("--encoder", default="dinov2", choices=["dinov2", "ijepa"])
    p.add_argument("--dino_pool", default="cls", choices=["cls", "mean"])
    p.add_argument("--rrt_script", default=os.path.join(SCRIPT_DIR, "go_explore_rrt_reach.py"))
    p.add_argument("--out", default=os.path.join(SCRIPT_DIR, "camera_check.png"))
    a = p.parse_args()

    import torch
    m = load_go_explore(a.rrt_script)
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    print("device:", device)

    # --- 1. load the demo's stored frame + its joint config + object state ---
    with h5py.File(a.goal_demo_hdf5, "r") as f:
        root = f["data"] if "data" in f else f
        g = root[a.goal_demo]
        og = g["obs"] if "obs" in g else g
        demo_img = np.asarray(og[a.image_key][a.frame])            # stored goal-pipeline frame
        # joint positions: prefer a 'robot0_joint_pos' obs; else fall back to states
        jkey = next((k for k in ("robot0_joint_pos", "joint_pos") if k in og), None)
        demo_q = np.asarray(og[jkey][a.frame], float) if jkey else None
        obj = np.asarray(og["object"][a.frame], float) if "object" in og else None
    print(f"demo frame {a.frame}: image {demo_img.shape} {demo_img.dtype}"
          + (f" | joints from '{jkey}'" if demo_q is not None else
             " | NO joint obs found (will use env reset pose)"))

    # --- 2. render the SAME config through the RRT backend, using the EXACT
    # cube-freezing mechanism go_explore_rrt_reach.py uses in production
    # (RobosuiteBackend's cube_pos/cube_quat -> deterministic placement
    # initializer). Constructing the backend without these and patching qpos
    # afterward would validate a different code path than the real RRT run.
    cube_pos = cube_quat = None
    if obj is not None:
        cube_pos = obj[0:3]
        cube_quat = m.obs_quat_to_mujoco(obj[3:7])   # 'object' obs -> [pos(3), quat_xyzw(4), ...]
        print(f"demo cube pose: pos={np.round(cube_pos,3)} quat(mj)={np.round(cube_quat,3)}")
    else:
        print("(no 'object' obs found; cube will not be frozen -- ignore the cube "
              "region when judging the viewpoint)")
    backend = m.RobosuiteBackend(camera=a.camera, img_hw=a.img_hw,
                                 cube_pos=cube_pos, cube_quat=cube_quat)
    if demo_q is not None:
        backend.set_q(demo_q)                 # match arm configuration exactly
    if cube_pos is not None:
        got = backend.cube_pos()
        xy_err = float(np.linalg.norm(got[:2] - cube_pos[:2]))
        print(f"cube read-back: xy={np.round(got[:2],4)} | target xy={np.round(cube_pos[:2],4)} "
              f"| xy-err {xy_err:.4f}  (z is physics-settled, not force-set; rotation is "
              "frozen via the placement initializer, see '[cube frozen]' line above)")
        if xy_err > 1e-3:
            print("  >> cube xy did NOT hold -- this is the exact mechanism "
                  "go_explore_rrt_reach.py uses, so this is a real freeze bug, not a "
                  "check-harness limitation.")
    backend_img = backend.render()

    # --- 3. pixel comparison (+ flip detector) ---
    mae, mae_flip = pixel_stats(demo_img, backend_img)
    print(f"\npixel MAE (0-255): {mae:.2f}")
    print(f"pixel MAE if backend flipped vertically: {mae_flip:.2f}")
    if mae_flip < mae * 0.5:
        print("  >> LIKELY VERTICAL-FLIP MISMATCH: the two pipelines disagree on "
              "image orientation. Fix the [::-1] convention before running the RRT.")
    elif mae > 40:
        print("  >> LARGE pixel difference: cameras may differ in pose/FOV/framing "
              "(or the cube pose couldn't be matched). Inspect the side-by-side.")
    else:
        print("  >> Pixel difference small: viewpoint/framing/orientation look "
              "consistent (residual diff is expected from cube pose / lighting).")

    # --- 4. encoder comparison (the metric the RRT actually uses) ---
    enc = (m.DINOv2Encoder(device, pooling=a.dino_pool) if a.encoder == "dinov2"
           else m.IJEPAEncoder(device))
    sim = m.cosine(enc.embed(demo_img), enc.embed(backend_img))
    print(f"\nencoder ({enc.name}) cosine-sim between demo frame and backend "
          f"render of the same config: {sim:.4f}")
    print("  >> If cameras match, this should be very high (~0.95+). A low value "
          "means the RRT's candidate renders live in a different visual "
          "distribution than the goal frame -> similarity threshold is measuring "
          "viewpoint differences, not gripper-cube proximity.")

    # --- 5. save side-by-side ---
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(12, 4.2))
        ax[0].imshow(demo_img); ax[0].set_title(f"demo hdf5 (goal pipeline)\nframe {a.frame}")
        ax[1].imshow(backend_img); ax[1].set_title("RRT backend render\n(same joints)")
        diff = np.abs(demo_img.astype(int) - backend_img.astype(int)).astype(np.uint8)
        ax[2].imshow(diff); ax[2].set_title(f"abs diff\nMAE {mae:.1f} | sim {sim:.3f}")
        for x in ax: x.set_xticks([]); x.set_yticks([])
        fig.suptitle("Camera consistency: demo frame vs RRT backend render "
                     "(same configuration)")
        fig.tight_layout(); fig.savefig(a.out, dpi=130)
        print(f"\nsaved side-by-side: {a.out}")
    except Exception as ex:
        print(f"(plot skipped: {ex})")


if __name__ == "__main__":
    main()
