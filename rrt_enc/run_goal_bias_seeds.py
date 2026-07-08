"""
Batch runner for rrt_ik_encoder_goal_bias.py: runs IKRRT across multiple seeds
(against the same frozen cube / goal frame) and reports aggregate TP/FP/
no-termination statistics. Encoder + demo data are loaded once and the
robosuite backend is reused (reset between seeds) to avoid reloading DINOv2 /
recompiling the MuJoCo model per seed.

Usage:
    python run_goal_bias_seeds.py --n_seeds 25
    python run_goal_bias_seeds.py --n_seeds 10 --start_seed 100 --sim_threshold 0.9
"""

import argparse
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_seeds", type=int, default=25, help="number of seeds to run")
    p.add_argument("--start_seed", type=int, default=0)
    p.add_argument("--goal_demo_hdf5", default="./datasets/lift/ph/image224.hdf5")
    p.add_argument("--goal_demo", default="demo_0")
    p.add_argument("--image_key", default="agentview_image")
    p.add_argument("--camera", default="agentview")
    p.add_argument("--img_hw", type=int, default=224)
    p.add_argument("--dino_pool", default="cls", choices=["cls", "mean"])
    p.add_argument("--sim_threshold", type=float, default=0.9)
    p.add_argument("--eps_dist", type=float, default=0.03)
    p.add_argument("--goal_eps", type=float, default=None)
    p.add_argument("--delta", type=float, default=0.05)
    p.add_argument("--goal_bias", type=float, default=0.15)
    p.add_argument("--null_space_gain", type=float, default=0.3)
    p.add_argument("--x_range", type=float, nargs=2, default=[-0.2, 0.3])
    p.add_argument("--y_range", type=float, nargs=2, default=[-0.3, 0.3])
    p.add_argument("--z_range", type=float, nargs=2, default=[0.80, 1.05])
    p.add_argument("--m_iters", type=int, default=1000)
    p.add_argument("--ik_tol", type=float, default=1e-3)
    p.add_argument("--out", default=None, help="optional path to save per-seed results as JSON")
    p.add_argument("--summary_out", default=None,
                   help="path to save the aggregate summary as text (default: "
                        "summary.txt next to --out, or ./summary.txt)")
    p.add_argument("--video_dir", default=None,
                   help="if set, saves a per-seed exploration video to <video_dir>/seed_XXXX.mp4")
    p.add_argument("--video_fps", type=int, default=10)
    a = p.parse_args()

    import os
    import torch, h5py
    from rrt_ik_encoder_goal_bias import (RobosuiteBackend, DINOv2Encoder, IKRRT,
                                           select_reach_index)

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    print("device:", device)
    enc = DINOv2Encoder(device, pooling=a.dino_pool)

    with h5py.File(a.goal_demo_hdf5, "r") as f:
        root = f["data"] if "data" in f else f
        og = root[a.goal_demo]["obs"]
        frames = [np.asarray(og[a.image_key][i]) for i in range(og[a.image_key].shape[0])]
        eef = np.asarray(og["robot0_eef_pos"][:], float)
        obj = np.asarray(og["object"][:], float)[:, :3]
        joint_pos = np.asarray(og["robot0_joint_pos"][:], float)

    goal_eps = a.goal_eps if a.goal_eps is not None else a.eps_dist
    ridx, _, note = select_reach_index(eef, obj, goal_eps)
    print(note)
    goal_img = frames[ridx]
    cube_xyz = obj[ridx]
    q_bias = joint_pos[ridx]
    print(f"goal frame t={ridx} | threshold {a.sim_threshold} | "
          f"null_space_gain {a.null_space_gain} | n_seeds {a.n_seeds}")

    backend = RobosuiteBackend(camera=a.camera, img_hw=a.img_hw, cube_pos=cube_xyz)

    if a.video_dir:
        os.makedirs(a.video_dir, exist_ok=True)

    results = []
    for seed in range(a.start_seed, a.start_seed + a.n_seeds):
        backend.reset()
        video_path = (os.path.join(a.video_dir, f"seed_{seed:04d}.mp4")
                      if a.video_dir else None)
        rrt = IKRRT(backend, enc, goal_img, sim_threshold=a.sim_threshold,
                    x_range=tuple(a.x_range), y_range=tuple(a.y_range),
                    z_range=tuple(a.z_range), delta=a.delta, m_iters=a.m_iters,
                    eps_dist=a.eps_dist, ik_tol=a.ik_tol, rng_seed=seed,
                    log_every=0, goal_bias=a.goal_bias, q_bias=q_bias,
                    null_space_gain=a.null_space_gain,
                    video_path=video_path, video_fps=a.video_fps)
        res = rrt.run()
        res["seed"] = seed
        results.append(res)

        print(f"\n=== RESULT (seed {seed}) ===")
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

        print(f"seed {seed:4d} | terminated={res['terminated']!s:5} "
              f"verdict={res['verdict']} sim={res['crossing_sim']} "
              f"true_dist={res['crossing_true_dist']} nodes={res['nodes']}")

    n = len(results)
    tp = sum(1 for r in results if r["verdict"] == "TRUE_POSITIVE")
    fp = sum(1 for r in results if r["verdict"] == "FALSE_POSITIVE")
    no_term = sum(1 for r in results if not r["terminated"])
    tp_sims = [r["crossing_sim"] for r in results if r["verdict"] == "TRUE_POSITIVE"]
    fp_sims = [r["crossing_sim"] for r in results if r["verdict"] == "FALSE_POSITIVE"]
    node_counts = [r["nodes"] for r in results]

    summary_lines = [f"=== SUMMARY over {n} seeds ==="]
    summary_lines.append(f"TRUE_POSITIVE:   {tp}/{n} ({100 * tp / n:.1f}%)")
    summary_lines.append(f"FALSE_POSITIVE:  {fp}/{n} ({100 * fp / n:.1f}%)")
    summary_lines.append(f"no termination:  {no_term}/{n} ({100 * no_term / n:.1f}%)")
    if tp_sims:
        summary_lines.append(f"TP sim: mean {np.mean(tp_sims):.4f}  min {np.min(tp_sims):.4f}  "
                              f"max {np.max(tp_sims):.4f}")
    if fp_sims:
        summary_lines.append(f"FP sim: mean {np.mean(fp_sims):.4f}  min {np.min(fp_sims):.4f}  "
                              f"max {np.max(fp_sims):.4f}")
    summary_lines.append(f"nodes: mean {np.mean(node_counts):.1f}  min {np.min(node_counts)}  "
                          f"max {np.max(node_counts)}")

    print("\n" + "\n".join(summary_lines))

    if a.out:
        import json
        with open(a.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nsaved per-seed results -> {a.out}")

    summary_out = a.summary_out
    if summary_out is None:
        summary_out = os.path.join(os.path.dirname(a.out), "summary.txt") if a.out else "summary.txt"
    with open(summary_out, "w") as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"saved summary -> {summary_out}")


if __name__ == "__main__":
    main()
