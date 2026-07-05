"""
Visualize a Go-Explore RRT tree saved by go_explore_rrt_reach.py (tree.npz).

Each node is plotted at its END-EFFECTOR (x,y,z) position; parent->child edges are
drawn as line segments; the CUBE (goal) and the START eef are marked. Nodes are
colored by true distance-to-cube so you can see at a glance whether the tree's
point cloud ever reached the neighborhood of the cube or stayed clustered away.

Usage:
    python3 visualize_tree.py --tree rrt_frames/tree.npz --out tree3d.png
    python3 visualize_tree.py --tree tree.npz --subsample 5 --max_edges 5000
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)
from mpl_toolkits.mplot3d.art3d import Line3DCollection


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tree", required=True, help="path to tree.npz")
    p.add_argument("--out", default="tree3d.png")
    p.add_argument("--subsample", type=int, default=1,
                   help="plot every Nth node (edges follow); use >1 for big trees")
    p.add_argument("--max_edges", type=int, default=20000,
                   help="cap on drawn edges for legibility (randomly thinned)")
    p.add_argument("--elev", type=float, default=25.0)
    p.add_argument("--azim", type=float, default=-60.0)
    a = p.parse_args()

    d = np.load(a.tree)
    eef = d["eef"]; parent = d["parent_id"]; depth = d["depth"]
    cube = d["cube"]; start = d["start_eef"]
    reached_id = int(d["reached_id"]) if "reached_id" in d else -1
    n = eef.shape[0]
    print(f"tree: {n} nodes | depth max {depth.max()} | "
          f"cube {np.round(cube,3)} | reached_id {reached_id}")

    dist = np.linalg.norm(eef - cube[None, :], axis=1)   # per-node distance to cube
    print(f"distance-to-cube: min {dist.min():.3f} | median {np.median(dist):.3f} "
          f"| max {dist.max():.3f}")
    print(f"closest node to cube: id {int(dist.argmin())} at "
          f"{np.round(eef[dist.argmin()],3)} ({dist.min():.3f} m)")

    # node subsample mask (always keep root, reached node, and global closest)
    keep = np.zeros(n, dtype=bool)
    keep[::max(1, a.subsample)] = True
    keep[0] = True
    keep[int(dist.argmin())] = True
    if reached_id >= 0:
        keep[reached_id] = True

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")

    # edges parent->child (only where both endpoints are kept), thinned to max_edges
    seg = [( eef[i], eef[parent[i]] )
           for i in range(n)
           if parent[i] >= 0 and keep[i] and keep[parent[i]]]
    if len(seg) > a.max_edges:
        idx = np.random.default_rng(0).choice(len(seg), a.max_edges, replace=False)
        seg = [seg[i] for i in idx]
    if seg:
        ax.add_collection3d(Line3DCollection(seg, colors="0.7", linewidths=0.3,
                                             alpha=0.5))

    # nodes colored by distance to cube
    ke = eef[keep]; kd = dist[keep]
    sc = ax.scatter(ke[:, 0], ke[:, 1], ke[:, 2], c=kd, cmap="viridis_r",
                    s=3, alpha=0.6)
    cb = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
    cb.set_label("distance to cube (m)")

    # cube (goal), start, and reached node
    ax.scatter(*cube, c="red", s=200, marker="*", label="cube (goal)",
               edgecolors="k", linewidths=0.5, depthshade=False)
    ax.scatter(*start, c="blue", s=120, marker="^", label="start EEF",
               edgecolors="k", linewidths=0.5, depthshade=False)
    if reached_id >= 0:
        ax.scatter(*eef[reached_id], c="lime", s=160, marker="o",
                   label="reached (encoder)", edgecolors="k", depthshade=False)
    # mark the closest-ever node
    ci = int(dist.argmin())
    ax.scatter(*eef[ci], c="orange", s=90, marker="D",
               label=f"closest node ({dist.min():.3f} m)",
               edgecolors="k", depthshade=False)

    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
    ax.set_title(f"Go-Explore RRT tree in EEF space  ({n} nodes)\n"
                 f"closest approach to cube: {dist.min():.3f} m")
    ax.legend(loc="upper left", fontsize=8)
    ax.view_init(elev=a.elev, azim=a.azim)
    fig.tight_layout()
    fig.savefig(a.out, dpi=140)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
