# RRT Motion Planning for Franka Panda in Robosuite

A joint-space RRT (Rapidly-exploring Random Tree) motion planner for the Franka Panda 7-DOF arm in the robosuite `Lift` environment. Plans collision-free paths to reach 8 cm above a target cube and optionally smooths them with random shortcutting + cubic B-spline fitting.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Basic RRT](#basic-rrt)
  - [RRT with Path Smoothing](#rrt-with-path-smoothing)
- [Key Parameters](#key-parameters)
- [Algorithms](#algorithms)
- [Outputs](#outputs)
- [Examples](#examples)

---

## Overview

| Feature | Details |
|---|---|
| Robot | Franka Panda (7 joints + gripper) |
| Simulator | MuJoCo via [robosuite](https://robosuite.ai/) |
| Planning space | 7D joint angle space |
| Goal checking | Task space — end-effector position within threshold of target |
| Collision checking | MuJoCo contact detection (self + environment) |
| Path smoothing | Random shortcutting + cubic B-spline fitting |

The planner uses **goal biasing** (15% chance to sample near an IK seed) and **damped least-squares IK** to compute the seed configuration, making convergence significantly faster than uniform random sampling.

---

## Project Structure

```
ROBOSUITE_RRT/
├── rrt/
│   ├── rrt_panda.py          # RRT planner (no smoothing)
│   ├── plans/                # Saved waypoint trajectories (.npz)
│   ├── trees/                # RRT tree visualizations (.png)
│   └── videos/               # Execution recordings (.mp4)
└── rrt_smooth/
    ├── rrt_panda_smooth.py   # RRT + B-spline path smoothing
    ├── plans/                # Original and smoothed trajectories (.npz)
    ├── trees/                # Tree and comparison plots (.png)
    └── videos/               # Execution recordings (.mp4)
```

---

## Installation

This project requires `robosuite`, `mujoco`, and a few scientific Python libraries.

```bash
# Activate your environment (adjust path as needed)
source /path/to/your/env/bin/activate

# Install dependencies
pip install robosuite mujoco numpy opencv-python matplotlib scipy
```

---

## Usage

### Basic RRT

Plan and execute a path without smoothing:

```bash
cd rrt
python rrt_panda.py
```

Plan only (skip execution):

```bash
python rrt_panda.py --no_exec
```

Show the MuJoCo viewer during execution:

```bash
python rrt_panda.py --render
```

### RRT with Path Smoothing

Plan and smooth (shortcutting + B-spline), then execute the smoothed path:

```bash
cd rrt_smooth
python rrt_panda_smooth.py --smoother shortcut_bspline
```

Execute both the original and smoothed paths (for comparison):

```bash
python rrt_panda_smooth.py --execute both
```

Use B-spline only (no shortcutting):

```bash
python rrt_panda_smooth.py --smoother bspline_only
```

---

## Key Parameters

### RRT

| Flag | Default | Description |
|---|---|---|
| `--step_size` | `0.1` | Max joint change per RRT extension (rad) |
| `--goal_threshold` | `0.03` | EEF distance to goal for success (m) |
| `--goal_bias` | `0.15` | Fraction of iterations biased toward IK seed |
| `--max_iters` | `5000` | Maximum RRT iterations |
| `--interp_step` | `0.02` | Edge collision interpolation step (rad) |
| `--exec_tol` | `0.02` | Joint error tolerance per waypoint |
| `--exec_max_steps` | `150` | Max sim steps per waypoint during execution |
| `--no_exec` | — | Skip execution after planning |
| `--render` | — | Show MuJoCo viewer |
| `--show` | — | Show matplotlib plots interactively |
| `--video_fps` | `20` | Frames per second for output video |

### Path Smoothing (rrt_panda_smooth.py only)

| Flag | Default | Description |
|---|---|---|
| `--smoother` | `shortcut_bspline` | `shortcut_bspline` or `bspline_only` |
| `--n_shortcut` | `200` | Number of random shortcut attempts |
| `--spline_s` | `0.0` | B-spline smoothing factor (`0` = interpolating) |
| `--n_samples` | `100` | Waypoints sampled from the fitted spline |
| `--execute` | `smoothed` | Which path to execute: `original`, `smoothed`, or `both` |
| `--plot_per_joint` | — | Save per-joint angle comparison plots |

---

## Algorithms

### RRT (Rapidly-exploring Random Tree)

Explores 7D joint space by iteratively sampling random configurations, finding the nearest existing node, and extending toward the sample by `step_size` radians. A new node is added if the edge is collision-free. Success is declared when the forward kinematics of a node places the end-effector within `goal_threshold` of the target position.

**Goal biasing**: With probability `goal_bias`, the sampler draws from a Gaussian centered on the IK seed configuration instead of sampling uniformly. This dramatically reduces the number of iterations needed to find a path.

### Damped Least-Squares IK

Computes a joint-space seed near the goal using the Jacobian pseudoinverse with a damping factor (λ = 0.05) to avoid singularities. Iterates until the end-effector is within 8 mm of the goal or a step limit is reached.

### Random Shortcutting

After planning, randomly selects pairs of waypoints (i, j) and checks if the direct edge between them is collision-free. If so, all intermediate waypoints are removed. Runs for `n_shortcut` iterations, reducing path length and jaggedness before B-spline fitting.

### Cubic B-spline Fitting

Fits a cubic B-spline (degree 3) to the anchor points from shortcutting. The smoothing factor `spline_s` controls the trade-off between closeness to anchor points (`s=0` interpolates exactly) and overall smoothness (`s>0` approximates). `n_samples` evenly spaced points are then re-sampled from the spline to produce the final trajectory. Joint limit violations from spline overshoot are clamped, and the final path is validated for collisions.

---

## Outputs

Each run produces timestamped files in the respective subdirectories:

| File | Contents |
|---|---|
| `plans/plan_YYYYMMDD_HHMMSS.npz` | Planned waypoints as joint configurations |
| `plans/plan_smoothed_YYYYMMDD_HHMMSS.npz` | Smoothed waypoints (rrt_smooth only) |
| `trees/tree_YYYYMMDD_HHMMSS.png` | 3D task-space visualization of the RRT tree |
| `trees/path_comparison_YYYYMMDD_HHMMSS.png` | Side-by-side original vs smoothed paths |
| `trees/joints_comparison_YYYYMMDD_HHMMSS.png` | Per-joint angle profile comparison |
| `videos/execution_YYYYMMDD_HHMMSS.mp4` | Rendered video of path execution with HUD |

---

## Examples

**Custom RRT parameters**:

```bash
python rrt/rrt_panda.py --step_size 0.05 --max_iters 10000 --goal_threshold 0.02
```

**Smoother with more shortcut attempts and an approximating spline**:

```bash
python rrt_smooth/rrt_panda_smooth.py \
  --smoother shortcut_bspline \
  --n_shortcut 500 \
  --spline_s 0.1 \
  --n_samples 150 \
  --execute both \
  --render
```
