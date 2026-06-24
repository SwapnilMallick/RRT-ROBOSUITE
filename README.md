# RRT Motion Planning for Franka Panda in Robosuite

A collection of motion planning experiments for the Franka Panda 7-DOF arm in the robosuite `Lift` environment. Covers joint-space RRT, path smoothing, visual latent-space representation learning (Beta-VAE), and latent-space planning using a DINO-based world model (DINO-WM).

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Tracks](#tracks)
  - [1. Basic RRT](#1-basic-rrt)
  - [2. RRT with Path Smoothing](#2-rrt-with-path-smoothing)
  - [3. Beta-VAE for Latent Space](#3-beta-vae-for-latent-space)
  - [4. DINO-WM Floor Control Baseline](#4-dino-wm-floor-control-baseline)
- [Key Parameters](#key-parameters)
- [Algorithms](#algorithms)
- [Outputs](#outputs)

---

## Overview

| Experiment | Space | Method |
|---|---|---|
| Basic RRT | 7D joint space | RRT with goal biasing |
| RRT + Smoothing | 7D joint space | RRT → shortcutting → B-spline |
| Beta-VAE | Visual (84×84) | MS-SSIM + L1 + KL annealing |
| DINO-WM CEM | Visual latent | CEM over frozen DINOv2 + ViT predictor |

All experiments use the robosuite `Lift` environment with the Franka Panda arm. The joint-space planners use MuJoCo contact detection for collision checking. The visual experiments consume robomimic HDF5 datasets.

---

## Project Structure

```
ROBOSUITE_RRT/
├── rrt/
│   ├── rrt_panda.py              # RRT planner (no smoothing)
│   ├── plans/                    # Saved waypoint trajectories (.npz)
│   ├── trees/                    # RRT tree visualizations (.png)
│   └── videos/                   # Execution recordings (.mp4)
├── rrt_smooth/
│   ├── rrt_panda_smooth.py       # RRT + B-spline path smoothing
│   ├── plans/
│   ├── trees/
│   └── videos/
├── rrt_latent_space/
│   ├── train_betavae.py          # Beta-VAE training
│   ├── recon_check.py            # Reconstruction quality check
│   ├── extract_robomimic_images.py
│   ├── vae_images.hdf5           # Extracted training images
│   └── ckpt/                     # Saved checkpoints
├── dino_wm/
│   ├── floor_control_plan.py     # Floor-metric CEM baseline (run from here or dino_wm_repo/)
│   └── lift_dset.py              # Robomimic → DINO-WM dataset adapter
├── dino_wm_repo/                 # Upstream DINO-WM submodule
│   ├── models/                   # VWorldModel, DINOv2 encoder, ViT predictor
│   ├── planning/                 # CEM, GD planners + objectives
│   ├── datasets/                 # TrajDataset base + env-specific adapters
│   └── conf/                     # Hydra configs (encoder, predictor, planner, env)
└── datasets/lift/ph/
    ├── image.hdf5                # 84×84 robomimic demos
    └── image224.hdf5             # 224×224 robomimic demos (for DINO-WM)
```

---

## Installation

```bash
# Activate your environment
source /path/to/your/env/bin/activate

# Core dependencies
pip install robosuite mujoco numpy opencv-python matplotlib scipy

# For Beta-VAE and DINO-WM tracks
pip install torch einops h5py hydra-core omegaconf
```

---

## Tracks

### 1. Basic RRT

Joint-space RRT planner. Plans a collision-free path from the Panda's home configuration to 8 cm above the cube, then executes it.

```bash
cd rrt
python rrt_panda.py                          # plan + execute
python rrt_panda.py --no_exec               # plan only
python rrt_panda.py --render                # show MuJoCo viewer
python rrt_panda.py --step_size 0.05 --goal_threshold 0.02 --max_iters 10000
```

### 2. RRT with Path Smoothing

Same as above, but post-processes the raw RRT path with random shortcutting followed by cubic B-spline fitting.

```bash
cd rrt_smooth
python rrt_panda_smooth.py                           # default: shortcut + B-spline, execute smoothed
python rrt_panda_smooth.py --smoother bspline_only   # skip shortcutting
python rrt_panda_smooth.py --execute both --render   # compare original vs smoothed
python rrt_panda_smooth.py --no_exec --plot_per_joint
python rrt_panda_smooth.py --spline_s 0.01 --n_samples 150 --n_shortcut 300
```

### 3. Beta-VAE for Latent Space

Trains a convolutional Beta-VAE on 84×84 robomimic `agentview_image` frames. Uses MS-SSIM + L1 reconstruction with KL annealing. The trained encoder/decoder expose `encode(x) -> z` and `decode(z) -> x_hat` for downstream latent-space planners.

**Prepare images** (extract from robomimic HDF5 if not already done):
```bash
python3 rrt_latent_space/extract_robomimic_images.py
```

**Train:**
```bash
python3 rrt_latent_space/train_betavae.py \
  --data rrt_latent_space/vae_images.hdf5 \
  --key agentview_image \
  --latent 32 --beta 1.0 --anneal 10 --alpha 0.85 \
  --epochs 100 --batch 128 --lr 1e-3 --workers 4 \
  --ckpt_dir rrt_latent_space/ckpt
```

**Check reconstruction quality:**
```bash
python3 rrt_latent_space/recon_check.py \
  --ckpt rrt_latent_space/ckpt/betavae_last.pt \
  --hdf5 datasets/lift/ph/image.hdf5 \
  --key agentview_image --filter_key valid --n 8 \
  --out rrt_latent_space/recon_check.png
```

### 4. DINO-WM Floor Control Baseline

Establishes a **floor metric** for latent-space planning: runs CEM over a frozen DINOv2 encoder and a *randomly-initialized* ViT predictor. With no learned dynamics, the objective landscape is meaningless and CEM cannot make progress — this is the baseline a trained DINO-WM must beat.

The script requires `models` and `planning` from `dino_wm_repo/`, so either run from inside it or set `DINO_WM_ROOT`:

```bash
export DINO_WM_ROOT=/path/to/ROBOSUITE_RRT/dino_wm_repo

# Smoke test — CPU, no real model or dataset
cd dino_wm_repo
python floor_control_plan.py --smoke

# Full run (GPU recommended) — needs the 224px dataset
export DATASET_DIR=/path/to/ROBOSUITE_RRT/datasets/lift/ph
python floor_control_plan.py --n_evals 3 --opt_steps 10 --num_samples 100
```

Results are saved to `dino_wm_repo/results/floor_<timestamp>.json`.

The `dino_wm/lift_dset.py` adapter bridges the robomimic HDF5 into the `TrajDataset` interface that DINO-WM's training pipeline expects. Place or symlink it at `dino_wm_repo/datasets/lift_dset.py` before running `train.py`.

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

### Path Smoothing

| Flag | Default | Description |
|---|---|---|
| `--smoother` | `shortcut_bspline` | `shortcut_bspline` or `bspline_only` |
| `--n_shortcut` | `200` | Number of random shortcut attempts |
| `--spline_s` | `0.0` | B-spline smoothing factor (`0` = interpolating) |
| `--n_samples` | `100` | Waypoints sampled from the fitted spline |
| `--execute` | `smoothed` | Which path to run: `original`, `smoothed`, or `both` |
| `--plot_per_joint` | — | Save per-joint angle profile comparison |

### Beta-VAE

| Flag | Default | Description |
|---|---|---|
| `--latent` | `32` | Latent dimension |
| `--beta` | `1.0` | KL weight at end of annealing |
| `--anneal` | `10` | Epochs over which KL is linearly annealed in |
| `--alpha` | `0.85` | MS-SSIM weight (`1 - alpha` goes to L1) |
| `--epochs` | `100` | Training epochs |
| `--batch` | `128` | Batch size |

### DINO-WM Floor Control

| Flag | Default | Description |
|---|---|---|
| `--num_hist` | `3` | Observation history frames fed to the world model |
| `--horizon` | `5` | CEM planning horizon (steps) |
| `--action_dim` | `35` | Action dimension (7-DoF × frameskip 5) |
| `--num_samples` | `100` | CEM population size |
| `--topk` | `10` | CEM elite count |
| `--opt_steps` | `10` | CEM optimization steps |
| `--n_evals` | `3` | Number of test episodes to evaluate |
| `--smoke` | — | Fast CPU smoke test with synthetic data |

---

## Algorithms

### RRT

Iteratively samples random joint configurations, finds the nearest tree node, extends by `step_size`, and adds the node if the edge is collision-free. With probability `goal_bias`, samples near the IK seed instead of uniformly — this reduces convergence from thousands of iterations to ~200–500.

### Damped Least-Squares IK

Computes a joint-space seed near the goal using the Jacobian pseudoinverse with damping factor λ = 0.05 to avoid singularities near joint limits.

### Random Shortcutting

Randomly selects waypoint pairs (i, j) and checks whether the direct edge is collision-free. Removes all intermediate waypoints when it is. Reduces path length before B-spline fitting.

### Cubic B-spline Fitting

Fits a degree-3 spline to the shortcutted anchors via `scipy.interpolate.splprep`. `spline_s = 0` interpolates exactly; `spline_s > 0` approximates (smoother but may deviate). Joint-limit violations from spline overshoot are clamped, and the result is collision-validated.

### Beta-VAE (MS-SSIM + L1 + KL)

Loss = α · (1 − MS-SSIM(x, x̂)) + (1 − α) · L1(x, x̂) + β(t) · KL  

MS-SSIM captures multi-scale structure (edges, shape) better than MSE. L1 anchors absolute color. KL is linearly annealed from 0 so the decoder learns to reconstruct before the prior pressure collapses the posterior. A 7×7 window is used (vs. the standard 11×11) because the image size (84px) underflows at the finest scale with larger windows.

### DINO-WM CEM

Cross-Entropy Method over the action sequence: samples N action trajectories from N(μ, σ), rolls each forward through the world model, scores by latent distance to goal at the final step, refits μ/σ from the top-K elites. With a random predictor the scores are noise, so μ/σ don't improve — establishing the floor.

---

## Outputs

### RRT / RRT-Smooth

| File | Contents |
|---|---|
| `plans/plan_<ts>.npz` | Planned joint-space waypoints |
| `plans/plan_smoothed_<ts>.npz` | Smoothed waypoints (rrt_smooth only) |
| `trees/tree_<ts>.png` | Task-space RRT tree visualization |
| `trees/path_comparison_<ts>.png` | Original vs smoothed path overlay |
| `trees/joints_comparison_<ts>.png` | Per-joint angle profiles |
| `videos/execution_<ts>.mp4` | Rendered execution video with HUD |

### Beta-VAE

| File | Contents |
|---|---|
| `rrt_latent_space/ckpt/betavae_last.pt` | Final checkpoint |
| `rrt_latent_space/recon_check.png` | Grid of input vs reconstructed frames |

### DINO-WM Floor Control

| File | Contents |
|---|---|
| `dino_wm_repo/results/floor_<ts>.json` | Loss history, floor start/end, config |
