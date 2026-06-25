"""
KNOWN-FAIL CONTROL: open-loop latent-space CEM planning with DINO-WM's real ViT
predictor at RANDOM init (no predictor training).

Purpose (this run is a guaranteed failure, run deliberately):
  * confirm the encode -> rollout -> CEM loop runs end-to-end without crashing,
  * establish the FLOOR metric (final latent distance-to-goal with an untrained
    predictor) that a *trained* DINO-WM must beat to show it learned anything.

It is "open-loop / latent-space" because we never step robosuite: success is
measured purely as how close CEM can drive the *predicted* final latent to the
goal latent. With a random predictor, rollouts are noise, so the objective
landscape is meaningless and CEM cannot make progress -- that's the point.

Faithfulness: uses the repo's real models.visual_world_model.VWorldModel
(frozen DINOv2 encoder + randomly-initialized ViT predictor) and the repo's real
planning.objectives.create_objective_fn. The CEM loop mirrors planning/cem.py's
math (sample ~ N(mu,sigma), rollout, score, keep top-k, refit mu/sigma) without
the evaluator/wandb/preprocessor stack so it runs standalone.

Run inside the dino_wm repo (so `models` / `planning` import). Real use needs a
GPU; a --smoke flag verifies the loop on synthetic features (CPU).
"""

import argparse
import json
import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from einops import repeat

# Add the DINO-WM repo root to sys.path so `models` and `planning` packages are
# importable.  Set DINO_WM_ROOT to the repo root when running from outside it;
# when this script is copied into the repo root, the default (same dir as __file__)
# is correct.
_DINO_WM_ROOT = os.environ.get(
    "DINO_WM_ROOT", os.path.dirname(os.path.abspath(__file__))
)
if _DINO_WM_ROOT not in sys.path:
    sys.path.insert(0, _DINO_WM_ROOT)

from models.visual_world_model import VWorldModel
from planning.objectives import create_objective_fn


# --------------------------------------------------------------------------- #
def build_decoder(decoder_path, device):
    """Instantiate VQVAE with emb_dim=384 (matching DINOv2 visual token dim).

    Must override the default emb_dim=64 from vqvae.yaml, which is wrong for
    DINOv2 features. quantize=False skips the discrete codebook so the decoder
    path runs without a pretrained codebook.

    If decoder_path is given and exists, loads checkpoint weights. Otherwise
    decoder weights are random — decoded images will be visual noise, which is
    the expected and correct output for the floor-control (random predictor) run.
    """
    from models.vqvae import VQVAE
    decoder = VQVAE(emb_dim=384, quantize=False)
    if decoder_path and os.path.exists(decoder_path):
        ckpt = torch.load(decoder_path, map_location=device)
        decoder.load_state_dict(ckpt)
        print(f"[decoder] loaded checkpoint from {decoder_path}")
    else:
        if decoder_path:
            print(f"[decoder] WARNING: path {decoder_path!r} not found — using random weights.")
        else:
            print("[decoder] no --decoder_path provided; using random VQVAE weights. "
                  "Decoded images will be noise (correct for the floor-control baseline).")
    decoder = decoder.to(device)
    decoder.eval()
    return decoder


def build_wm(cfg, device):
    """Instantiate the real VWorldModel with a frozen DINOv2 encoder and a
    RANDOM-init predictor + action/proprio encoders. Mirrors how train.py wires
    the model, but loads NO predictor weights (the whole point of this control).

    num_patches is derived from the config (matching train.py's formula:
      num_patches = (img_size // decoder_scale)^2, decoder_scale=16
    rather than the CLI --num_patches flag, which is dropped for this path.

    Uses OmegaConf directly (instead of Hydra's compose) to avoid the
    submitit_slurm launcher dependency baked into train.yaml's defaults.
    """
    import hydra.utils
    from omegaconf import OmegaConf

    _conf_dir = os.path.abspath(os.path.join(_DINO_WM_ROOT, "conf"))
    train_cfg = OmegaConf.load(os.path.join(_conf_dir, "train.yaml"))
    # Build a flat config with the fields build_wm actually uses.
    c = OmegaConf.create({
        "img_size":           train_cfg.get("img_size", 224),
        "num_hist":           cfg.num_hist,
        "num_pred":           train_cfg.get("num_pred", 1),
        "concat_dim":         train_cfg.get("concat_dim", 1),
        "action_emb_dim":     train_cfg.get("action_emb_dim", 10),
        "proprio_emb_dim":    train_cfg.get("proprio_emb_dim", 10),
        "num_action_repeat":  train_cfg.get("num_action_repeat", 1),
        "num_proprio_repeat": train_cfg.get("num_proprio_repeat", 1),
        "encoder":       OmegaConf.load(os.path.join(_conf_dir, "encoder",        "dino.yaml")),
        "proprio_encoder":OmegaConf.load(os.path.join(_conf_dir, "proprio_encoder","proprio.yaml")),
        "action_encoder": OmegaConf.load(os.path.join(_conf_dir, "action_encoder", "proprio.yaml")),
        "predictor":      OmegaConf.load(os.path.join(_conf_dir, "predictor",      "vit.yaml")),
    })

    # encoder (frozen DINOv2), action/proprio encoders, predictor (random init)
    encoder = hydra.utils.instantiate(c.encoder)
    proprio_encoder = hydra.utils.instantiate(
        c.proprio_encoder, in_chans=cfg.proprio_dim, emb_dim=c.proprio_emb_dim)
    action_encoder = hydra.utils.instantiate(
        c.action_encoder, in_chans=cfg.action_dim, emb_dim=c.action_emb_dim)
    proprio_emb = proprio_encoder.emb_dim
    action_emb = action_encoder.emb_dim
    enc_emb = encoder.emb_dim if hasattr(encoder, "emb_dim") else cfg.enc_emb_dim

    # Match train.py's num_patches calculation exactly (decoder_scale=16 from vqvae).
    # DINOv2-ViT-S/14 with img_size=224: VWorldModel resizes input to
    # (img_size // 16) * patch_size = 14*14 = 196 before encoding, yielding
    # (img_size // 16)^2 = 196 patches — NOT 224/14=16^2=256 as naive math suggests.
    if encoder.latent_ndim == 1:
        num_patches = 1
    else:
        decoder_scale = 16
        num_patches = (c.img_size // decoder_scale) ** 2   # 196 for 224px
    if c.concat_dim == 0:
        num_patches += 2  # action + proprio as extra tokens (concat_dim=0 mode)

    print(f"[build_wm] num_patches derived from config: {num_patches} "
          f"(img_size={c.img_size}, enc latent_ndim={encoder.latent_ndim})")

    # Predictor dim matches train.py: enc_emb + (proprio+action extra) * concat_dim.
    # With concat_dim=1 the extra embeddings are tiled into the feature dim;
    # with concat_dim=0 they are appended as extra patch tokens (no extra dim).
    predictor_dim = enc_emb + (proprio_emb * c.num_proprio_repeat
                                + action_emb * c.num_action_repeat) * c.concat_dim
    predictor = hydra.utils.instantiate(
        c.predictor,
        num_patches=num_patches,
        num_frames=c.num_hist,
        dim=predictor_dim,
    )
    print(f"[build_wm] predictor dim={predictor_dim} "
          f"(enc={enc_emb} + proprio={proprio_emb}*{c.num_proprio_repeat} "
          f"+ action={action_emb}*{c.num_action_repeat}) * concat_dim={c.concat_dim}")

    wm = VWorldModel(
        image_size=c.img_size, num_hist=c.num_hist, num_pred=c.num_pred,
        encoder=encoder, proprio_encoder=proprio_encoder,
        action_encoder=action_encoder, predictor=predictor, decoder=None,
        proprio_dim=proprio_emb, action_dim=action_emb,
        concat_dim=c.concat_dim, num_action_repeat=c.num_action_repeat,
        num_proprio_repeat=c.num_proprio_repeat,
        train_encoder=False, train_predictor=False, train_decoder=False,
    ).to(device)
    wm.eval()  # VWorldModel.eval() doesn't return self, so call separately
    print("[build_wm] VWorldModel built — encoder frozen, predictor RANDOM init, "
          "no weights loaded.")
    return wm


def load_real_obs(data_path, num_hist, n_evals, img_size, device):
    """Load (obs_0, obs_g) pairs from the test split of image224.hdf5.

    obs_0 : first num_hist frames of each demo (visual + proprio).
    obs_g : final frame of each demo (the lifted-cube end state).

    Observations come out of LiftDataset already preprocessed:
      visual  : (T, C, H, W) float32 in [-1, 1]  (Resize+CenterCrop+Normalize)
      proprio : (T, 9)  float32  (Z-score normalised by the dataset)
    No further transforms are applied here — this matches how train.py feeds the model.
    """
    from datasets.lift_dset import LiftDataset
    from datasets.img_transforms import default_transform

    transform = default_transform(img_size=img_size)
    dset = LiftDataset(
        data_path=data_path,
        filter_key="test",
        transform=transform,
        normalize_action=True,
    )
    print(f"[load_real_obs] LiftDataset loaded: {len(dset)} test demos from {data_path}")

    n = min(n_evals, len(dset))
    obs_0_vis, obs_0_pro = [], []
    obs_g_vis, obs_g_pro = [], []

    for i in range(n):
        obs, _, _, _ = dset[i]            # obs["visual"]: (T, C, H, W)
        obs_0_vis.append(obs["visual"][:num_hist])    # (num_hist, C, H, W)
        obs_0_pro.append(obs["proprio"][:num_hist])   # (num_hist, 9)
        obs_g_vis.append(obs["visual"][-1:])          # (1, C, H, W)
        obs_g_pro.append(obs["proprio"][-1:])         # (1, 9)

    obs_0 = {
        "visual": torch.stack(obs_0_vis).to(device),    # (n, num_hist, C, H, W)
        "proprio": torch.stack(obs_0_pro).to(device),   # (n, num_hist, 9)
    }
    obs_g = {
        "visual": torch.stack(obs_g_vis).to(device),    # (n, 1, C, H, W)
        "proprio": torch.stack(obs_g_pro).to(device),   # (n, 1, 9)
    }
    print(f"[load_real_obs] obs_0 visual {tuple(obs_0['visual'].shape)}, "
          f"obs_g visual {tuple(obs_g['visual'].shape)}")
    return obs_0, obs_g


# --------------------------------------------------------------------------- #
# Minimal CEM mirroring planning/cem.py math (per-instance sample/rollout/refit).
def cem_plan(wm, obj_fn, obs_0, obs_g, horizon, action_dim, device,
             num_samples=100, topk=10, opt_steps=10, var_scale=1.0, log=True):
    """Returns (mu, history, elite_actions).

    elite_actions: list of n tensors each shape (min(2,topk), horizon, action_dim)
    on CPU — the top-2 elite action sequences from the FINAL CEM iteration for
    each evaluation goal. Used for decoder visualization.
    """
    z_obs_g = wm.encode_obs({k: v.to(device) for k, v in obs_g.items()})
    n = obs_0["visual"].shape[0]
    mu = torch.zeros(n, horizon, action_dim, device=device)
    sigma = var_scale * torch.ones(n, horizon, action_dim, device=device)
    history = []
    elite_actions = [None] * n  # filled on the last iteration
    for it in range(opt_steps):
        step_losses = []
        is_last = (it == opt_steps - 1)
        for traj in range(n):
            o0 = {k: repeat(v[traj].unsqueeze(0), "1 ... -> s ...", s=num_samples).to(device)
                  for k, v in obs_0.items()}
            zg = {k: repeat(v[traj].unsqueeze(0), "1 ... -> s ...", s=num_samples)
                  for k, v in z_obs_g.items()}
            action = torch.randn(num_samples, horizon, action_dim, device=device) * sigma[traj] + mu[traj]
            action[0] = mu[traj]
            with torch.no_grad():
                z_obses, _ = wm.rollout(obs_0=o0, act=action)
                loss = obj_fn(z_obses, zg)             # (num_samples,)
            topk_idx = torch.argsort(loss)[:topk]
            elite = action[topk_idx]
            mu[traj] = elite.mean(0)
            sigma[traj] = elite.std(0)
            step_losses.append(loss[topk_idx[0]].item())
            if is_last:
                n_vis = min(2, topk)
                elite_actions[traj] = action[topk_idx[:n_vis]].detach().cpu()
        best = float(np.mean(step_losses))
        history.append(best)
        if log:
            print(f"  CEM step {it+1:02d}/{opt_steps} | best top-1 loss {best:.5f}")
    return mu, history, elite_actions


def synthetic_obs(n, num_hist, img_size, proprio_dim, device):
    return {
        "visual": torch.rand(n, num_hist, 3, img_size, img_size, device=device),
        "proprio": torch.randn(n, num_hist, proprio_dim, device=device),
    }


def save_elite_images(wm, decoder, obs_0, elite_actions, out_path, num_hist, device):
    """Decode top-2 CEM elite action sequences and save a visualization grid.

    Layout: one row per (eval goal × elite); columns = [num_hist real input
    frames] | [T VQVAE-decoded predicted frames].

    With random decoder weights the predicted frames are visual noise — that
    is the correct, expected output for the floor-control baseline. A trained
    decoder would show plausible future frames.

    Args:
        obs_0        : dict with "visual" (n, num_hist, 3, H, W) on device,
                       normalized to [-1, 1] by default_transform.
        elite_actions: list of n tensors, each (2, horizon, action_dim) on CPU.
        out_path     : PNG save path.
    """
    n = len(elite_actions)
    n_elites = elite_actions[0].shape[0]  # ≤ 2
    rows = n * n_elites

    all_rows = []
    with torch.no_grad():
        for i in range(n):
            for e in range(n_elites):
                act_e = elite_actions[i][e].unsqueeze(0).to(device)   # (1, horizon, A)
                o0 = {k: v[i:i+1] for k, v in obs_0.items()}          # (1, num_hist, ...)

                z_obses, _ = wm.rollout(obs_0=o0, act=act_e)
                z_vis = z_obses["visual"]                               # (1, T, 196, 384)

                decoded, _ = decoder(z_vis)                             # (T, 3, 224, 224)
                # min-max normalize per batch so noise is visible regardless of scale
                d_min, d_max = decoded.min(), decoded.max()
                decoded = ((decoded - d_min) / (d_max - d_min + 1e-6)).clamp(0, 1).cpu()

                # un-normalize input frames from [-1,1] to [0,1]
                inp = (obs_0["visual"][i].cpu() * 0.5 + 0.5).clamp(0, 1)  # (num_hist, 3, H, W)

                all_rows.append({
                    "label": f"G{i+1}/E{e+1}",
                    "input": inp,
                    "predicted": decoded,
                })

    T = all_rows[0]["predicted"].shape[0]
    ncols = num_hist + 1 + T    # +1 separator column between input and predicted

    fig, axes = plt.subplots(rows, ncols, figsize=(ncols * 1.8, rows * 1.8 + 0.6), dpi=80)
    if rows == 1:
        axes = axes[np.newaxis, :]

    for r, rd in enumerate(all_rows):
        # Input frames
        for t in range(num_hist):
            ax = axes[r, t]
            ax.imshow(rd["input"][t].permute(1, 2, 0).numpy())
            ax.axis("off")
            if r == 0:
                ax.set_title(f"in t={t}", fontsize=7)
        axes[r, 0].set_ylabel(rd["label"], fontsize=7, rotation=0,
                               labelpad=30, ha='right', va='center')

        # Separator column — blank
        axes[r, num_hist].set_visible(False)

        # Predicted (decoded) frames
        for t in range(T):
            ax = axes[r, num_hist + 1 + t]
            ax.imshow(rd["predicted"][t].permute(1, 2, 0).numpy())
            ax.axis("off")
            if r == 0:
                ax.set_title(f"pred t={t}", fontsize=7)

    fig.suptitle(
        "Top-2 CEM Elite Rollout Visualizations  "
        "(left: real input | right: VQVAE-decoded predicted states)\n"
        "NOTE: random decoder weights → predicted frames are noise "
        "(floor-control baseline; train decoder to see real predictions)",
        fontsize=8,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"[decoder] elite visualization saved → {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="lift")
    p.add_argument("--frameskip", type=int, default=5)
    p.add_argument("--num_hist", type=int, default=3)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--action_dim", type=int, default=35)   # 7-DoF * frameskip 5
    p.add_argument("--proprio_dim", type=int, default=9)
    p.add_argument("--enc_emb_dim", type=int, default=384) # fallback if encoder has no .emb_dim
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--num_samples", type=int, default=100)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--opt_steps", type=int, default=10)
    p.add_argument("--n_evals", type=int, default=3)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--smoke", action="store_true",
                   help="run on a tiny synthetic, no DINOv2/hydra, to verify the CEM math")
    p.add_argument("--decoder_path", type=str, default=None,
                   help="optional path to VQVAE checkpoint for elite visualization; "
                        "omit to use random weights (images will be noise)")
    a = p.parse_args()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[device] using {device}")
    obj_fn = create_objective_fn(alpha=a.alpha, base=1.0, mode="last")

    if a.smoke:
        # tiny stand-in world model: exercises the CEM loop + objective wiring
        # without DINOv2/hydra (those need the full env). Predictor = identity+noise.
        D = 8; P = 4
        class TinyWM:
            num_hist = a.num_hist; num_pred = 1
            def encode_obs(self, obs):
                b, t = obs["visual"].shape[:2]
                return {"visual": torch.randn(b, t, P, D), "proprio": torch.randn(b, t, 2)}
            def rollout(self, obs_0, act):
                b = obs_0["visual"].shape[0]; T = self.num_hist + act.shape[1]
                z = {"visual": torch.randn(b, T, P, D), "proprio": torch.randn(b, T, 2)}
                return z, None
        wm = TinyWM()
        obs_0 = {"visual": torch.rand(a.n_evals, a.num_hist, 3, 8, 8),
                 "proprio": torch.randn(a.n_evals, a.num_hist, 2)}
        obs_g = {"visual": torch.rand(a.n_evals, 1, 3, 8, 8),
                 "proprio": torch.randn(a.n_evals, 1, 2)}
        mu, hist, _ = cem_plan(wm, obj_fn, obs_0, obs_g, a.horizon, a.action_dim, device,
                               num_samples=20, topk=5, opt_steps=5)
        print(f"\n[smoke] planned actions {tuple(mu.shape)} | "
              f"loss start {hist[0]:.4f} -> end {hist[-1]:.4f}")
        print("[smoke] loop runs; with a RANDOM predictor loss does not "
              "meaningfully decrease (expected).")
        return

    # full run: real frozen DINOv2 + random-init ViT predictor + real Lift frames
    dataset_dir = os.environ.get("DATASET_DIR", "")
    if not dataset_dir:
        raise RuntimeError(
            "Set DATASET_DIR to the directory containing image224.hdf5 "
            "(e.g. export DATASET_DIR=/path/to/datasets/lift/ph)")
    data_path = os.path.join(dataset_dir, "image224.hdf5")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    wm = build_wm(a, device)
    obs_0, obs_g = load_real_obs(data_path, a.num_hist, a.n_evals, a.img_size, device)

    print(f"\nFLOOR CONTROL: frozen DINOv2 + RANDOM predictor, open-loop CEM "
          f"({a.n_evals} goals, {a.opt_steps} steps, {a.num_samples} samples)\n"
          f"  obs from mask/test split of {data_path}")
    mu, hist, elite_actions = cem_plan(
        wm, obj_fn, obs_0, obs_g, a.horizon, a.action_dim, device,
        num_samples=a.num_samples, topk=a.topk, opt_steps=a.opt_steps)
    print(f"\nFLOOR final latent-distance objective: {hist[-1]:.5f} "
          f"(start {hist[0]:.5f})")
    print("Interpretation: with an untrained predictor this number is the floor; "
          "a trained DINO-WM must drive it substantially lower to show it learned "
          "dynamics. Little/no decrease here is the expected, uninformative-failure "
          "baseline.")

    results_dir = os.path.join(_DINO_WM_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save JSON results
    out_json = os.path.join(results_dir, f"floor_{ts}.json")
    record = {
        "timestamp": ts,
        "device": str(device),
        "predictor": "random_init",
        "dataset": data_path,
        "config": {
            "env": a.env, "frameskip": a.frameskip, "num_hist": a.num_hist,
            "horizon": a.horizon, "action_dim": a.action_dim,
            "proprio_dim": a.proprio_dim, "n_evals": a.n_evals,
            "num_samples": a.num_samples, "topk": a.topk,
            "opt_steps": a.opt_steps, "alpha": a.alpha,
        },
        "loss_history": hist,
        "floor_start": hist[0],
        "floor_end": hist[-1],
        "floor_decrease_pct": 100.0 * (hist[0] - hist[-1]) / hist[0],
    }

    # Decode and visualize top-2 elites per goal
    decoder = build_decoder(a.decoder_path, device)
    out_png = os.path.join(results_dir, f"floor_{ts}_elites.png")
    save_elite_images(wm, decoder, obs_0, elite_actions, out_png, a.num_hist, device)
    record["elite_vis"] = out_png

    with open(out_json, "w") as f:
        json.dump(record, f, indent=2)
    print(f"\nResults saved to {out_json}")


if __name__ == "__main__":
    main()
