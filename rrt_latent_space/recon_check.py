"""
Reconstruction sanity check for the trained beta-VAE.

Pulls HELD-OUT frames (mask/<filter_key>, default 'test') from the robomimic
image HDF5 -- frames the VAE never trained on -- reconstructs them, and shows
original vs reconstruction side by side with per-image MS-SSIM. Also reports how
many latent dimensions are actually active (a low count => partial posterior
collapse), which contextualizes the upcoming interpolation test.

This is the endpoint-CONTROL validation: the interpolation test is only valid if
held-out endpoints reconstruct sharply. Run this first, in isolation.

Reads the model definition + MS-SSIM from train_betavae.py (same directory).
"""

import os
import argparse
import numpy as np
import h5py
import torch
import matplotlib.pyplot as plt

from train_betavae import BetaVAE, ms_ssim


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    latent = ck["cfg"]["latent"] if "cfg" in ck else 32
    model = BetaVAE(latent_dim=latent).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, latent


def held_out_demos(f, filter_key):
    if "mask" in f and filter_key in f["mask"]:
        names = [n.decode() if isinstance(n, bytes) else str(n)
                 for n in f["mask"][filter_key][:]]
        return names
    raise KeyError(f"mask/{filter_key} not found. "
                   f"masks present: {list(f['mask'].keys()) if 'mask' in f else 'none'}")


def gather_frames(f, demos, key, n_frames, rng):
    """Sample n_frames (demo, t) pairs spread across the held-out demos."""
    root = f["data"] if "data" in f else f
    pool = []
    for d in demos:
        og = root[d]["obs"] if "obs" in root[d] else root[d]
        if key in og:
            T = og[key].shape[0]
            pool.append((d, T))
    picks = []
    for _ in range(n_frames):
        d, T = pool[rng.integers(len(pool))]
        picks.append((d, int(rng.integers(T))))
    imgs = []
    for d, t in picks:
        og = root[d]["obs"] if "obs" in root[d] else root[d]
        imgs.append(np.asarray(og[key][t]))         # HWC uint8
    return picks, np.stack(imgs)


@torch.no_grad()
def reconstruct(model, imgs_uint8, device):
    x = torch.from_numpy(imgs_uint8.astype(np.float32) / 255.0)
    x = x.permute(0, 3, 1, 2).to(device)            # NCHW [0,1]
    mu, logvar = model.encode(x)
    x_hat = model.decode(mu)                          # use mean (deterministic)
    per = np.array([ms_ssim(x_hat[i:i+1], x[i:i+1]).item() for i in range(x.shape[0])])
    return x.cpu(), x_hat.cpu(), mu.cpu(), logvar.cpu(), per


@torch.no_grad()
def active_dims(model, f, demos, key, device, n=400, thresh=0.01):
    """Fraction/count of latent dims that carry information.
    Active if the variance of the posterior mean across data exceeds `thresh`,
    i.e. the dim is actually used rather than pinned to the prior."""
    rng = np.random.default_rng(0)
    _, imgs = gather_frames(f, demos, key, min(n, 400), rng)
    x = torch.from_numpy(imgs.astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(device)
    mu, logvar = model.encode(x)
    var_mu = mu.var(0).cpu().numpy()                 # variance of means per dim
    kl_per = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).mean(0).cpu().numpy()
    return var_mu, kl_per, int((var_mu > thresh).sum())


def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available()
                          else ("mps" if torch.backends.mps.is_available() else "cpu"))
    model, latent = load_model(cfg["ckpt"], device)
    rng = np.random.default_rng(cfg["seed"])
    with h5py.File(cfg["hdf5"], "r") as f:
        demos = held_out_demos(f, cfg["filter_key"])
        print(f"held-out demos ({cfg['filter_key']}): {len(demos)} | latent {latent} | device {device}")
        picks, imgs = gather_frames(f, demos, cfg["key"], cfg["n"], rng)
        x, x_hat, mu, logvar, per = reconstruct(model, imgs, device)
        var_mu, kl_per, n_active = active_dims(model, f, demos, cfg["key"], device)

    print(f"\nMS-SSIM on held-out frames: mean {per.mean():.4f} | "
          f"min {per.min():.4f} | max {per.max():.4f}")
    print(f"active latent dims: {n_active}/{latent} "
          f"(var(mu)>0.01)   [low => partial posterior collapse]")
    print(f"per-frame MS-SSIM: {np.round(per, 3)}")

    # side-by-side panel: originals (top) vs reconstructions (bottom)
    N = x.shape[0]
    fig, axes = plt.subplots(2, N, figsize=(2.0*N, 4.4))
    for i in range(N):
        axes[0, i].imshow(x[i].permute(1, 2, 0).numpy())
        axes[0, i].set_title(f"{picks[i][0]} t{picks[i][1]}", fontsize=7)
        axes[1, i].imshow(x_hat[i].permute(1, 2, 0).numpy())
        axes[1, i].set_title(f"MS-SSIM {per[i]:.3f}", fontsize=7)
        for r in (0, 1):
            axes[r, i].set_xticks([]); axes[r, i].set_yticks([])
    axes[0, 0].set_ylabel("original", fontsize=10)
    axes[1, 0].set_ylabel("reconstruction", fontsize=10)
    fig.suptitle(f"Held-out reconstruction sanity check  "
                 f"(mean MS-SSIM {per.mean():.3f}, {n_active}/{latent} active dims)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(cfg["out"], dpi=130)
    print(f"\nsaved {cfg['out']}")

    # per-dim KL bar (which dims are used)
    fig2, ax = plt.subplots(figsize=(8, 3))
    order = np.argsort(kl_per)[::-1]
    ax.bar(range(latent), kl_per[order], color="#36506e")
    ax.axhline(0.01, color="#d62828", lw=1, ls="--", label="active threshold (KL ~ 0.01)")
    ax.set_xlabel("latent dim (sorted by KL)"); ax.set_ylabel("KL per dim")
    ax.set_title("per-dimension KL — how many dims carry information")
    ax.legend(fontsize=8); fig2.tight_layout()
    fig2.savefig(cfg["out"].replace(".png", "_kl.png"), dpi=130)
    print(f"saved {cfg['out'].replace('.png', '_kl.png')}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="/Users/swapnilmallick/Desktop/ROBOSUITE_RRT/rrt_latent_space/ckpt/betavae_last.pt")
    p.add_argument("--hdf5", default="/Users/swapnilmallick/Desktop/ROBOSUITE_RRT/datasets/lift/ph/image.hdf5")
    p.add_argument("--key", default="agentview_image")
    p.add_argument("--filter_key", default="valid")
    p.add_argument("--n", type=int, default=8, help="number of held-out frames to show")
    p.add_argument("--out", default="/Users/swapnilmallick/Desktop/ROBOSUITE_RRT/rrt_latent_space/recon_check.png")
    p.add_argument("--seed", type=int, default=0)
    main(vars(p.parse_args()))
