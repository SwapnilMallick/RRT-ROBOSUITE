"""
Beta-VAE for 84x84x3 robomimic images, trained with an MS-SSIM + L1
reconstruction loss and KL annealing.

Loss = alpha * (1 - MS_SSIM(x, x_hat)) + (1 - alpha) * L1(x, x_hat) + beta(t) * KL

Rationale:
  * MS-SSIM scores multi-scale structure (edges of arm/cube) instead of the
    per-pixel mean MSE collapses toward -> crisper reconstructions, which the
    downstream interpolation test needs as its endpoint control.
  * L1 anchors absolute color/brightness (MS-SSIM is partly luminance-invariant).
  * Replacing the recon NLL with a structural score leaves the strict ELBO, so
    beta is treated as a weight on a comparable-scale KL (per-dim, per-pixel
    normalized) and is ANNEALED from ~0 so the decoder learns to reconstruct
    before prior pressure can collapse the posterior.

84x84 caveat: 5-scale MS-SSIM with an 11x11 window underflows (84->...->5 < 11).
We use a 7x7 window and only as many scales as fit the image, renormalizing the
scale weights. A forward+backward smoke test at the bottom verifies it runs.

Exposes encode(x)->z (mean) and decode(z)->x_hat for the interpolation harness.
"""

import os
import math
import argparse
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# --------------------------------------------------------------------------- #
# Differentiable MS-SSIM (windowed, Gaussian) for images in [0, 1]
# --------------------------------------------------------------------------- #
def _gaussian_window(win_size, sigma, channels, device, dtype):
    coords = torch.arange(win_size, dtype=dtype, device=device) - win_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).unsqueeze(0)
    win2d = (g.t() @ g).unsqueeze(0).unsqueeze(0)          # (1,1,w,w)
    return win2d.expand(channels, 1, win_size, win_size).contiguous()


def _ssim_map(x, y, win, C1, C2):
    ch = x.shape[1]
    pad = win.shape[-1] // 2
    mu_x = F.conv2d(x, win, padding=pad, groups=ch)
    mu_y = F.conv2d(y, win, padding=pad, groups=ch)
    mu_x2, mu_y2, mu_xy = mu_x*mu_x, mu_y*mu_y, mu_x*mu_y
    sig_x = F.conv2d(x*x, win, padding=pad, groups=ch) - mu_x2
    sig_y = F.conv2d(y*y, win, padding=pad, groups=ch) - mu_y2
    sig_xy = F.conv2d(x*y, win, padding=pad, groups=ch) - mu_xy
    cs = (2*sig_xy + C2) / (sig_x + sig_y + C2)             # contrast-structure
    ssim = ((2*mu_xy + C1) / (mu_x2 + mu_y2 + C1)) * cs
    return ssim.mean([1, 2, 3]), cs.mean([1, 2, 3])         # per-image


def ms_ssim(x, y, win_size=7, sigma=1.5, data_range=1.0,
            weights=(0.0448, 0.2856, 0.3001, 0.2363, 0.1333)):
    """Multi-scale SSIM in [0,1]; auto-limits scales so the window always fits."""
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    device, dtype = x.device, x.dtype
    win = _gaussian_window(win_size, sigma, x.shape[1], device, dtype)
    # how many scales fit before the image is smaller than the window?
    min_side = min(x.shape[-2], x.shape[-1])
    max_scales = int(math.floor(math.log2(min_side / win_size))) + 1
    n = max(1, min(len(weights), max_scales))
    w = torch.tensor(weights[:n], device=device, dtype=dtype)
    w = w / w.sum()
    mcs = []
    for i in range(n):
        ssim_i, cs_i = _ssim_map(x, y, win, C1, C2)
        if i < n - 1:
            mcs.append(torch.relu(cs_i))
            x = F.avg_pool2d(x, 2)
            y = F.avg_pool2d(y, 2)
    ssim_last = torch.relu(ssim_i)
    out = ssim_last ** w[-1]
    for i in range(n - 1):
        out = out * (mcs[i] ** w[i])
    return out.mean()                                       # scalar in [0,1]


# --------------------------------------------------------------------------- #
# Beta-VAE
# --------------------------------------------------------------------------- #
class BetaVAE(nn.Module):
    def __init__(self, latent_dim=32, ch=3):
        super().__init__()
        self.latent_dim = latent_dim
        # encoder: 84 -> 42 -> 21 -> 10 -> 5  (stride-2, kernel-4, padding-1)
        self.enc = nn.Sequential(
            nn.Conv2d(ch, 32, 4, 2, 1), nn.ReLU(True),      # 84 -> 42
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(True),      # 42 -> 21
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(True),     # 21 -> 10 (floor)
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(True),    # 10 -> 5
        )
        self.flat = 256 * 5 * 5
        self.fc_mu = nn.Linear(self.flat, latent_dim)
        self.fc_lv = nn.Linear(self.flat, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, self.flat)
        # decoder mirrors encoder; ConvT out = 2*in + output_padding (s2,p1,k4)
        #   5 ->10 : out_pad=0 ;  10->21 : out_pad=1 ;
        #  21->42 : out_pad=0 ;  42->84 : out_pad=0
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1, output_padding=0), nn.ReLU(True),  # 5 -> 10
            nn.ConvTranspose2d(128, 64, 4, 2, 1, output_padding=1), nn.ReLU(True),   # 10 -> 21
            nn.ConvTranspose2d(64, 32, 4, 2, 1, output_padding=0), nn.ReLU(True),    # 21 -> 42
            nn.ConvTranspose2d(32, ch, 4, 2, 1, output_padding=0), nn.Sigmoid(),     # 42 -> 84
        )

    def encode(self, x):
        h = self.enc(x).flatten(1)
        return self.fc_mu(h), self.fc_lv(h)

    def reparam(self, mu, logvar):
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        h = self.fc_dec(z).view(-1, 256, 5, 5)
        return self.dec(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)
        return self.decode(z), mu, logvar


def kl_per_dim(mu, logvar):
    # mean over batch AND latent dims -> scale-stable, comparable to per-pixel recon
    return (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).mean()


def recon_loss(x, x_hat, alpha):
    l1 = F.l1_loss(x_hat, x)
    dssim = 1.0 - ms_ssim(x_hat, x)
    return alpha * dssim + (1 - alpha) * l1, dssim.item(), l1.item()


# --------------------------------------------------------------------------- #
# Data: read vae_images.hdf5 (N,84,84,3) uint8 -> float [0,1], CHW
# --------------------------------------------------------------------------- #
class H5Images(Dataset):
    def __init__(self, path, key="agentview_image"):
        self.path, self.key = path, key
        with h5py.File(path, "r") as f:
            self.n = f[key].shape[0]
        self._f = None
    def __len__(self):
        return self.n
    def __getitem__(self, i):
        if self._f is None:                       # lazy open (DataLoader workers)
            self._f = h5py.File(self.path, "r")
        img = self._f[self.key][i].astype(np.float32) / 255.0   # HWC [0,1]
        return torch.from_numpy(img).permute(2, 0, 1)           # CHW


def beta_at(epoch, beta_max, anneal_epochs):
    return beta_max * min(1.0, (epoch + 1) / max(1, anneal_epochs))


def save_recon_grid(x, x_hat, beta, epoch, img_dir):
    os.makedirs(img_dir, exist_ok=True)
    n = min(8, x.shape[0])
    x_np  = x[:n].cpu().detach().permute(0, 2, 3, 1).numpy()
    xh_np = x_hat[:n].cpu().detach().permute(0, 2, 3, 1).numpy()
    fig, axes = plt.subplots(2, n, figsize=(n * 1.5, 3.5))
    fig.suptitle(f"Beta-VAE  |  beta={beta:.4f}  |  epoch={epoch:03d}", fontsize=10)
    for i in range(n):
        axes[0, i].imshow(np.clip(x_np[i],  0, 1)); axes[0, i].axis("off")
        axes[1, i].imshow(np.clip(xh_np[i], 0, 1)); axes[1, i].axis("off")
    axes[0, 0].set_title("orig",  fontsize=7)
    axes[1, 0].set_title("recon", fontsize=7)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, f"recon_ep{epoch:03d}_{ts}.png"), dpi=100)
    plt.close(fig)


def train(cfg):
    dev = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    ds = H5Images(cfg["data"], key=cfg["key"])
    pin = torch.cuda.is_available()   # pin_memory only works on CUDA
    dl = DataLoader(ds, batch_size=cfg["batch"], shuffle=True,
                    num_workers=cfg["workers"], drop_last=True, pin_memory=pin)
    model = BetaVAE(latent_dim=cfg["latent"]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    os.makedirs(cfg["ckpt_dir"], exist_ok=True)

    beta_tag = f"beta_{cfg['beta']}"
    run_ts  = datetime.now().strftime("%Y%m%d_%H%M%S")

    # per-beta image subdirectory
    base_dir = os.path.dirname(os.path.abspath(cfg["ckpt_dir"]))
    img_dir = cfg.get("img_dir") or os.path.join(base_dir, "images", beta_tag)

    # timestamped log file so runs never overwrite each other
    log_path = os.path.join(cfg["ckpt_dir"], f"train_log_{beta_tag}_{run_ts}.txt")
    log_file = open(log_path, "a")

    def log(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    header = (f"device {dev} | frames {len(ds)} | latent {cfg['latent']} | "
              f"beta->{cfg['beta']} over {cfg['anneal']} ep | alpha {cfg['alpha']}")
    log(header)

    best_loss = float("inf")
    best_ckpt = os.path.join(cfg["ckpt_dir"], f"betavae_{cfg['beta']}.pt")

    for ep in range(cfg["epochs"]):
        model.train()
        beta = beta_at(ep, cfg["beta"], cfg["anneal"])
        agg = {"loss": 0, "dssim": 0, "l1": 0, "kl": 0, "nb": 0}
        last_x = last_xhat = None
        for x in dl:
            x = x.to(dev)
            x_hat, mu, logvar = model(x)
            rec, dssim, l1 = recon_loss(x, x_hat, cfg["alpha"])
            kl = kl_per_dim(mu, logvar)
            loss = rec + beta * kl
            opt.zero_grad(); loss.backward(); opt.step()
            for k, v in (("loss", loss.item()), ("dssim", dssim), ("l1", l1),
                         ("kl", kl.item()), ("nb", 1)):
                agg[k] += v
            last_x, last_xhat = x, x_hat
        nb = max(1, agg["nb"])
        ep_loss = agg["loss"] / nb
        log(f"ep {ep:03d} | beta {beta:.3f} | loss {ep_loss:.4f} | "
            f"1-msssim {agg['dssim']/nb:.4f} | L1 {agg['l1']/nb:.4f} | "
            f"KL/dim {agg['kl']/nb:.4f}")

        # rolling last checkpoint
        torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": ep},
                   os.path.join(cfg["ckpt_dir"], "betavae_last.pt"))

        # best checkpoint per beta — never overwritten by a different beta run
        if ep_loss < best_loss:
            best_loss = ep_loss
            torch.save({"model": model.state_dict(), "cfg": cfg,
                        "epoch": ep, "loss": best_loss}, best_ckpt)
            log(f"  -> new best ({best_loss:.4f}), saved {best_ckpt}")

        if last_x is not None:
            model.eval()
            with torch.no_grad():
                save_recon_grid(last_x, last_xhat, beta, ep, img_dir)
            model.train()

    log(f"done. best={best_loss:.4f} @ {best_ckpt}")
    log_file.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="/Users/swapnilmallick/Desktop/ROBOSUITE_RRT/rrt_latent_space/vae_images.hdf5")
    p.add_argument("--key", default="agentview_image")
    p.add_argument("--ckpt_dir", default="/Users/swapnilmallick/Desktop/ROBOSUITE_RRT/rrt_latent_space/ckpt")
    p.add_argument("--latent", type=int, default=32)
    p.add_argument("--beta", type=float, default=1.0)       # annealed target
    p.add_argument("--anneal", type=int, default=10)        # epochs to ramp beta
    p.add_argument("--alpha", type=float, default=0.85)     # MS-SSIM vs L1 weight
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--img_dir", default=None,
                   help="where to save recon grids (default: <ckpt_dir>/../images/)")
    p.add_argument("--smoke_test", action="store_true",
                   help="run a forward+backward pass on synthetic data and exit")
    a = p.parse_args()
    cfg = vars(a)
    if a.smoke_test:
        dev = "cpu"
        m = BetaVAE(latent_dim=a.latent)
        x = torch.rand(4, 3, 84, 84)
        x_hat, mu, lv = m(x)
        assert x_hat.shape == x.shape, x_hat.shape
        rec, dssim, l1 = recon_loss(x, x_hat, a.alpha)
        kl = kl_per_dim(mu, lv)
        (rec + kl).backward()
        gnorm = sum(p.grad.norm().item() for p in m.parameters() if p.grad is not None)
        print(f"[smoke] x_hat {tuple(x_hat.shape)} | recon {rec.item():.4f} "
              f"(1-msssim {dssim:.4f}, L1 {l1:.4f}) | KL/dim {kl.item():.4f} "
              f"| grad-norm {gnorm:.3f}")
        z = m.encode(x)[0]; xd = m.decode(z)
        print(f"[smoke] encode->z {tuple(z.shape)} | decode->{tuple(xd.shape)}  OK")
    else:
        train(cfg)
