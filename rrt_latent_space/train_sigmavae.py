"""
sigma-VAE for 84x84x3 robomimic images: a calibrated-decoder VAE that learns
the decoder variance analytically and self-balances the ELBO, removing the
need to tune beta. (Rybkin, Daniilidis, Levine, "Simple and Effective VAE
Training with Calibrated Decoders", ICML 2021; optimal-sigma variant.)

Loss = sum_pixels  NLL(x | x_hat, sigma*)   +   beta * sum_dims KL
        with sigma* = analytic batch-wise MLE of the decoder scale,
        and beta = 1 (annealed 0->1 only as a short warm-up).

Why this replaces the MS-SSIM + L1 + mean-KL setup:
  * The previous objective MEAN-reduced both the reconstruction and the KL.
    For a VAE that destroys the natural ~(84*84*3):(latent_dim) magnitude ratio
    that balances the ELBO, leaving the KL massively over-weighted (beta=1 there
    behaved like beta~=660). That over-pressure -> haze + posterior collapse.
  * The calibrated decoder instead SUMS the reconstruction NLL over pixels and
    the KL over dims, and rescales the reconstruction by 1/sigma*^2. When recon
    is good sigma* shrinks and the reconstruction gradient is amplified (sharper);
    when recon is poor sigma* grows and the KL is allowed to shape the latent.
    The balance is set analytically per batch -> no beta search.
  * MS-SSIM is NOT a log-likelihood, so it can't host a calibrated variance. It
    is kept here only as a MONITORING metric (logged, not optimized) so we can
    still watch multi-scale structure during training.

Collapse guard (kept because of prior collapse history): free-bits floors each
latent dim's KL at `--free_bits` nats so dimensions can't be zeroed out early.

decoder_dist: "gaussian" (optimal-sigma, matches the paper) or "laplace"
(optimal-scale = mean abs error; the calibrated analog of your old L1, sharper
edges, heavier-tailed). Output activation is Sigmoid and inputs are [0,1], so
the scale is estimated in the same space - no range mismatch.

84x84 caveat (MS-SSIM monitor only): 5-scale MS-SSIM with an 11x11 window
underflows (84->...->5 < 11). We use a 7x7 window and only as many scales as
fit, renormalizing the scale weights.

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


# --------------------------------------------------------------------------- #
# Calibrated decoder (sigma-VAE) pieces
# --------------------------------------------------------------------------- #
def softclip(tensor, min_val):
    """Soft lower-bound on log-scale; keeps learning stable as sigma -> 0.
    (From 'Handful of Trials' / the sigma-VAE reference implementation.)"""
    return min_val + F.softplus(tensor - min_val)


def gaussian_nll(x, x_hat, log_sigma):
    # per-element Gaussian NLL with scalar (broadcast) log_sigma
    return 0.5 * ((x - x_hat) / log_sigma.exp()) ** 2 + log_sigma + 0.5 * math.log(2 * math.pi)


def laplace_nll(x, x_hat, log_b):
    # per-element Laplace NLL; calibrated analog of L1
    return (x - x_hat).abs() / log_b.exp() + log_b + math.log(2.0)


def calibrated_recon(x, x_hat, dist="gaussian", scale_min=-6.0):
    """Optimal-(sigma|b) calibrated decoder: analytic batch-wise MLE of the
    decoder scale, then a SUMMED NLL (sum over pixels, mean over batch).

    Returns (rec, scale) where rec is the per-example summed NLL averaged over
    the batch, and scale is the current decoder scale (for logging)."""
    if dist == "gaussian":
        # sigma* = RMS reconstruction error over the whole batch
        log_scale = ((x - x_hat) ** 2).mean([0, 1, 2, 3], keepdim=True).sqrt().log()
        log_scale = softclip(log_scale, scale_min)
        nll = gaussian_nll(x, x_hat, log_scale)
    elif dist == "laplace":
        # b* = mean absolute error over the whole batch
        log_scale = (x - x_hat).abs().mean([0, 1, 2, 3], keepdim=True).log()
        log_scale = softclip(log_scale, scale_min)
        nll = laplace_nll(x, x_hat, log_scale)
    else:
        raise ValueError(f"unknown decoder_dist {dist!r}")
    rec = nll.sum(dim=[1, 2, 3]).mean()         # SUM over pixels, mean over batch
    return rec, log_scale.exp().item()


def kl_summed(mu, logvar, free_bits=0.0):
    """KL summed over latent dims, mean over batch (matches the recon reduction).
    With free_bits > 0, each latent dim's (batch-averaged) KL is floored at
    `free_bits` nats so dims can't be driven to zero -> guards posterior collapse.
    sum_d mean_b == mean_b sum_d, so the two branches are on the same scale."""
    kld = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())     # (B, D)
    if free_bits > 0.0:
        kld_dim = kld.mean(dim=0)                            # (D,)
        return torch.clamp(kld_dim, min=free_bits).sum()
    return kld.sum(dim=1).mean()


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

    beta_tag = f"sigma_{cfg['beta']}"
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
              f"decoder {cfg['decoder_dist']} (optimal-scale) | beta->{cfg['beta']} "
              f"warmup {cfg['anneal']} ep | free_bits {cfg['free_bits']}")
    log(header)

    best_loss = float("inf")
    best_ckpt = os.path.join(cfg["ckpt_dir"], f"betavae_{cfg['beta']}.pt")

    for ep in range(cfg["epochs"]):
        model.train()
        beta = beta_at(ep, cfg["beta"], cfg["anneal"])
        agg = {"loss": 0, "rec": 0, "kl": 0, "msssim": 0, "sigma": 0, "nb": 0}
        last_x = last_xhat = None
        for x in dl:
            x = x.to(dev)
            x_hat, mu, logvar = model(x)
            rec, scale = calibrated_recon(x, x_hat, dist=cfg["decoder_dist"])
            kl = kl_summed(mu, logvar, free_bits=cfg["free_bits"])
            loss = rec + beta * kl          # beta targets 1.0; calibration does the balancing
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                msssim_val = ms_ssim(x_hat, x).item()   # monitored, NOT optimized
            for k, v in (("loss", loss.item()), ("rec", rec.item()), ("kl", kl.item()),
                         ("msssim", msssim_val), ("sigma", scale), ("nb", 1)):
                agg[k] += v
            last_x, last_xhat = x, x_hat
        nb = max(1, agg["nb"])
        ep_loss = agg["loss"] / nb
        log(f"ep {ep:03d} | beta {beta:.3f} | loss {ep_loss:.1f} | "
            f"rec {agg['rec']/nb:.1f} | KL {agg['kl']/nb:.2f} | "
            f"MS-SSIM {agg['msssim']/nb:.4f} | sigma {agg['sigma']/nb:.4f}")

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
    _HERE = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=os.path.join(_HERE, "vae_images.hdf5"))
    p.add_argument("--key", default="agentview_image")
    p.add_argument("--ckpt_dir", default=os.path.join(_HERE, "ckpt"))
    p.add_argument("--latent", type=int, default=32)
    p.add_argument("--beta", type=float, default=1.0)       # target; keep at 1.0 (calibration balances)
    p.add_argument("--anneal", type=int, default=10)        # epochs to warm beta 0->target
    p.add_argument("--decoder_dist", default="gaussian",
                   choices=["gaussian", "laplace"],
                   help="optimal-sigma Gaussian (paper) or optimal-scale Laplace (calibrated L1)")
    p.add_argument("--free_bits", type=float, default=0.5,
                   help="per-dim KL floor in nats; collapse guard (0 disables)")
    p.add_argument("--alpha", type=float, default=0.85,     # UNUSED (legacy MS-SSIM/L1 weight)
                   help="deprecated; MS-SSIM is now a monitor only")
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
        rec, scale = calibrated_recon(x, x_hat, dist=a.decoder_dist)
        kl = kl_summed(mu, lv, free_bits=a.free_bits)
        (rec + kl).backward()
        gnorm = sum(p.grad.norm().item() for p in m.parameters() if p.grad is not None)
        msssim_val = ms_ssim(x_hat, x).item()
        print(f"[smoke] x_hat {tuple(x_hat.shape)} | rec(sum) {rec.item():.1f} "
              f"| KL(sum) {kl.item():.2f} | sigma {scale:.4f} "
              f"| MS-SSIM(monitor) {msssim_val:.4f} | grad-norm {gnorm:.3f}")
        z = m.encode(x)[0]; xd = m.decode(z)
        print(f"[smoke] encode->z {tuple(z.shape)} | decode->{tuple(xd.shape)}  OK")
    else:
        train(cfg)
