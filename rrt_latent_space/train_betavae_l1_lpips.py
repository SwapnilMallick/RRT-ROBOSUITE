"""
Beta-VAE for 84x84x3 robomimic images, trained with an L1 + lambda*LPIPS
reconstruction loss and KL annealing.

Loss = L1(x, x_hat) + lambda_lpips * LPIPS(x, x_hat) + beta(t) * KL

Rationale:
  * L1 anchors absolute color/brightness.
  * LPIPS (perceptual loss) captures high-level structural similarity using
    deep features — sharper, less blurry than pixel-level MS-SSIM for
    textures and edges.
  * LPIPS expects inputs in [-1, 1]; passing normalize=True in the forward
    call lets the library rescale from [0, 1] internally, so we never need
    to remap tensors manually.
  * lambda_lpips in [0.1, 1] balances perceptual vs. pixel fidelity.
  * KL is annealed from ~0 so the decoder learns to reconstruct before prior
    pressure can collapse the posterior.

Exposes encode(x)->z (mean) and decode(z)->x_hat for the interpolation harness.
"""

import os
import argparse
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    import lpips
except ImportError as e:
    raise ImportError("lpips not found — install with: pip install lpips") from e


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


def recon_loss(x, x_hat, lambda_lpips, lpips_fn):
    """
    Returns (total_recon_tensor, l1_float, lpips_float).

    x and x_hat are in [0, 1].  normalize=True tells the lpips library to
    rescale [0, 1] -> [-1, 1] before computing perceptual distance, so we
    never need to remap tensors manually.
    """
    l1 = F.l1_loss(x_hat, x)
    # lpips returns (B,1,1,1); mean -> scalar. normalize=True handles [0,1] input.
    lp = lpips_fn(x_hat, x, normalize=True).mean()
    total = l1 + lambda_lpips * lp
    return total, l1.item(), lp.item()


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
    fig.suptitle(f"Beta-VAE L1+LPIPS  |  beta={beta:.4f}  |  epoch={epoch:03d}", fontsize=10)
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
    pin = torch.cuda.is_available()
    dl = DataLoader(ds, batch_size=cfg["batch"], shuffle=True,
                    num_workers=cfg["workers"], drop_last=True, pin_memory=pin)
    model = BetaVAE(latent_dim=cfg["latent"]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    os.makedirs(cfg["ckpt_dir"], exist_ok=True)

    # LPIPS network (frozen AlexNet by default; weights downloaded on first use)
    lpips_fn = lpips.LPIPS(net=cfg["lpips_net"]).to(dev)
    lpips_fn.eval()
    for p in lpips_fn.parameters():
        p.requires_grad_(False)

    beta_tag = f"beta_{cfg['beta']}_lam_{cfg['lambda_lpips']}"
    run_ts  = datetime.now().strftime("%Y%m%d_%H%M%S")

    base_dir = os.path.dirname(os.path.abspath(cfg["ckpt_dir"]))
    img_dir = cfg.get("img_dir") or os.path.join(base_dir, "images", beta_tag)

    log_path = os.path.join(cfg["ckpt_dir"], f"train_log_{beta_tag}_{run_ts}.txt")
    log_file = open(log_path, "a")

    def log(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    header = (f"device {dev} | frames {len(ds)} | latent {cfg['latent']} | "
              f"beta->{cfg['beta']} over {cfg['anneal']} ep | "
              f"lambda_lpips {cfg['lambda_lpips']} | lpips_net {cfg['lpips_net']}")
    log(header)
    log("epoch | beta   | total   | recon   | L1      | lam*LPIPS | beta*KL")

    best_loss = float("inf")
    best_ckpt = os.path.join(cfg["ckpt_dir"], f"betavae_l1lpips_{cfg['beta']}.pt")

    for ep in range(cfg["epochs"]):
        model.train()
        beta = beta_at(ep, cfg["beta"], cfg["anneal"])
        agg = {"loss": 0, "recon": 0, "l1": 0, "lpips": 0, "kl": 0, "nb": 0}
        last_x = last_xhat = None

        for x in dl:
            x = x.to(dev)
            x_hat, mu, logvar = model(x)
            rec, l1_val, lpips_val = recon_loss(x, x_hat, cfg["lambda_lpips"], lpips_fn)
            kl = kl_per_dim(mu, logvar)
            loss = rec + beta * kl
            opt.zero_grad(); loss.backward(); opt.step()
            agg["loss"]  += loss.item()
            agg["recon"] += rec.item()
            agg["l1"]    += l1_val
            agg["lpips"] += lpips_val
            agg["kl"]    += kl.item()
            agg["nb"]    += 1
            last_x, last_xhat = x, x_hat

        nb = max(1, agg["nb"])
        ep_loss  = agg["loss"]  / nb
        ep_recon = agg["recon"] / nb
        ep_l1    = agg["l1"]    / nb
        ep_lp    = agg["lpips"] / nb
        ep_kl    = agg["kl"]    / nb
        # log the actual weighted contributions so you can see relative magnitudes
        log(
            f"ep {ep:03d} | beta {beta:.3f} | "
            f"total {ep_loss:.4f} | "
            f"recon {ep_recon:.4f} | "
            f"L1 {ep_l1:.4f} | "
            f"lam*LPIPS {cfg['lambda_lpips'] * ep_lp:.4f} | "
            f"beta*KL {beta * ep_kl:.4f}"
        )

        torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": ep},
                   os.path.join(cfg["ckpt_dir"], "betavae_l1lpips_last.pt"))

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
    p.add_argument("--beta", type=float, default=1.0)           # annealed target
    p.add_argument("--anneal", type=int, default=10)            # epochs to ramp beta
    p.add_argument("--lambda_lpips", type=float, default=0.5,   # perceptual weight [0.1, 1]
                   help="weight on LPIPS term (0.1=light perceptual, 1.0=heavy)")
    p.add_argument("--lpips_net", default="alex",
                   choices=["alex", "vgg", "squeeze"],
                   help="backbone for LPIPS (alex=fastest, vgg=highest quality)")
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
        lpips_fn = lpips.LPIPS(net=a.lpips_net)
        lpips_fn.eval()
        for param in lpips_fn.parameters():
            param.requires_grad_(False)
        m = BetaVAE(latent_dim=a.latent)
        x = torch.rand(4, 3, 84, 84)
        x_hat, mu, lv = m(x)
        assert x_hat.shape == x.shape, x_hat.shape
        rec, l1_val, lpips_val = recon_loss(x, x_hat, a.lambda_lpips, lpips_fn)
        kl = kl_per_dim(mu, lv)
        loss = rec + a.beta * kl
        loss.backward()
        gnorm = sum(param.grad.norm().item() for param in m.parameters() if param.grad is not None)
        print(
            f"[smoke] x_hat {tuple(x_hat.shape)} | "
            f"recon {rec.item():.4f} (L1 {l1_val:.4f}, "
            f"lam*LPIPS {a.lambda_lpips * lpips_val:.4f}) | "
            f"KL/dim {kl.item():.4f} | beta*KL {a.beta * kl.item():.4f} | "
            f"grad-norm {gnorm:.3f}"
        )
        z = m.encode(x)[0]; xd = m.decode(z)
        print(f"[smoke] encode->z {tuple(z.shape)} | decode->{tuple(xd.shape)}  OK")
    else:
        train(cfg)
