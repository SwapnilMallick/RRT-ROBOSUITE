"""
Train VQVAE decoder: DINOv2 patch tokens (196 × 384) → RGB (3 × 224 × 224).

DINOv2 encoder is frozen. Only VQVAE decoder weights are updated.
Loss: MSE(VQVAE_decoder(DINOv2(img)), img) — matches VWorldModel.decoder_criterion.

Speed: DINOv2 features are pre-computed ONCE and cached to disk (HDF5, float16).
Training then reads (feature, image) pairs from the cache — no DINOv2 forward during
training. This is ~20-35x faster per epoch than running DINOv2 every batch.

Run from dino_wm_repo/:
  DATASET_DIR=/path/to/datasets/lift/ph python train_decoder.py

First run pre-computes features (one-time cost, ~15 min on MPS), then trains.
The cache is reused on subsequent runs automatically.

Best checkpoint → <ckpt_dir>/decoder_best.pt
Pass to floor_control_plan.py via --decoder_path <ckpt_dir>/decoder_best.pt
"""

import argparse
import os
import sys
import time
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

_DINO_WM_ROOT = os.environ.get(
    "DINO_WM_ROOT", os.path.dirname(os.path.abspath(__file__))
)
if _DINO_WM_ROOT not in sys.path:
    sys.path.insert(0, _DINO_WM_ROOT)

from models.vqvae import VQVAE


# --------------------------------------------------------------------------- #
# Raw frame dataset — used only during pre-computation, not during training.
class LiftFrameDataset(Dataset):
    """Flat frame-level view of the Lift HDF5.  Returns (C,224,224) in [-1,1]."""

    _transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    def __init__(self, data_path, filter_key, image_key="agentview_image"):
        self.data_path = data_path
        self.image_key = image_key
        self._h5 = None

        with h5py.File(data_path, "r") as f:
            mask = f["mask"][filter_key][:]
            demo_names = sorted(
                [m.decode() if isinstance(m, bytes) else m for m in mask],
                key=lambda s: int(s.split("_")[-1]),
            )
            self.index = []
            for d in demo_names:
                T = f["data"][d]["obs"][image_key].shape[0]
                for t in range(T):
                    self.index.append((d, t))

        print(f"[LiftFrameDataset] {filter_key}: {len(demo_names)} demos, "
              f"{len(self.index)} frames")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        if self._h5 is None:
            self._h5 = h5py.File(self.data_path, "r")
        dname, t = self.index[idx]
        img = self._h5["data"][dname]["obs"][self.image_key][t]  # (H,W,C) uint8
        img = torch.from_numpy(img.copy()).float() / 255.0        # (H,W,C) [0,1]
        img = img.permute(2, 0, 1)                                # (C,H,W)
        return self._transform(img)                               # (C,224,224) [-1,1]


# --------------------------------------------------------------------------- #
# Cache dataset — reads pre-computed (feature, image) pairs. Used during training.
class FeatureCacheDataset(Dataset):
    """Loads the entire feature cache into RAM at init for O(1) random access.

    The cache is ~4.4 GB float16 total; loading once (~30-60s) eliminates all
    disk I/O during training. __getitem__ is a pure RAM read after that.
    num_workers=0 is correct here — no benefit from spawning processes when
    data is already in memory, and avoids macOS process-spawn overhead.
    """

    def __init__(self, cache_path, split):
        print(f"[FeatureCacheDataset] loading {split} into RAM ...",
              flush=True, end=" ")
        t0 = time.time()
        with h5py.File(cache_path, "r") as f:
            # Store as float16 torch tensors. __getitem__ slices are zero-copy views;
            # .float() converts the small per-sample slice (not the whole array).
            self.features = torch.from_numpy(f[split]["features"][:])  # (N,196,384) f16
            self.images   = torch.from_numpy(f[split]["images"][:])    # (N,3,224,224) f16
        mem_gb = (self.features.nbytes + self.images.nbytes) / 1e9
        print(f"{len(self.features)} frames  [{time.time()-t0:.0f}s]  "
              f"({mem_gb:.1f} GB RAM)")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Tensor slicing is a zero-copy view; .float() converts only the single sample.
        return self.features[idx].float(), self.images[idx].float()


# --------------------------------------------------------------------------- #
def build_encoder(device):
    """Frozen DINOv2 from conf files — identical setup to floor_control_plan.py."""
    import hydra.utils
    from omegaconf import OmegaConf

    _conf_dir = os.path.join(_DINO_WM_ROOT, "conf")
    enc_cfg   = OmegaConf.load(os.path.join(_conf_dir, "encoder", "dino.yaml"))
    train_cfg = OmegaConf.load(os.path.join(_conf_dir, "train.yaml"))
    img_size  = train_cfg.get("img_size", 224)

    encoder = hydra.utils.instantiate(enc_cfg).to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    encoder_image_size = (img_size // 16) * encoder.patch_size  # 196
    enc_transform = transforms.Resize(encoder_image_size)

    print(f"[encoder] frozen DINOv2  emb_dim={encoder.emb_dim}  "
          f"patch_size={encoder.patch_size}  "
          f"input resized to {encoder_image_size}×{encoder_image_size}")
    return encoder, enc_transform


def precompute_features(encoder, enc_transform, data_path, cache_path, device,
                        batch=128):
    """Run DINOv2 on all frames once and save (features, images) as float16 HDF5.

    This is the one-time cost that makes all subsequent training epochs fast.
    Cache is written split by split so a partial run can resume cleanly.
    """
    print(f"\n[precompute] building feature cache → {cache_path}")
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)

    with h5py.File(cache_path, "w") as cache:
        for split in ("train", "test"):
            dset = LiftFrameDataset(data_path, split)
            loader = DataLoader(dset, batch_size=batch, shuffle=False, num_workers=0)
            N = len(dset)
            grp = cache.create_group(split)
            feat_ds = grp.create_dataset("features", shape=(N, 196, 384),
                                         dtype="float16")
            img_ds  = grp.create_dataset("images",   shape=(N, 3, 224, 224),
                                         dtype="float16")
            ptr = 0
            t0 = time.time()
            for imgs in loader:
                imgs = imgs.to(device)
                with torch.no_grad():
                    feats = encoder(enc_transform(imgs))  # (B, 196, 384)
                B = imgs.size(0)
                feat_ds[ptr:ptr+B] = feats.cpu().to(torch.float16).numpy()
                img_ds[ptr:ptr+B]  = imgs.cpu().to(torch.float16).numpy()
                ptr += B
                if ptr % 1000 < batch:
                    print(f"  {split}: {ptr}/{N} frames  "
                          f"[{time.time()-t0:.0f}s]", flush=True)
            print(f"  {split} done: {N} frames in {time.time()-t0:.0f}s")

    print(f"[precompute] cache written: {cache_path}  "
          f"({os.path.getsize(cache_path)/1e9:.2f} GB)\n")


# --------------------------------------------------------------------------- #
def _sync(device):
    if str(device) == "mps":
        torch.mps.synchronize()


def train_one_epoch(decoder, loader, optimizer, device, profile=False):
    decoder.train()
    # Autocast runs Conv/ConvTranspose in float16 — ~2x faster on MPS/CUDA.
    # MSE loss is computed in float32 (recon.float()) for stability.
    dev_type = "cuda" if str(device).startswith("cuda") else str(device)
    use_autocast = dev_type in ("mps", "cuda")

    total, n = 0.0, 0
    t_data = t_fwd = t_bwd = 0.0
    t_iter = time.perf_counter()
    for feats, imgs in loader:
        t0 = time.perf_counter()
        feats = feats.unsqueeze(1).to(device)   # (B, 1, 196, 384)
        imgs  = imgs.to(device)                  # (B, 3, 224, 224)
        _sync(device)
        t_data += time.perf_counter() - t0

        t0 = time.perf_counter()
        with torch.autocast(device_type=dev_type, dtype=torch.float16,
                            enabled=use_autocast):
            recon, _ = decoder(feats)
        loss = F.mse_loss(recon.float(), imgs)
        _sync(device)
        t_fwd += time.perf_counter() - t0

        t0 = time.perf_counter()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        _sync(device)
        t_bwd += time.perf_counter() - t0

        total += loss.item() * imgs.size(0)
        n += imgs.size(0)

    if profile:
        print(f"  [profile] data={t_data:.1f}s  fwd={t_fwd:.1f}s  "
              f"bwd={t_bwd:.1f}s  total={time.perf_counter()-t_iter:.1f}s")
    return total / n


@torch.no_grad()
def eval_epoch(decoder, loader, device):
    decoder.eval()
    total, n = 0.0, 0
    for feats, imgs in loader:
        feats = feats.unsqueeze(1).to(device)
        imgs  = imgs.to(device)
        recon, _ = decoder(feats)
        loss = F.mse_loss(recon, imgs)
        total += loss.item() * imgs.size(0)
        n += imgs.size(0)
    return total / n


@torch.no_grad()
def save_recon_samples(decoder, loader, device, out_path, n=8):
    """2-row PNG: originals (top) vs VQVAE reconstructions (bottom)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    decoder.eval()
    feats, imgs = next(iter(loader))
    feats = feats[:n].unsqueeze(1).to(device)
    imgs  = imgs[:n].to(device)
    recon, _ = decoder(feats)

    def to_np(t):
        return (t.cpu() * 0.5 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).numpy()

    orig_np  = to_np(imgs)
    recon_np = to_np(recon)
    k = min(n, len(orig_np))

    fig, axes = plt.subplots(2, k, figsize=(k * 2, 4), dpi=80)
    for i in range(k):
        axes[0, i].imshow(orig_np[i]);  axes[0, i].axis("off")
        axes[1, i].imshow(recon_np[i]); axes[1, i].axis("off")
    axes[0, 0].set_ylabel("original", fontsize=8, rotation=0, labelpad=44, va="center")
    axes[1, 0].set_ylabel("decoded",  fontsize=8, rotation=0, labelpad=44, va="center")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"  [recon samples] → {out_path}")


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path",  default=None,
                   help="path to image224.hdf5; defaults to $DATASET_DIR/image224.hdf5")
    p.add_argument("--cache_path", default=None,
                   help="HDF5 feature cache path (created on first run, reused after). "
                        "Defaults to <ckpt_dir>/feature_cache.h5")
    p.add_argument("--epochs",     type=int,   default=100)
    p.add_argument("--batch",      type=int,   default=256,
                   help="batch size for training (larger is fine once DINOv2 is cached)")
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--workers",    type=int,   default=4,
                   help="DataLoader workers for cache reads (0 on MPS if issues arise)")
    p.add_argument("--ckpt_dir",   default="results/decoder_ckpt")
    p.add_argument("--save_every", type=int,   default=10)
    a = p.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[device] {device}")

    # data path
    if a.data_path:
        data_path = a.data_path
    else:
        ddir = os.environ.get("DATASET_DIR", "")
        if not ddir:
            raise RuntimeError("Set DATASET_DIR or pass --data_path")
        data_path = os.path.join(ddir, "image224.hdf5")
    if not os.path.exists(data_path):
        raise FileNotFoundError(data_path)

    os.makedirs(a.ckpt_dir, exist_ok=True)
    cache_path = a.cache_path or os.path.join(a.ckpt_dir, "feature_cache.h5")

    # Pre-compute DINOv2 features once; reuse on subsequent runs.
    if not os.path.exists(cache_path):
        encoder, enc_transform = build_encoder(device)
        precompute_features(encoder, enc_transform, data_path, cache_path,
                            device, batch=128)
        del encoder, enc_transform  # free memory before training
        if str(device) == "mps":
            torch.mps.empty_cache()
    else:
        print(f"[precompute] cache found at {cache_path} — skipping DINOv2 pass")

    # Training uses only the cache — no DINOv2 involved.
    train_dset = FeatureCacheDataset(cache_path, "train")
    val_dset   = FeatureCacheDataset(cache_path, "test")

    # Data is in RAM — num_workers=0 is fastest (no process spawn / pickle overhead).
    train_loader = DataLoader(train_dset, batch_size=a.batch, shuffle=True,
                              num_workers=0)
    val_loader   = DataLoader(val_dset,   batch_size=a.batch, shuffle=False,
                              num_workers=0)

    decoder   = VQVAE(emb_dim=384, quantize=False).to(device)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=a.lr)

    best_path = os.path.join(a.ckpt_dir, "decoder_best.pt")
    best_val  = float("inf")

    print(f"\nTrain: {len(train_dset)} frames  |  Val: {len(val_dset)} frames")
    print(f"epochs={a.epochs}  batch={a.batch}  lr={a.lr}  "
          f"ckpt_dir={a.ckpt_dir}\n")

    save_recon_samples(decoder, val_loader, device,
                       os.path.join(a.ckpt_dir, "recon_ep000.png"))

    # Pre-compile MPS Metal kernels (forward + backward) before epoch timing.
    # ConvTranspose2d kernels can take 60-120s to JIT-compile on first use.
    # After this warmup, all subsequent epochs use cached compiled kernels.
    if str(device) == "mps":
        print("[warmup] pre-compiling MPS Metal kernels "
              "(one-time per session, ~1-2 min) ...", flush=True)
        _wf = torch.randn(a.batch, 1, 196, 384, device=device)
        _wi = torch.randn(a.batch, 3, 224, 224, device=device)
        _wr, _ = decoder(_wf)
        _wloss = F.mse_loss(_wr, _wi)
        _wloss.backward()
        optimizer.zero_grad()
        torch.mps.synchronize()
        del _wf, _wi, _wr, _wloss
        torch.mps.empty_cache()
        print("[warmup] done — Metal kernels compiled, epoch timing starts now")

    for epoch in range(1, a.epochs + 1):
        t0  = time.time()
        tr  = train_one_epoch(decoder, train_loader, optimizer, device,
                              profile=(epoch == 1))
        val = eval_epoch(decoder, val_loader, device)

        improved = val < best_val
        if improved:
            best_val = val
            torch.save(decoder.state_dict(), best_path)

        print(f"epoch {epoch:03d}/{a.epochs}  "
              f"train {tr:.5f}  val {val:.5f}"
              f"{'  *' if improved else ''}  [{time.time()-t0:.1f}s]")

        if epoch % a.save_every == 0:
            ckpt = os.path.join(a.ckpt_dir, f"decoder_ep{epoch:03d}.pt")
            torch.save(decoder.state_dict(), ckpt)
            save_recon_samples(decoder, val_loader, device,
                               os.path.join(a.ckpt_dir, f"recon_ep{epoch:03d}.png"))

    save_recon_samples(decoder, val_loader, device,
                       os.path.join(a.ckpt_dir, f"recon_ep{a.epochs:03d}.png"))

    print(f"\nDone. Best val MSE: {best_val:.5f}")
    print(f"Best checkpoint → {best_path}")
    print(f"\nUse with floor_control_plan.py:")
    print(f"  python floor_control_plan.py --decoder_path {best_path} ...")


if __name__ == "__main__":
    main()
