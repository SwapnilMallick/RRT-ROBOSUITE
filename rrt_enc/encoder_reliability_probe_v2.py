"""
Encoder reliability probe for the cube-reach goal signal.

Question: is cosine similarity between a frame's embedding and the GOAL frame's
embedding a usable proxy for physical closeness to the goal? We test this BEFORE
building any RRT, because the RRT's termination depends entirely on it.

Method (per encoder in {OpenCLIP, DINOv2, I-JEPA}):
  * embed every frame of one or more expert demos,
  * cosine-similarity each frame's embedding to the GOAL frame (last frame),
  * x-axis = TRUE gripper-cube distance ||eef_pos - cube_pos|| if the cube
    position key is present (auto-detected); else fall back to frame index with
    a printed warning,
  * score with Spearman correlation (similarity vs negative-distance -> want ~ -1
    against distance, i.e. similarity up as distance down) and report the
    similarity DYNAMIC RANGE near the goal (a monotonic-but-flat encoder fails
    RRT termination even with good correlation),
  * overlay multiple demos per encoder.

Reliable encoder  -> strong negative Spearman(sim, distance) AND wide dynamic
                     range, consistent across demos.
Unreliable        -> weak/zero/positive correlation, or flat similarity (arm
                     dominates the frame, small cube contributes little).

Runs on Apple MPS (falls back to CPU). Real encoders are loaded on your machine;
a --synthetic mode verifies the analysis/plotting without weights.
"""

import argparse
import os
import warnings
import numpy as np
import h5py
import torch
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
# Encoders: each returns L2-normalized (N, D) embeddings for a batch of HWC
# uint8 frames, using ITS OWN preprocessing. Pooling choice is explicit per encoder.
# --------------------------------------------------------------------------- #
class OpenCLIPEncoder:
    name = "OpenCLIP"
    backbone = "ViT-B/32 (CLIP, LAION-2B)"
    def __init__(self, device, model_name="ViT-B-32", pretrained="laion2b_s34b_b79k"):
        import open_clip
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained)
        self.model = self.model.to(device).eval()
    @torch.no_grad()
    def embed(self, frames_uint8):
        from PIL import Image
        ims = torch.stack([self.preprocess(Image.fromarray(f)) for f in frames_uint8])
        z = self.model.encode_image(ims.to(self.device))      # global image embedding
        return torch.nn.functional.normalize(z, dim=-1).cpu().numpy()


class DINOv2Encoder:
    name = "DINOv2"
    backbone = "ViT-S/14 (DINOv2)"
    def __init__(self, device, repo="dinov2_vits14", pooling="cls"):
        self.device = device
        self.pooling = pooling                                 # "cls" or "mean"
        self.model = torch.hub.load("facebookresearch/dinov2", repo).to(device).eval()
        import torchvision.transforms as T
        self.tf = T.Compose([
            T.ToPILImage(), T.Resize(224), T.CenterCrop(224), T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    @torch.no_grad()
    def embed(self, frames_uint8):
        ims = torch.stack([self.tf(f) for f in frames_uint8]).to(self.device)
        if self.pooling == "cls":
            z = self.model(ims)                                # CLS token (N, D)
        else:
            feats = self.model.forward_features(ims)
            z = feats["x_norm_patchtokens"].mean(1)            # mean-pooled patches
        return torch.nn.functional.normalize(z, dim=-1).cpu().numpy()


class IJEPAEncoder:
    name = "I-JEPA"
    backbone = "ViT-H/14 (I-JEPA)"
    def __init__(self, device, hf_repo="facebook/ijepa_vith14_1k"):
        # I-JEPA via HuggingFace transformers (no CLS -> mean-pool patch tokens)
        from transformers import AutoModel, AutoProcessor
        self.device = device
        self.proc = AutoProcessor.from_pretrained(hf_repo)
        self.model = AutoModel.from_pretrained(hf_repo).to(device).eval()
    @torch.no_grad()
    def embed(self, frames_uint8):
        from PIL import Image
        ims = [Image.fromarray(f) for f in frames_uint8]
        inp = self.proc(images=ims, return_tensors="pt").to(self.device)
        out = self.model(**inp).last_hidden_state               # (N, tokens, D)
        z = out.mean(1)                                         # mean-pool (no CLS)
        return torch.nn.functional.normalize(z, dim=-1).cpu().numpy()


class IBOTEncoder:
    name = "iBOT"
    backbone = "ViT-B/16 (iBOT)"
    def __init__(self, device, ckpt_path="ibot_vitb16.pth", pooling="cls"):
        # iBOT release is ViT (no Swin checkpoint). Build a timm ViT-B/16 and load
        # the iBOT state dict (download the checkpoint from bytedance/ibot first).
        import timm
        self.device, self.pooling = device, pooling
        self.model = timm.create_model("vit_base_patch16_224", pretrained=False,
                                       num_classes=0).to(device).eval()
        sd = torch.load(ckpt_path, map_location="cpu")
        sd = sd.get("state_dict", sd.get("teacher", sd))
        sd = {k.replace("module.", "").replace("backbone.", ""): v for k, v in sd.items()}
        self.model.load_state_dict(sd, strict=False)
        import torchvision.transforms as T
        self.tf = T.Compose([
            T.ToPILImage(), T.Resize(256), T.CenterCrop(224), T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    @torch.no_grad()
    def embed(self, frames_uint8):
        ims = torch.stack([self.tf(f) for f in frames_uint8]).to(self.device)
        tok = self.model.forward_features(ims)                  # (N, 1+P, D)
        z = tok[:, 0] if self.pooling == "cls" else tok[:, 1:].mean(1)
        return torch.nn.functional.normalize(z, dim=-1).cpu().numpy()


class MAEEncoder:
    name = "MAE"
    backbone = "ViT-B/16 (MAE)"
    def __init__(self, device, ckpt_path="mae_pretrain_vit_base.pth", pooling="mean"):
        # MAE's CLS token is weak (no contrastive/CLS objective) -> default mean-pool.
        import timm
        self.device, self.pooling = device, pooling
        self.model = timm.create_model("vit_base_patch16_224", pretrained=False,
                                       num_classes=0, global_pool="").to(device).eval()
        sd = torch.load(ckpt_path, map_location="cpu")
        sd = sd.get("model", sd)
        self.model.load_state_dict(sd, strict=False)
        import torchvision.transforms as T
        self.tf = T.Compose([
            T.ToPILImage(), T.Resize(256), T.CenterCrop(224), T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    @torch.no_grad()
    def embed(self, frames_uint8):
        ims = torch.stack([self.tf(f) for f in frames_uint8]).to(self.device)
        tok = self.model.forward_features(ims)                  # (N, 1+P, D)
        z = tok[:, 0] if self.pooling == "cls" else tok[:, 1:].mean(1)
        return torch.nn.functional.normalize(z, dim=-1).cpu().numpy()


class SigLIPEncoder:
    name = "SigLIP"
    backbone = "ViT-B/16 (SigLIP)"
    def __init__(self, device, hf_repo="google/siglip-base-patch16-224"):
        from transformers import AutoModel, AutoProcessor
        self.device = device
        self.proc = AutoProcessor.from_pretrained(hf_repo)
        self.model = AutoModel.from_pretrained(hf_repo).to(device).eval()
    @torch.no_grad()
    def embed(self, frames_uint8):
        from PIL import Image
        ims = [Image.fromarray(f) for f in frames_uint8]
        inp = self.proc(images=ims, return_tensors="pt").to(self.device)
        z = self.model.get_image_features(**inp)
        # transformers >=5.x may return BaseModelOutputWithPooling instead of tensor
        if not isinstance(z, torch.Tensor):
            z = z.pooler_output
        return torch.nn.functional.normalize(z, dim=-1).cpu().numpy()


class R3MEncoder:
    name = "R3M"
    backbone = "ResNet-50 (R3M, Ego4D)"
    def __init__(self, device, arch="resnet50"):
        # R3M wants raw [0,255] images at 224 (it normalizes internally). ResNet
        # backbone -> a single global vector, no CLS/mean choice.
        from r3m import load_r3m
        import torchvision.transforms as T
        self.device = device
        self.model = load_r3m(arch).to(device).eval()
        self.tf = T.Compose([T.ToPILImage(), T.Resize(256), T.CenterCrop(224),
                             T.ToTensor()])                      # keep [0,1]->*255 below
    @torch.no_grad()
    def embed(self, frames_uint8):
        ims = torch.stack([self.tf(f) for f in frames_uint8]).to(self.device) * 255.0
        z = self.model(ims)                                     # (N, 2048)
        return torch.nn.functional.normalize(z, dim=-1).cpu().numpy()


class VC1Encoder:
    name = "VC-1"
    backbone = "ViT-L/14 (VC-1, Ego4D+IN)"
    def __init__(self, device, model_name="vc1_vitl"):
        # VC-1 ships its own loader + transforms (expects ~250px input -> 224).
        from vc_models.models.vit import model_utils
        self.device = device
        (self.model, self.embd_size, self.model_transforms,
         self.info) = model_utils.load_model(model_utils.VC1_LARGE_NAME)
        self.model = self.model.to(device).eval()
    @torch.no_grad()
    def embed(self, frames_uint8):
        import torchvision.transforms as T
        to_t = T.ToTensor()
        ims = torch.stack([self.model_transforms(to_t(f)) for f in frames_uint8])
        z = self.model(ims.to(self.device))                    # (N, 1024)
        return torch.nn.functional.normalize(z, dim=-1).cpu().numpy()


# --------------------------------------------------------------------------- #
# Data: frames + true gripper-cube distance (auto-detected), per demo
# --------------------------------------------------------------------------- #
CUBE_KEY_CANDIDATES = ["object", "cube_pos", "object-state", "cube_position"]


def find_cube_key(obs_group):
    for k in CUBE_KEY_CANDIDATES:
        if k in obs_group:
            return k
    return None


def cube_pos_from(obs_group, key, T):
    arr = np.asarray(obs_group[key][:], dtype=np.float64)
    # Lift's 'object' packs [cube_pos(3), cube_quat(4), ...]; take first 3 dims
    return arr[:, :3] if arr.ndim == 2 and arr.shape[1] >= 3 else arr.reshape(T, -1)[:, :3]


def load_demo(path, demo, image_key="agentview_image"):
    with h5py.File(path, "r") as f:
        root = f["data"] if "data" in f else f
        g = root[demo]
        og = g["obs"] if "obs" in g else g
        frames = np.asarray(og[image_key][:])                  # (T,H,W,3) uint8
        eef = np.asarray(og["robot0_eef_pos"][:], np.float64)  # (T,3)
        T = frames.shape[0]
        ckey = find_cube_key(og)
        if ckey is not None:
            cube = cube_pos_from(og, ckey, T)
            dist = np.linalg.norm(eef - cube, axis=1)          # true gripper-cube dist
            axis_kind = f"true gripper-cube distance (key '{ckey}')"
        else:
            dist = None
            axis_kind = "FRAME INDEX (no cube key found -> proxy axis)"
    return frames, dist, axis_kind


# --------------------------------------------------------------------------- #
def analyze(frames, dist, encoder, frame_step=1):
    idx = np.arange(0, len(frames), frame_step)
    fr = frames[idx]
    z = encoder.embed(list(fr))                                # (n, D), normalized
    zg = z[-1:]                                                 # goal = last frame
    sim = (z * zg).sum(-1)                                      # cosine sim to goal
    if dist is not None:
        x = dist[idx]
        rho, _ = spearmanr(sim, -x)         # want sim up as distance down -> rho ~ +1
        xlabel = "gripper-cube distance"
    else:
        x = idx.astype(float)
        rho, _ = spearmanr(sim, x)          # sim up as frame index up
        xlabel = "frame index (proxy)"
    return dict(x=x, sim=sim, rho=rho, xlabel=xlabel,
                drange=float(sim.max() - sim.min()))


def run(cfg):
    device = get_device()
    print(f"device: {device}")
    encoders = build_encoders(cfg, device)

    fig, axes = plt.subplots(1, len(encoders), figsize=(5.2*len(encoders), 4.6),
                             squeeze=False)
    axes = axes[0]
    summary = {e.name: [] for e in encoders}
    for di, demo in enumerate(cfg["demos"]):
        frames, dist, axis_kind = load_demo(cfg["hdf5"], demo, cfg["image_key"])
        if dist is None and di == 0:
            warnings.warn("No cube-position key found; x-axis falls back to frame "
                          "index (a proxy). Spearman is vs frame index, not true "
                          "distance. Check obs keys with get_dataset_info --verbose.")
        print(f"demo {demo}: {len(frames)} frames | x-axis = {axis_kind}")
        for ax, enc in zip(axes, encoders):
            r = analyze(frames, dist, enc, cfg["frame_step"])
            summary[enc.name].append((r["rho"], r["drange"]))
            ax.plot(r["x"], r["sim"], marker="o", ms=3, lw=1, alpha=0.8,
                    label=f"{demo} (ρ={r['rho']:+.2f})")
            ax.set_xlabel(r["xlabel"]); ax.set_ylabel("cosine sim to goal frame")
            ax.set_title(enc.name)
            if dist is not None:
                ax.invert_xaxis()    # distance shrinks toward goal -> goal on the right

    print("\n=== reliability summary (mean over demos) ===")
    bb = {e.name: getattr(e, "backbone", "?") for e in encoders}
    for name, vals in summary.items():
        rhos = np.array([v[0] for v in vals]); drs = np.array([v[1] for v in vals])
        verdict = ("RELIABLE" if rhos.mean() > 0.7 and drs.mean() > 0.05
                   else "WEAK" if rhos.mean() > 0.4 else "UNRELIABLE")
        print(f"  {name:9s} [{bb[name]:24s}] | mean Spearman {rhos.mean():+.3f} "
              f"| mean dyn-range {drs.mean():.3f} | {verdict}")
    for ax in axes:
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.suptitle("Encoder reliability: cosine-sim to goal vs distance "
                 "(goal on right; monotone rise = reliable)", fontsize=12)
    os.makedirs(os.path.dirname(os.path.abspath(cfg["out"])), exist_ok=True)
    fig.tight_layout(); fig.savefig(cfg["out"], dpi=130)
    print(f"\nsaved {cfg['out']}")


# --------------------------------------------------------------------------- #
class SyntheticEncoder:
    """Stand-in to verify the analysis/plotting without weights.
    quality in [0,1]: 1 = perfectly tracks distance, 0 = noise."""
    def __init__(self, name, quality, flat=False):
        self.name, self.quality, self.flat = name, quality, flat
        self.backbone = "synthetic"
        self._goal = np.random.default_rng(abs(hash(name)) % 2**32).normal(size=64)
    def embed(self, frames_uint8):
        n = len(frames_uint8); rng = np.random.default_rng(len(frames_uint8))
        # fabricate embeddings whose sim-to-last rises toward the end by `quality`
        prog = np.linspace(0, 1, n)[:, None]
        base = self._goal[None, :] * prog * self.quality
        noise = rng.normal(scale=(1 - self.quality) + 0.05, size=(n, 64))
        z = base + noise
        if self.flat:
            z = self._goal[None, :] + 0.02 * rng.normal(size=(n, 64))  # near-identical
        return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9)


def build_encoders(cfg, device):
    if cfg["synthetic"]:
        return [SyntheticEncoder("OpenCLIP", 0.9),
                SyntheticEncoder("DINOv2", 0.6),
                SyntheticEncoder("I-JEPA", 0.3, flat=True),
                SyntheticEncoder("iBOT", 0.7),
                SyntheticEncoder("MAE", 0.4),
                SyntheticEncoder("SigLIP", 0.85),
                SyntheticEncoder("R3M", 0.8),
                SyntheticEncoder("VC-1", 0.75)]
    encs = []
    for name in cfg["encoders"]:
        if name == "openclip": encs.append(OpenCLIPEncoder(device))
        elif name == "dinov2": encs.append(DINOv2Encoder(device, pooling=cfg["dino_pool"]))
        elif name == "ijepa":  encs.append(IJEPAEncoder(device))
        elif name == "ibot":   encs.append(IBOTEncoder(device, ckpt_path=cfg["ibot_ckpt"],
                                                       pooling=cfg["vit_pool"]))
        elif name == "mae":    encs.append(MAEEncoder(device, ckpt_path=cfg["mae_ckpt"],
                                                      pooling=cfg["vit_pool"]))
        elif name == "siglip": encs.append(SigLIPEncoder(device))
        elif name == "r3m":    encs.append(R3MEncoder(device))
        elif name == "vc1":    encs.append(VC1Encoder(device))
        else: raise ValueError(f"unknown encoder '{name}'")
    return encs


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--hdf5", default="/path/to/image224.hdf5")
    p.add_argument("--demos", nargs="+", default=["demo_0", "demo_1", "demo_2"])
    p.add_argument("--image_key", default="agentview_image")
    p.add_argument("--encoders", nargs="+",
                   default=["openclip", "dinov2", "ijepa", "ibot", "mae",
                            "siglip", "r3m", "vc1"])
    p.add_argument("--dino_pool", default="cls", choices=["cls", "mean"])
    p.add_argument("--vit_pool", default="cls", choices=["cls", "mean"],
                   help="pooling for iBOT/MAE (try both; mean often better for local signal)")
    p.add_argument("--ibot_ckpt", default="ibot_vitb16.pth")
    p.add_argument("--mae_ckpt", default="mae_pretrain_vit_base.pth")
    p.add_argument("--frame_step", type=int, default=1)
    p.add_argument("--out", default="./plots_v2/encoder_reliability.png")
    p.add_argument("--synthetic", action="store_true",
                   help="verify analysis/plotting with fake encoders, no weights")
    run(vars(p.parse_args()))
