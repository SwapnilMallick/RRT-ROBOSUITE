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
    for name, vals in summary.items():
        rhos = np.array([v[0] for v in vals]); drs = np.array([v[1] for v in vals])
        verdict = ("RELIABLE" if rhos.mean() > 0.7 and drs.mean() > 0.05
                   else "WEAK" if rhos.mean() > 0.4 else "UNRELIABLE")
        print(f"  {name:10s} | mean Spearman {rhos.mean():+.3f} "
              f"(want ~ +1) | mean dyn-range {drs.mean():.3f} | {verdict}")
    for ax in axes:
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.suptitle("Encoder reliability: cosine-sim to goal vs distance "
                 "(goal on right; monotone rise = reliable)", fontsize=12)
    fig.tight_layout(); fig.savefig(cfg["out"], dpi=130)
    print(f"\nsaved {cfg['out']}")


# --------------------------------------------------------------------------- #
class SyntheticEncoder:
    """Stand-in to verify the analysis/plotting without weights.
    quality in [0,1]: 1 = perfectly tracks distance, 0 = noise."""
    def __init__(self, name, quality, flat=False):
        self.name, self.quality, self.flat = name, quality, flat
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
                SyntheticEncoder("I-JEPA", 0.3, flat=True)]
    encs = []
    for name in cfg["encoders"]:
        if name == "openclip": encs.append(OpenCLIPEncoder(device))
        elif name == "dinov2": encs.append(DINOv2Encoder(device, pooling=cfg["dino_pool"]))
        elif name == "ijepa":  encs.append(IJEPAEncoder(device))
    return encs


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--hdf5", default="/path/to/image224.hdf5")
    p.add_argument("--demos", nargs="+", default=["demo_0", "demo_1", "demo_2"])
    p.add_argument("--image_key", default="agentview_image")
    p.add_argument("--encoders", nargs="+", default=["openclip", "dinov2", "ijepa"])
    p.add_argument("--dino_pool", default="cls", choices=["cls", "mean"])
    p.add_argument("--frame_step", type=int, default=1)
    p.add_argument("--out", default="./encoder_reliability.png")
    p.add_argument("--synthetic", action="store_true",
                   help="verify analysis/plotting with fake encoders, no weights")
    run(vars(p.parse_args()))
