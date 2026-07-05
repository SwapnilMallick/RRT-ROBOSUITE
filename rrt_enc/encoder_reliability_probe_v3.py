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
import pathlib
import warnings
import numpy as np
import h5py
import torch
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

_HERE = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent


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
        # some transformers versions return BaseModelOutputWithPooling instead of a tensor
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


def select_reach_index(dist, eps):
    """First frame within eps of the cube (approach moment for a REACH task),
    else closest-approach argmin with a warning. dist is per-frame gripper-cube
    distance."""
    below = np.where(dist < eps)[0]
    if len(below) > 0:
        return int(below[0]), False
    return int(np.argmin(dist)), True     # (idx, fell_back)


def load_demo(path, demo, image_key="agentview_image", goal_eps=None):
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
    # REACH task: truncate at the first frame within goal_eps of the cube, so the
    # goal (last frame after truncation) is the approach moment, not grasp/lift.
    if dist is not None and goal_eps is not None:
        ridx, fell = select_reach_index(dist, goal_eps)
        if fell:
            warnings.warn(f"{demo}: no frame within goal_eps={goal_eps}; using "
                          f"closest approach at t={ridx} as reach goal.")
        frames = frames[:ridx + 1]
        dist = dist[:ridx + 1]
        axis_kind += f" | truncated at reach t={ridx}"
    return frames, dist, axis_kind


def load_demo_two_views(path_agent, path_side, demo,
                        agent_key="agentview_image", side_key="sideview_image",
                        goal_eps=None):
    """Load agentview + sideview for one demo from TWO files, with alignment
    verification (equal frame counts + identical robot0_eef_pos). Then, for the
    REACH task, truncate BOTH views jointly at the first frame within goal_eps of
    the cube so the goal (last frame) is the approach moment, not grasp/lift."""
    with h5py.File(path_agent, "r") as f:
        ra = f["data"] if "data" in f else f
        oga = ra[demo]["obs"] if "obs" in ra[demo] else ra[demo]
        fa = np.asarray(oga[agent_key][:])
        eef_a = np.asarray(oga["robot0_eef_pos"][:], np.float64)
        T = fa.shape[0]
        ckey = find_cube_key(oga)
        da = (np.linalg.norm(eef_a - cube_pos_from(oga, ckey, T), axis=1)
              if ckey is not None else None)
    with h5py.File(path_side, "r") as f:
        rs = f["data"] if "data" in f else f
        if demo not in rs:
            raise KeyError(f"{demo} not in sideview file {path_side}")
        og = rs[demo]["obs"] if "obs" in rs[demo] else rs[demo]
        fs = np.asarray(og[side_key][:])
        eef_s = np.asarray(og["robot0_eef_pos"][:], np.float64)
    if fs.shape[0] != fa.shape[0]:
        raise ValueError(f"{demo}: frame-count mismatch agent={fa.shape[0]} "
                         f"side={fs.shape[0]} -- files are NOT aligned")
    max_dev = float(np.abs(eef_a - eef_s).max())
    if max_dev > 1e-4:
        raise ValueError(f"{demo}: eef_pos differs across files (max {max_dev:.2e}) "
                         "-- the two renders are NOT the same states; alignment failed")
    if da is not None and goal_eps is not None:
        ridx, fell = select_reach_index(da, goal_eps)
        if fell:
            warnings.warn(f"{demo}: no frame within goal_eps={goal_eps}; using "
                          f"closest approach at t={ridx} as reach goal.")
        fa, fs, da = fa[:ridx + 1], fs[:ridx + 1], da[:ridx + 1]
    return fa, fs, da            # distance from the agentview file (identical if aligned)


# --------------------------------------------------------------------------- #
def _sim_to_goal(z):
    zg = z[-1:]
    return (z * zg).sum(-1)                                     # cosine sim to last


def _score(sim, dist, idx):
    if dist is not None:
        x = dist[idx]
        rho, _ = spearmanr(sim, -x)          # want sim up as distance down
        xlabel = "gripper-cube distance"
    else:
        x = idx.astype(float)
        rho, _ = spearmanr(sim, x)
        xlabel = "frame index (proxy)"
    return x, float(rho), xlabel, float(sim.max() - sim.min())


def analyze(frames, dist, encoder, frame_step=1):
    """Single-view analysis (unchanged interface)."""
    idx = np.arange(0, len(frames), frame_step)
    z = encoder.embed(list(frames[idx]))                       # (n, D) normalized
    sim = _sim_to_goal(z)
    x, rho, xlabel, drange = _score(sim, dist, idx)
    return dict(x=x, sim=sim, rho=rho, xlabel=xlabel, drange=drange)


def analyze_three_way(fr_agent, fr_side, dist, encoder, frame_step=1):
    """Return agent-only, side-only, and concatenated(both) configs. Each view is
    embedded through the SAME encoder and L2-normalized; 'both' concatenates the
    two per-view (already-normalized) embeddings and re-normalizes the result so
    neither view dominates by norm."""
    idx = np.arange(0, len(fr_agent), frame_step)
    za = encoder.embed(list(fr_agent[idx]))                    # (n, Da) normalized
    zs = encoder.embed(list(fr_side[idx]))                     # (n, Ds) normalized
    zb = np.concatenate([za, zs], axis=1)
    zb = zb / (np.linalg.norm(zb, axis=1, keepdims=True) + 1e-9)
    out = {}
    for tag, z in (("agent", za), ("side", zs), ("both", zb)):
        sim = _sim_to_goal(z)
        x, rho, xlabel, drange = _score(sim, dist, idx)
        out[tag] = dict(x=x, sim=sim, rho=rho, xlabel=xlabel, drange=drange)
    return out


def run(cfg):
    device = get_device()
    print(f"device: {device}")
    if cfg.get("goal_eps") is not None and cfg["goal_eps"] <= 0:
        cfg["goal_eps"] = None            # disable reach-truncation
    encoders = build_encoders(cfg, device)
    if cfg.get("hdf5_side"):
        run_three_way(cfg, device, encoders)
    else:
        run_single_view(cfg, device, encoders)


def run_single_view(cfg, device, encoders):
    fig, axes = plt.subplots(1, len(encoders), figsize=(5.2*len(encoders), 4.6),
                             squeeze=False)
    axes = axes[0]
    summary = {e.name: [] for e in encoders}
    for di, demo in enumerate(cfg["demos"]):
        frames, dist, axis_kind = load_demo(cfg["hdf5"], demo, cfg["image_key"], cfg["goal_eps"])
        if dist is None and di == 0:
            warnings.warn("No cube-position key found; x-axis falls back to frame "
                          "index (a proxy).")
        print(f"demo {demo}: {len(frames)} frames | x-axis = {axis_kind}")
        for ax, enc in zip(axes, encoders):
            r = analyze(frames, dist, enc, cfg["frame_step"])
            summary[enc.name].append((r["rho"], r["drange"]))
            ax.plot(r["x"], r["sim"], marker="o", ms=3, lw=1, alpha=0.8,
                    label=f"{demo} (ρ={r['rho']:+.2f})")
            ax.set_xlabel(r["xlabel"]); ax.set_ylabel("cosine sim to goal frame")
            ax.set_title(enc.name)
            if dist is not None:
                ax.invert_xaxis()

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
    pathlib.Path(cfg["out"]).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(cfg["out"], dpi=130)
    print(f"\nsaved {cfg['out']}")


def run_three_way(cfg, device, encoders):
    """agentview vs sideview vs both(concat) per encoder, from two aligned files."""
    views = ["agent", "side", "both"]
    colors = {"agent": "#1f77b4", "side": "#ff7f0e", "both": "#2ca02c"}
    # one row per encoder, one column per view-config
    fig, axes = plt.subplots(len(encoders), 3, squeeze=False,
                             figsize=(15, 4.2*len(encoders)))
    # summary[enc][view] = list of (rho, drange) over demos
    summary = {e.name: {v: [] for v in views} for e in encoders}
    for demo in cfg["demos"]:
        fa, fs, dist = load_demo_two_views(
            cfg["hdf5"], cfg["hdf5_side"], demo,
            cfg["image_key"], cfg["side_key"], cfg["goal_eps"])
        print(f"demo {demo}: {len(fa)} frames (agent+side aligned, reach-truncated)")
        for ei, enc in enumerate(encoders):
            res = analyze_three_way(fa, fs, dist, enc, cfg["frame_step"])
            for vi, v in enumerate(views):
                r = res[v]
                summary[enc.name][v].append((r["rho"], r["drange"]))
                ax = axes[ei][vi]
                ax.plot(r["x"], r["sim"], marker="o", ms=3, lw=1, alpha=0.8,
                        color=colors[v], label=f"{demo} (ρ={r['rho']:+.2f})")
                ax.set_title(f"{enc.name} — {v}")
                ax.set_xlabel(r["xlabel"]); ax.set_ylabel("cos sim to goal")
                if dist is not None:
                    ax.invert_xaxis()
                ax.grid(alpha=0.3); ax.legend(fontsize=6)

    print("\n=== three-way view comparison (mean over demos) ===")
    bb = {e.name: getattr(e, "backbone", "?") for e in encoders}
    for name in summary:
        print(f"\n {name}  [{bb[name]}]")
        stats = {}
        for v in views:
            rr = np.array([a for a, _ in summary[name][v]])
            dd = np.array([b for _, b in summary[name][v]])
            stats[v] = (rr.mean(), dd.mean())
            print(f"   {v:6s} | Spearman {rr.mean():+.3f} | dyn-range {dd.mean():.3f}")
        # does 'both' earn its double cost? compare near-goal DISCRIMINABILITY
        best_single = max(stats["agent"][1], stats["side"][1])
        gain = stats["both"][1] - best_single
        verdict = ("BOTH HELPS (dyn-range beats best single view "
                   f"by {gain:+.3f})" if gain > 0.01 else
                   "both ~ best single view -> second view not worth double cost")
        print(f"   -> {verdict}")
    fig.suptitle("Three-way view comparison: agentview vs sideview vs both "
                 "(goal on right; want monotone rise + wide near-goal spread)",
                 fontsize=12)
    pathlib.Path(cfg["out"]).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(cfg["out"], dpi=120)
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
    p.add_argument("--hdf5",
                   default=str(_REPO_ROOT / "datasets" / "lift" / "ph" / "image224.hdf5"))
    p.add_argument("--hdf5_side", default=None,
                   help="sideview file; if given, runs the 3-way agent/side/both comparison")
    p.add_argument("--side_key", default="sideview_image")
    p.add_argument("--demos", nargs="+", default=["demo_0", "demo_1", "demo_2"])
    p.add_argument("--image_key", default="agentview_image")
    p.add_argument("--encoders", nargs="+",
                   default=["openclip", "dinov2", "ijepa", "ibot", "mae",
                            "siglip", "r3m", "vc1"])
    p.add_argument("--dino_pool", default="cls", choices=["cls", "mean"])
    p.add_argument("--vit_pool", default="cls", choices=["cls", "mean"],
                   help="pooling for iBOT/MAE (try both; mean often better for local signal)")
    p.add_argument("--ibot_ckpt", default=str(_HERE / "ibot_vitb16.pth"))
    p.add_argument("--mae_ckpt", default=str(_HERE / "mae_pretrain_vit_base.pth"))
    p.add_argument("--frame_step", type=int, default=1)
    p.add_argument("--goal_eps", type=float, default=0.03,
                   help="REACH goal: truncate each demo at the first frame within "
                        "this distance of the cube (approach), so the goal frame is "
                        "not the grasped/lifted end. Set 0 or negative to disable.")
    p.add_argument("--out", default=str(_HERE / "encoder_reliability.png"))
    p.add_argument("--synthetic", action="store_true",
                   help="verify analysis/plotting with fake encoders, no weights")
    run(vars(p.parse_args()))
