"""
Extract images from a robomimic-style HDF5 for beta-VAE training.

What it does, in order:
  1. INSPECT  -- open the file, locate the demos, and report which obs keys exist
                 with their shape / dtype / value range. This is where you catch
                 missing image keys, wrong dtype, or a flipped/odd layout BEFORE
                 training anything.
  2. SAMPLES  -- dump a few frames per image key as PNGs so you can eyeball
                 orientation, channel order, and content.
  3. EXTRACT  -- stream every frame for the chosen keys into an output HDF5
                 (memory-safe: appended demo-by-demo, never all in RAM), stored
                 as uint8 (H,W,3). Normalize to float at train time, not here.

Standard robomimic layout assumed:
    data/demo_<i>/obs/<key>            e.g. agentview_image, robot0_eye_in_hand_image
    data/mask/<filter_key>             optional train/valid split (list of demo names)
If the image keys are absent, the dataset is likely low-dim only -- regenerate
images with robomimic's dataset_states_to_obs.py (render on) and re-run.

NOTE: this script is written against the robomimic convention but cannot be run
here (no dataset/GPU); set the paths below and run it on your machine.
"""

import os
import argparse
import numpy as np
import h5py
from PIL import Image

# ----- configuration (override via CLI) ------------------------------------ #
DEFAULTS = dict(
    hdf5_path="/Users/swapnilmallick/Desktop/ROBOSUITE_RRT/datasets/lift/ph/image.hdf5",
    out_path="/Users/swapnilmallick/Desktop/ROBOSUITE_RRT/rrt_latent_space/vae_images.hdf5",
    image_keys=["agentview_image", "robot0_eye_in_hand_image"],
    filter_key=None,        # e.g. "train" to use data/mask/train; None = all demos
    max_demos=None,         # cap number of demos (None = all)
    sample_frames=8,        # PNGs dumped per key for eyeballing
    sample_dir=None,        # default: <out_dir>/samples
)


# --------------------------------------------------------------------------- #
def find_demo_group(f):
    """Return (group_holding_demos, sorted_demo_names)."""
    root = f["data"] if "data" in f else f
    demos = [k for k in root.keys() if k.startswith("demo")]
    # sort by integer index demo_<i> when possible
    def idx(name):
        try:
            return int(name.split("_")[-1])
        except ValueError:
            return name
    demos = sorted(demos, key=idx)
    return root, demos


def apply_filter(f, root, demos, filter_key):
    if filter_key is None:
        return demos
    if "mask" not in f or filter_key not in f["mask"]:
        raise KeyError(f"filter_key '{filter_key}' not found under data/mask/. "
                       f"Available: {list(f['mask'].keys()) if 'mask' in f else 'none'}")
    keep = {n.decode() if isinstance(n, bytes) else str(n) for n in f["mask"][filter_key][:]}
    return [d for d in demos if d in keep]


def obs_group(root, demo):
    g = root[demo]
    return g["obs"] if "obs" in g else g


def inspect(path, image_keys, filter_key):
    with h5py.File(path, "r") as f:
        root, demos = find_demo_group(f)
        demos = apply_filter(f, root, demos, filter_key)
        print(f"file              : {path}")
        print(f"demos             : {len(demos)}"
              + (f"  (filter_key={filter_key})" if filter_key else ""))
        if not demos:
            print("!! no demos found"); return None
        d0 = demos[0]
        og = obs_group(root, d0)
        all_obs = list(og.keys())
        print(f"obs keys in {d0}  : {all_obs}")
        present, total_frames = [], {}
        for k in image_keys:
            if k in og:
                ds = og[k]
                present.append(k)
                # total frame count across demos for this key
                n = sum(obs_group(root, d)[k].shape[0] for d in demos if k in obs_group(root, d))
                total_frames[k] = n
                arr0 = ds[0]
                print(f"  [OK]  {k}: per-demo shape {ds.shape}, dtype {ds.dtype}, "
                      f"frame {arr0.shape}, range [{arr0.min()},{arr0.max()}], "
                      f"total frames {n}")
            else:
                print(f"  [MISSING] {k}  -> not in obs. "
                      f"Dataset may be low-dim only; render images first.")
        return dict(demos=demos, present=present, total_frames=total_frames)


def dump_samples(path, image_keys, demos, sample_dir, n_samples):
    os.makedirs(sample_dir, exist_ok=True)
    with h5py.File(path, "r") as f:
        root, _ = find_demo_group(f)
        for k in image_keys:
            saved = 0
            for d in demos:
                og = obs_group(root, d)
                if k not in og:
                    continue
                ds = og[k]
                idxs = np.linspace(0, ds.shape[0]-1, min(n_samples, ds.shape[0])).astype(int)
                for t in idxs:
                    img = np.asarray(ds[t])
                    if img.dtype != np.uint8:   # best-effort display scaling
                        img = (255 * (img - img.min()) / (np.ptp(img) + 1e-9)).astype(np.uint8)
                    Image.fromarray(img).save(
                        os.path.join(sample_dir, f"{k}_{d}_t{t:04d}.png"))
                    saved += 1
                    if saved >= n_samples:
                        break
                if saved >= n_samples:
                    break
            print(f"  dumped {saved} sample PNGs for {k} -> {sample_dir}")


def extract(path, out_path, image_keys, demos, total_frames):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with h5py.File(path, "r") as f, h5py.File(out_path, "w") as out:
        root, _ = find_demo_group(f)
        # create one fixed-size uint8 dataset per present key (streamed fill)
        writers = {}
        for k in image_keys:
            if k not in total_frames:
                continue
            # infer frame shape from the first demo that has the key
            fshape = None
            for d in demos:
                og = obs_group(root, d)
                if k in og:
                    fshape = og[k].shape[1:]; break
            n = total_frames[k]
            dset = out.create_dataset(k, shape=(n, *fshape), dtype="uint8",
                                      chunks=(min(64, n), *fshape), compression="gzip")
            writers[k] = [dset, 0]      # [dataset, write cursor]

        for d in demos:
            og = obs_group(root, d)
            for k in writers:
                if k not in og:
                    continue
                arr = np.asarray(og[k][:])           # (T,H,W,3) for this demo
                if arr.dtype != np.uint8:
                    arr = arr.astype(np.uint8)        # assume already 0..255
                dset, cur = writers[k]
                dset[cur:cur+arr.shape[0]] = arr
                writers[k][1] = cur + arr.shape[0]
        # record provenance
        out.attrs["source"] = path
        out.attrs["keys"] = ",".join(writers.keys())
        for k, (dset, cur) in writers.items():
            print(f"  wrote {cur} frames to '{k}' in {out_path}  (shape {dset.shape})")


def main(cfg):
    info = inspect(cfg["hdf5_path"], cfg["image_keys"], cfg["filter_key"])
    if info is None or not info["present"]:
        print("No image keys present -> nothing to extract. Render images first.")
        return
    demos = info["demos"]
    if cfg["max_demos"]:
        demos = demos[:cfg["max_demos"]]
        # recompute totals for the capped set
        with h5py.File(cfg["hdf5_path"], "r") as f:
            root, _ = find_demo_group(f)
            info["total_frames"] = {
                k: sum(obs_group(root, d)[k].shape[0] for d in demos if k in obs_group(root, d))
                for k in info["present"]}
    sample_dir = cfg["sample_dir"] or os.path.join(
        os.path.dirname(cfg["out_path"]) or ".", "samples")
    print("\n[2] dumping sample frames ...")
    dump_samples(cfg["hdf5_path"], info["present"], demos, sample_dir, cfg["sample_frames"])
    print("\n[3] extracting frames ...")
    extract(cfg["hdf5_path"], cfg["out_path"], info["present"], demos, info["total_frames"])
    print("\nDone. Load for training with, e.g.:")
    print("  import h5py, numpy as np")
    print(f"  f = h5py.File('{cfg['out_path']}','r'); X = f['{info['present'][0]}']  # (N,H,W,3) uint8")
    print("  # normalize at train time: x = X[i].astype('float32')/255.0")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--hdf5_path", default=DEFAULTS["hdf5_path"])
    p.add_argument("--out_path", default=DEFAULTS["out_path"])
    p.add_argument("--image_keys", nargs="+", default=DEFAULTS["image_keys"])
    p.add_argument("--filter_key", default=DEFAULTS["filter_key"])
    p.add_argument("--max_demos", type=int, default=DEFAULTS["max_demos"])
    p.add_argument("--sample_frames", type=int, default=DEFAULTS["sample_frames"])
    p.add_argument("--sample_dir", default=DEFAULTS["sample_dir"])
    main(vars(p.parse_args()))
