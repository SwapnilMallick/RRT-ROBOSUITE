"""
Robomimic (Lift) -> DINO-WM dataset adapter.

Implements the exact interface DINO-WM expects (see datasets/traj_dset.py and
datasets/pusht_dset.py in gaoyuezhou/dino_wm):

  * a TrajDataset subclass with:
      - get_seq_length(idx) -> int
      - __len__() -> number of trajectories (demos)
      - __getitem__(idx) -> (obs, act, state, info)
            obs = {"visual": (T, C, H, W) float in [0,1] (pre-transform),
                   "proprio": (T, P) float (normalized)}
            act   : (T, A) float (normalized)
            state : (T, S) float
            info  : {"shape": ...}  (free-form; kept for parity)
      - attributes: action_dim, state_dim, proprio_dim
        and *_mean/*_std for action & proprio (pre-normalized in __init__)
  * a load_lift_slice_train_val(...) factory returning (datasets, traj_dset)
    dicts keyed "train"/"valid", each a TrajSlicerDataset / TrajDataset.

Reads the robomimic image HDF5 produced by dataset_states_to_obs.py:
    data/demo_<i>/obs/agentview_image     (T,84or224,84or224,3) uint8
    data/demo_<i>/obs/<proprio keys...>   (T, d) float
    data/demo_<i>/actions                 (T, A) float
    data/mask/train , data/mask/test      (optional split filter keys)

IMPORTANT (resolution): DINOv2 wants ~224px. Render the images at 224x224
(dataset_states_to_obs.py --camera_height 224 --camera_width 224). The adapter
will still run at 84px, but the default_transform Resize(224) upsamples 84->224,
which wastes detail. Prefer native 224.

This module imports TrajDataset / TrajSlicerDataset from DINO-WM, so place it in
the repo's datasets/ directory (e.g. datasets/lift_dset.py) or keep the repo on
PYTHONPATH.
"""

import h5py
import numpy as np
import torch
from einops import rearrange

from .traj_dset import TrajDataset, TrajSlicerDataset


# robomimic Lift proprio keys to concatenate (order fixed -> stable proprio_dim).
# eef pose (pos3 + quat4) + gripper joint qpos (2) = 9 dims. Adjust if your
# dataset uses different keys; verify with get_dataset_info.py --verbose.
PROPRIO_KEYS = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]


def _list_demos(f, filter_key=None):
    root = f["data"] if "data" in f else f
    if filter_key is not None:
        if "mask" not in f or filter_key not in f["mask"]:
            raise KeyError(f"mask/{filter_key} not found; "
                           f"have {list(f['mask'].keys()) if 'mask' in f else 'none'}")
        names = [n.decode() if isinstance(n, bytes) else str(n)
                 for n in f["mask"][filter_key][:]]
    else:
        names = [k for k in root.keys() if k.startswith("demo")]
    names.sort(key=lambda s: int(s.split("_")[-1]))
    return root, names


def _obs_group(root, demo):
    g = root[demo]
    return g["obs"] if "obs" in g else g


class LiftDataset(TrajDataset):
    def __init__(
        self,
        data_path,
        filter_key=None,                 # "train" / "test" / None (all)
        image_key="agentview_image",
        proprio_keys=PROPRIO_KEYS,
        transform=None,
        n_rollout=None,
        normalize_action=True,
        action_scale=1.0,                # robomimic actions already ~[-1,1]
        preload=True,                    # load low-dim into RAM; images read lazily
    ):
        self.data_path = data_path
        self.image_key = image_key
        self.proprio_keys = list(proprio_keys)
        self.transform = transform
        self.normalize_action = normalize_action
        self.action_scale = action_scale

        with h5py.File(data_path, "r") as f:
            root, demos = _list_demos(f, filter_key)
            if n_rollout:
                demos = demos[:n_rollout]
            self.demos = demos
            self.seq_lengths, actions, proprios, states = [], [], [], []
            for d in demos:
                g = root[d]
                og = _obs_group(root, d)
                a = np.asarray(g["actions"][:], dtype=np.float32)      # (T, A)
                pr = np.concatenate([np.asarray(og[k][:], np.float32)
                                     for k in self.proprio_keys], axis=-1)  # (T, P)
                # state = full proprio (used by DINO-WM only for logging/eval)
                self.seq_lengths.append(a.shape[0])
                actions.append(a)
                proprios.append(pr)
                states.append(pr.copy())
            self.image_hw = _obs_group(root, demos[0])[image_key].shape[1:3]

        Tmax = max(self.seq_lengths)
        A = actions[0].shape[-1]
        P = proprios[0].shape[-1]
        # pad to (N, Tmax, dim) so indexing matches PushTDataset convention
        self.actions = torch.zeros(len(demos), Tmax, A)
        self.proprios = torch.zeros(len(demos), Tmax, P)
        self.states = torch.zeros(len(demos), Tmax, P)
        for i, (a, pr, st) in enumerate(zip(actions, proprios, states)):
            T = a.shape[0]
            self.actions[i, :T] = torch.from_numpy(a)
            self.proprios[i, :T] = torch.from_numpy(pr)
            self.states[i, :T] = torch.from_numpy(st)
        self.actions = self.actions / action_scale

        self.action_dim = A
        self.proprio_dim = P
        self.state_dim = P

        # normalization stats computed over valid (unpadded) timesteps
        self.action_mean, self.action_std = self._stats(self.actions)
        self.proprio_mean, self.proprio_std = self._stats(self.proprios)
        if not normalize_action:
            self.action_mean = torch.zeros(A); self.action_std = torch.ones(A)
        self.actions = (self.actions - self.action_mean) / self.action_std
        self.proprios = (self.proprios - self.proprio_mean) / self.proprio_std
        print(f"[LiftDataset] {len(demos)} demos | img {self.image_hw} | "
              f"A={A} P={P} | filter={filter_key}")

    def _stats(self, padded):
        rows = [padded[i, :T] for i, T in enumerate(self.seq_lengths)]
        cat = torch.cat(rows, dim=0)
        return cat.mean(0), cat.std(0).clamp_min(1e-6)

    def get_seq_length(self, idx):
        return self.seq_lengths[idx]

    def get_all_actions(self):
        return torch.cat([self.actions[i, :T] for i, T in enumerate(self.seq_lengths)], 0)

    def get_frames(self, idx, frames):
        frames = list(frames)
        with h5py.File(self.data_path, "r") as f:
            root = f["data"] if "data" in f else f
            og = _obs_group(root, self.demos[idx])
            imgs = np.asarray(og[self.image_key][frames])          # (T,H,W,C) uint8
        image = torch.from_numpy(imgs).float() / 255.0
        image = rearrange(image, "t h w c -> t c h w")             # (T,C,H,W) in [0,1]
        if self.transform:
            image = self.transform(image)
        obs = {"visual": image, "proprio": self.proprios[idx, frames]}
        return obs, self.actions[idx, frames], self.states[idx, frames], {"shape": "T"}

    def __getitem__(self, idx):
        return self.get_frames(idx, range(self.get_seq_length(idx)))

    def __len__(self):
        return len(self.demos)

    def preprocess_imgs(self, imgs):
        if isinstance(imgs, torch.Tensor):
            return rearrange(imgs, "b h w c -> b c h w") / 255.0
        raise NotImplementedError


def load_lift_slice_train_val(
    transform,
    n_rollout=None,
    data_path="data/lift_image.hdf5",
    normalize_action=True,
    split_ratio=0.9,                     # unused: split comes from mask/ keys
    num_hist=0,
    num_pred=0,
    frameskip=0,
    train_filter_key="train",
    val_filter_key="test",
    image_key="agentview_image",
    proprio_keys=PROPRIO_KEYS,
    with_velocity=False,                 # parity with other loaders; ignored
):
    """Factory matching datasets.pusht_dset.load_pusht_slice_train_val.
    Splits via robomimic mask/<key> filter keys rather than a random ratio."""
    train_dset = LiftDataset(
        data_path=data_path, filter_key=train_filter_key, image_key=image_key,
        proprio_keys=proprio_keys, transform=transform, n_rollout=n_rollout,
        normalize_action=normalize_action,
    )
    val_dset = LiftDataset(
        data_path=data_path, filter_key=val_filter_key, image_key=image_key,
        proprio_keys=proprio_keys, transform=transform, n_rollout=n_rollout,
        normalize_action=normalize_action,
    )
    num_frames = num_hist + num_pred
    train_slices = TrajSlicerDataset(train_dset, num_frames, frameskip)
    val_slices = TrajSlicerDataset(val_dset, num_frames, frameskip)
    datasets = {"train": train_slices, "valid": val_slices}
    traj_dset = {"train": train_dset, "valid": val_dset}
    return datasets, traj_dset
