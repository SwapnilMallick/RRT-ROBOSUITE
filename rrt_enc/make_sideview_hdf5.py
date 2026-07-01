"""
Create datasets/lift/ph/image224_sideview.hdf5 by replaying every demo in
image224.hdf5 with the MuJoCo sideview camera.

Structure is identical to the source file except:
  - obs/agentview_image        -> obs/sideview_image
  - next_obs/agentview_image   -> next_obs/sideview_image
  - demo attrs: camera_info updated to sideview intrinsics/extrinsics
  - data attrs: env_args camera_names updated to ["sideview"]

All other datasets (actions, dones, rewards, states, robot obs, object obs)
are copied byte-for-byte from the source. The mask group is copied as-is.
"""

import argparse
import json
import os
import warnings

import h5py
import mujoco
import numpy as np
import robosuite as suite

warnings.filterwarnings("ignore")

SRC  = "datasets/lift/ph/image224.hdf5"
DST  = "datasets/lift/ph/image224_sideview.hdf5"
CAM  = "sideview"
HW   = 224

# Keys inside obs / next_obs that are images (will be replaced)
IMAGE_KEYS = {"agentview_image"}


def build_env():
    env = suite.make(
        "Lift",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=False,      # we render manually
        use_object_obs=False,
        camera_names=[CAM],
        camera_heights=HW,
        camera_widths=HW,
        ignore_done=True,
        control_freq=20,
    )
    env.reset()
    return env


def get_sideview_camera_info(env):
    """Extract intrinsics and extrinsics for the sideview camera."""
    sim = env.sim
    cam_id = mujoco.mj_name2id(sim.model._model, mujoco.mjtObj.mjOBJ_CAMERA, CAM)

    # Intrinsics: robosuite uses fovy (vertical FOV in degrees).
    # Default fovy for non-specified cameras is 45 deg in MuJoCo.
    fovy_deg = sim.model.cam_fovy[cam_id]
    fovy_rad = np.deg2rad(fovy_deg)
    fy = (HW / 2.0) / np.tan(fovy_rad / 2.0)
    fx = fy                              # square pixels
    cx = cy = (HW - 1) / 2.0
    intrinsics = [[fx, 0.0, cx],
                  [0.0, fy, cy],
                  [0.0, 0.0, 1.0]]

    # Extrinsics: 4×4 camera-to-world matrix from MuJoCo cam_xpos / cam_xmat.
    mujoco.mj_forward(sim.model._model, sim.data._data)
    pos  = np.array(sim.data.cam_xpos[cam_id])        # (3,)
    rmat = np.array(sim.data.cam_xmat[cam_id]).reshape(3, 3)
    extrinsics = np.eye(4)
    extrinsics[:3, :3] = rmat
    extrinsics[:3,  3] = pos
    return {"intrinsics": intrinsics, "extrinsics": extrinsics.tolist()}


def render_frame(env, state_flat):
    """Set sim state and return sideview render (H,W,3) uint8, top-up."""
    env.sim.set_state_from_flattened(state_flat)
    env.sim.forward()
    img = env.sim.render(width=HW, height=HW, camera_name=CAM)
    return img[::-1].copy()            # robosuite renders bottom-up -> flip


def copy_non_image_datasets(src_group, dst_group):
    """Recursively copy every dataset that is NOT an image key."""
    for key in src_group.keys():
        item = src_group[key]
        if isinstance(item, h5py.Group):
            grp = dst_group.require_group(key)
            copy_non_image_datasets(item, grp)
        elif isinstance(item, h5py.Dataset):
            if key not in IMAGE_KEYS:
                dst_group.create_dataset(key, data=item[:],
                                         compression=item.compression,
                                         compression_opts=item.compression_opts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=SRC)
    p.add_argument("--dst", default=DST)
    args = p.parse_args()

    print("Building robosuite env …")
    env = build_env()
    cam_info = get_sideview_camera_info(env)
    print(f"sideview fovy: {env.sim.model.cam_fovy[mujoco.mj_name2id(env.sim.model._model, mujoco.mjtObj.mjOBJ_CAMERA, CAM)]:.1f}°")

    with h5py.File(args.src, "r") as src, h5py.File(args.dst, "w") as dst:
        # ------------------------------------------------------------------ #
        # Top-level attrs (none in this file, but copy just in case)
        for k, v in src.attrs.items():
            dst.attrs[k] = v

        # ------------------------------------------------------------------ #
        # data group attrs (env_args with camera_names patched)
        data_grp = dst.require_group("data")
        for k, v in src["data"].attrs.items():
            if k == "env_args":
                ea = json.loads(v)
                ea["env_kwargs"]["camera_names"] = [CAM]
                data_grp.attrs[k] = json.dumps(ea, indent=4)
            else:
                data_grp.attrs[k] = v

        # ------------------------------------------------------------------ #
        # mask group — copy byte-for-byte
        if "mask" in src:
            mask_grp = dst.require_group("mask")
            for k in src["mask"].keys():
                mask_grp.create_dataset(k, data=src["mask"][k][:])

        # ------------------------------------------------------------------ #
        # demos
        all_demos = sorted(src["data"].keys(),
                           key=lambda x: int(x.split("_")[1]))
        n = len(all_demos)
        print(f"Processing {n} demos …\n")

        for di, demo in enumerate(all_demos):
            src_demo = src["data"][demo]
            states   = np.asarray(src_demo["states"])       # (T, 32)
            T        = states.shape[0]

            # Render obs frames: state[t] -> obs[t]
            obs_frames = np.empty((T, HW, HW, 3), dtype=np.uint8)
            for t in range(T):
                obs_frames[t] = render_frame(env, states[t])

            # Render next_obs frames: state[t+1] -> next_obs[t];
            # for the last step repeat the final frame (terminal state).
            nxt_frames = np.empty((T, HW, HW, 3), dtype=np.uint8)
            nxt_frames[:-1] = obs_frames[1:]
            nxt_frames[-1]  = obs_frames[-1]

            # Create demo group and copy non-image data
            dst_demo = dst["data"].require_group(demo)
            copy_non_image_datasets(src_demo, dst_demo)

            # Write sideview images
            dst_demo["obs"].create_dataset(
                "sideview_image", data=obs_frames, compression="gzip", compression_opts=4)
            dst_demo["next_obs"].create_dataset(
                "sideview_image", data=nxt_frames, compression="gzip", compression_opts=4)

            # Copy demo attrs and patch camera_info
            for k, v in src_demo.attrs.items():
                if k == "camera_info":
                    dst_demo.attrs[k] = json.dumps({CAM: cam_info}, indent=4)
                else:
                    dst_demo.attrs[k] = v

            print(f"  [{di+1:3d}/{n}]  {demo}: {T} frames")

    env.close()
    print(f"\nDone → {args.dst}")


if __name__ == "__main__":
    main()
