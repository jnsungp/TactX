"""Fix the 2pi action offset in processed HDF5 episodes.

Reason: for dai_dai_small_ball_pnp, the gello agent's init-time 2pi unwrap
did not run at collection time, so the saved `action` on arm joints 0..6 is
`state + k*2*pi` for some integer k that's typically ±1 on joints 0,1,2,3,6
and 0 on joints 4,5. This script applies the same unwrap teleop applies:

    action[:, j] -= 2*pi * round((action[:, j] - state[:, j]) / (2*pi))

for j in 0..6. Gripper (j=7) and padded dims (>=8) are untouched.

The script does NOT modify the source folder. It copies each episode file to
`<parent>/processed_unwrapped/` and rewrites only the `action` dataset in
the copy.
"""

import argparse
import os
import glob
import shutil
import numpy as np
import h5py


def unwrap_episode(src_path: str, dst_path: str, num_arm_joints: int = 7) -> dict:
    shutil.copy2(src_path, dst_path)
    with h5py.File(dst_path, "r+") as f:
        action = f["action"][...]
        state = f["observation.state"][...]

        arm_action = action[:, :num_arm_joints].copy()
        arm_state = state[:, :num_arm_joints]

        diff = arm_action - arm_state
        k = np.round(diff / (2 * np.pi)).astype(np.int64)
        arm_action = arm_action - 2 * np.pi * k

        action[:, :num_arm_joints] = arm_action

        del f["action"]
        f.create_dataset(
            "action",
            data=action,
            dtype=np.float32,
            compression="gzip",
            compression_opts=4,
        )

        k_per_joint_counts = [
            dict(zip(*np.unique(k[:, j], return_counts=True))) for j in range(num_arm_joints)
        ]

    return {
        "k_per_joint_counts": k_per_joint_counts,
        "T": int(arm_action.shape[0]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--src",
        default="/media/ExtHDD/sachin/teleop/dai_dai_small_ball_pnp/processed",
        help="source processed folder containing episode_*.hdf5",
    )
    ap.add_argument(
        "--dst",
        default=None,
        help="destination folder (default: <src>/../processed_unwrapped)",
    )
    ap.add_argument("--num-arm-joints", type=int, default=7)
    args = ap.parse_args()

    src = os.path.abspath(args.src)
    dst = args.dst or os.path.join(os.path.dirname(src), "processed_unwrapped")
    os.makedirs(dst, exist_ok=True)

    files = sorted(glob.glob(os.path.join(src, "episode_*.hdf5")))
    if not files:
        raise SystemExit(f"No episode_*.hdf5 files found in {src}")

    print(f"Source: {src}")
    print(f"Dest:   {dst}")
    print(f"Episodes: {len(files)}")

    summary = {j: {} for j in range(args.num_arm_joints)}
    for i, src_fp in enumerate(files):
        name = os.path.basename(src_fp)
        dst_fp = os.path.join(dst, name)
        info = unwrap_episode(src_fp, dst_fp, args.num_arm_joints)
        for j, d in enumerate(info["k_per_joint_counts"]):
            for k_val, c in d.items():
                summary[j][int(k_val)] = summary[j].get(int(k_val), 0) + int(c)
        print(f"  [{i+1:3d}/{len(files)}] {name}  T={info['T']}  k_per_joint={[max(d, key=d.get) for d in info['k_per_joint_counts']]}")

    print("\nOverall k (unwrap steps) distribution across ALL frames:")
    for j in range(args.num_arm_joints):
        items = sorted(summary[j].items())
        print(f"  joint {j}: {items}")


if __name__ == "__main__":
    main()
