#!/usr/bin/env python3
"""
Process ACEF episode data from PKL+MP4 format to HDF5 format for ACT training.

Input format (per episode):
  episode_####/
    ├── cam0.mp4 (wrist camera)
    ├── cam1.mp4 (3rd person camera)
    ├── tac0.mp4 (tactile sensor 0) **optional**
    ├── tac1.mp4 (tactile sensor 1) **optional**
    ├── low_dim_obs.pkl (contains state and action)
    └── tactile_data.pkl (contains tactile sensor data at higher frequency)

Output format:
  processed/episode_####.hdf5 with:
    - observation.state: [T, 128] float32 (padded)
    - action: [T, 128] float32 (padded)
    - observation.image.cam0: [T, 240, 320, 3] uint8 or compressed JPEG
    - observation.image.cam1: [T, 240, 320, 3] uint8 or compressed JPEG
    - observation.image.tac0: [T, 240, 320, 3] uint8 or compressed JPEG
    - observation.image.tac1: [T, 240, 320, 3] uint8 or compressed JPEG
    - observation.tac0: [T, 384] float32 (flattened tactile sensor 0)
    - observation.tac1: [T, 384] float32 (flattened tactile sensor 1)
    - attrs: {'description': str, 'embodiment': str,  etc.}

Usage:
    python process_acef_data.py --input_dir /path/to/data 

    This will create: /path/to/data/processed/episode_####.hdf5 files
"""

import os
import pickle
import h5py
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import argparse


def extract_frames_from_video(video_path, target_num_frames=None, target_resolution=(240, 320)):
    """
    Extract frames from MP4 video.

    Args:
        video_path: Path to MP4 file
        target_num_frames: Expected number of frames (for validation)
        target_resolution: (height, width) tuple for resizing

    Returns:
        numpy array of shape [T, H, W, 3] in RGB format
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (target_resolution[1], target_resolution[0]))
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        raise ValueError(f"No frames extracted from {video_path}")

    if target_num_frames is not None and len(frames) != target_num_frames:
        print(f"  Warning: Expected {target_num_frames} frames but got {len(frames)} from {video_path}")

    return np.array(frames, dtype=np.uint8)

def resample_data_at_frequency(low_dim_data, tactile_data, target_freq):
    low_dim_timestamps = np.array([entry['state_timestamp'] for entry in low_dim_data])
    tactile_timestamps = tactile_data['timestamps']

    start_time = low_dim_timestamps[0]
    end_time = low_dim_timestamps[-1]
    duration = end_time - start_time

    num_samples = int(duration * target_freq) + 1
    target_timestamps = np.linspace(start_time, end_time, num_samples)

    states = np.array([entry['state'] for entry in low_dim_data], dtype=np.float32)
    actions = np.array([entry['action'] for entry in low_dim_data], dtype=np.float32)

    resampled_states = []
    resampled_actions = []
    resampled_tac0 = []
    resampled_tac1 = []

    for target_ts in target_timestamps:
        low_dim_idx = np.argmin(np.abs(low_dim_timestamps - target_ts))
        resampled_states.append(states[low_dim_idx])
        resampled_actions.append(actions[low_dim_idx])

        tactile_idx = np.argmin(np.abs(tactile_timestamps - target_ts))
        resampled_tac0.append(tactile_data['s0'][tactile_idx].flatten())
        resampled_tac1.append(tactile_data['s1'][tactile_idx].flatten())

    return (np.array(resampled_states, dtype=np.float32),
            np.array(resampled_actions, dtype=np.float32),
            np.array(resampled_tac0, dtype=np.float32),
            np.array(resampled_tac1, dtype=np.float32))


def pad_to_dim(array, target_dim=128):
    current_dim = array.shape[1]
    if current_dim >= target_dim:
        raise ValueError(f"Array dimension {current_dim} >= target dimension {target_dim}")

    pad_width = target_dim - current_dim
    padded = np.pad(array, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
    return padded.astype(np.float32)


def convert_episode_to_hdf5(episode_dir,
                            output_path, task_description="franka manipulation task",
                            embodiment="franka",
                            compress_images=False,
                            target_resolution=(240, 320), pad_dim=128,
                            tactile_freq=None,
                            allow_missing_tactile=False
                            ):
    episode_dir = Path(episode_dir)
    low_dim_path = episode_dir / "low_dim_obs.pkl"
    tactile_path = episode_dir / "tactile_data.pkl"

    if not low_dim_path.exists():
        raise FileNotFoundError(f"PKL file not found: {low_dim_path}")

    # Handle missing tactile data
    tactile_data_missing = not tactile_path.exists()
    if tactile_data_missing and not allow_missing_tactile:
        raise FileNotFoundError(f"Tactile data file not found: {tactile_path}")

    with open(low_dim_path, 'rb') as f:
        low_dim_data = pickle.load(f)

    if not tactile_data_missing:
        with open(tactile_path, 'rb') as f:
            tactile_data = pickle.load(f)
    else:
        # Create empty tactile data structure
        tactile_data = None

    low_dim_timestamps = np.array([entry['state_timestamp'] for entry in low_dim_data])
    low_dim_freq = 1.0 / np.mean(np.diff(low_dim_timestamps))

    if tactile_freq is None:
        tactile_freq = low_dim_freq

    if tactile_data_missing:
        print(f"Converting {episode_dir.name}: low_dim_freq={low_dim_freq:.1f}Hz, target_freq={tactile_freq}Hz (NO TACTILE DATA)")
        # Use empty 1D tactile arrays when tactile data is missing
        states = np.array([entry['state'] for entry in low_dim_data], dtype=np.float32)
        actions = np.array([entry['action'] for entry in low_dim_data], dtype=np.float32)
        num_timesteps = len(states)
        tac0 = np.zeros((num_timesteps, 1), dtype=np.float32)
        tac1 = np.zeros((num_timesteps, 1), dtype=np.float32)
        tactile_dim = 1
    else:
        print(f"Converting {episode_dir.name}: low_dim_freq={low_dim_freq:.1f}Hz, target_freq={tactile_freq}Hz")
        states, actions, tac0, tac1 = resample_data_at_frequency(low_dim_data, tactile_data, tactile_freq)
        num_timesteps = len(states)
        tactile_dim = tac0.shape[1]

    original_state_dim = states.shape[1]
    original_action_dim = actions.shape[1]

    print(f"  Resampled - State: {states.shape}, Action: {actions.shape}, Tac0: {tac0.shape}, Tac1: {tac1.shape}")

    states = pad_to_dim(states, target_dim=pad_dim)
    actions = pad_to_dim(actions, target_dim=pad_dim)

    cam0_path = episode_dir / "cam0.mp4"
    cam1_path = episode_dir / "cam1.mp4"
    tac0_img_path = episode_dir / "tac0.mp4"
    tac1_img_path = episode_dir / "tac1.mp4"

    if not cam0_path.exists() or not cam1_path.exists():
        raise FileNotFoundError(f"Video files not found in {episode_dir}")

    cam0_frames = extract_frames_from_video(cam0_path, len(low_dim_data), target_resolution)
    cam1_frames = extract_frames_from_video(cam1_path, len(low_dim_data), target_resolution)

    if tac0_img_path.exists() and tac1_img_path.exists():
        tac0_frames = extract_frames_from_video(tac0_img_path, len(low_dim_data), target_resolution)
        tac1_frames = extract_frames_from_video(tac1_img_path, len(low_dim_data), target_resolution)
    else:
        tac0_frames = np.zeros((len(low_dim_data), target_resolution[0], target_resolution[1], 3), dtype=np.uint8)
        tac1_frames = np.zeros((len(low_dim_data), target_resolution[0], target_resolution[1], 3), dtype=np.uint8)

    start_time = low_dim_timestamps[0]
    end_time = low_dim_timestamps[-1]
    duration = end_time - start_time
    target_timestamps = np.linspace(start_time, end_time, num_timesteps)

    cam0_frames_resampled = []
    cam1_frames_resampled = []
    tac0_frames_resampled = []
    tac1_frames_resampled = []

    for target_ts in target_timestamps:
        idx = np.argmin(np.abs(low_dim_timestamps - target_ts))
        if idx < len(cam0_frames):
            cam0_frames_resampled.append(cam0_frames[idx])
            cam1_frames_resampled.append(cam1_frames[idx])
            tac0_frames_resampled.append(tac0_frames[idx])
            tac1_frames_resampled.append(tac1_frames[idx])

    cam0_frames = np.array(cam0_frames_resampled, dtype=np.uint8)
    cam1_frames = np.array(cam1_frames_resampled, dtype=np.uint8)
    tac0_frames = np.array(tac0_frames_resampled, dtype=np.uint8)
    tac1_frames = np.array(tac1_frames_resampled, dtype=np.uint8)

    min_frames = min(len(cam0_frames), num_timesteps)
    if min_frames < num_timesteps:
        print(f"  Warning: Trimming to {min_frames} frames due to video/data mismatch")
        cam0_frames = cam0_frames[:min_frames]
        cam1_frames = cam1_frames[:min_frames]
        tac0_frames = tac0_frames[:min_frames]
        tac1_frames = tac1_frames[:min_frames]
        states = states[:min_frames]
        actions = actions[:min_frames]
        tac0 = tac0[:min_frames]
        tac1 = tac1[:min_frames]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with h5py.File(output_path, 'w') as f:
        # Save padded state and action
        f.create_dataset('observation.state', data=states, compression='gzip')
        f.create_dataset('observation.tac0', data=tac0, compression='gzip')
        f.create_dataset('observation.tac1', data=tac1, compression='gzip')
        f.create_dataset('action', data=actions, compression='gzip')

        if compress_images:
            cam0_compressed = []
            cam1_compressed = []
            tac0_compressed = []
            tac1_compressed = []
            for i in range(len(cam0_frames)):
                # JPEG encode
                _, cam0_jpg = cv2.imencode('.jpg', cv2.cvtColor(cam0_frames[i], cv2.COLOR_RGB2BGR),
                                           [cv2.IMWRITE_JPEG_QUALITY, 90])
                _, cam1_jpg = cv2.imencode('.jpg', cv2.cvtColor(cam1_frames[i], cv2.COLOR_RGB2BGR),
                                           [cv2.IMWRITE_JPEG_QUALITY, 90])
                _, tac0_jpg = cv2.imencode('.jpg', cv2.cvtColor(tac0_frames[i], cv2.COLOR_RGB2BGR),
                                             [cv2.IMWRITE_JPEG_QUALITY, 90])
                _, tac1_jpg = cv2.imencode('.jpg', cv2.cvtColor(tac1_frames[i], cv2.COLOR_RGB2BGR),
                                             [cv2.IMWRITE_JPEG_QUALITY, 90])
                cam0_compressed.append(cam0_jpg.tobytes())
                cam1_compressed.append(cam1_jpg.tobytes())
                tac0_compressed.append(tac0_jpg.tobytes())
                tac1_compressed.append(tac1_jpg.tobytes())
            dt = h5py.vlen_dtype(np.dtype('uint8'))
            f.create_dataset('observation.image.cam0', data=cam0_compressed, dtype=dt)
            f.create_dataset('observation.image.cam1', data=cam1_compressed, dtype=dt)
            f.create_dataset('observation.image.tac0', data=tac0_compressed, dtype=dt)
            f.create_dataset('observation.image.tac1', data=tac1_compressed, dtype=dt)
        else:
            f.create_dataset('observation.image.cam0', data=cam0_frames, compression='gzip')
            f.create_dataset('observation.image.cam1', data=cam1_frames, compression='gzip')
            f.create_dataset('observation.image.tac0', data=tac0_frames, compression='gzip')
            f.create_dataset('observation.image.tac1', data=tac1_frames, compression='gzip')

        # Save attributes (metadata)
        f.attrs['description'] = task_description
        f.attrs['embodiment'] = embodiment
        f.attrs['original_state_dim'] = original_state_dim
        f.attrs['original_action_dim'] = original_action_dim
        f.attrs['tactile_dim'] = tactile_dim
        f.attrs['padded_state_dim'] = pad_dim
        f.attrs['padded_action_dim'] = pad_dim

    print(f"  ✓ Saved to {output_path}")
    print(f"    State: {states.shape}, Action: {actions.shape}")
    print(f"    Cam0: {cam0_frames.shape}, Cam1: {cam1_frames.shape}")
    print(f"    Img_Tac0: {tac0_frames.shape}, Img_Tac1: {tac1_frames.shape}")
    print(f"    Tac0: {tac0.shape}, Tac1: {tac1.shape}")


def convert_all_episodes(input_dir, task_description="franka manipulation task",
                        embodiment="franka", compress_images=False, pad_dim=128, tactile_freq=None,
                        allow_missing_tactile=False):
    input_dir = Path(input_dir)
    output_dir = input_dir / "processed"
    output_dir.mkdir(exist_ok=True)
    episode_dirs = sorted([d for d in input_dir.iterdir()
                          if d.is_dir() and d.name.startswith('episode_')])

    if not episode_dirs:
        print(f"ERROR: No episode directories found in {input_dir}")
        print(f"Looking for directories matching pattern: episode_*")
        return

    print(f"Found {len(episode_dirs)} episodes to convert")
    print(f"Output directory: {output_dir}")

    successful = 0
    failed = 0

    for episode_dir in tqdm(episode_dirs, desc="Converting episodes"):
        episode_name = episode_dir.name
        output_path = output_dir / f"{episode_name}.hdf5"

        try:
            convert_episode_to_hdf5(
                episode_dir,
                output_path,
                task_description=task_description,
                embodiment=embodiment,
                compress_images=compress_images,
                pad_dim=pad_dim,
                tactile_freq=tactile_freq,
                allow_missing_tactile=allow_missing_tactile
            )
            successful += 1
        except Exception as e:
            print(f"\n✗ Error converting {episode_name}: {e}")
            failed += 1
            continue

    print(f"\n{'='*60}")
    print(f"Conversion complete!")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Convert ACEF episodes to HDF5 for ACT training")
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Directory containing episode_#### folders")
    parser.add_argument("--task_description", type=str,
                       default="franka manipulation task",
                       help="Language description of the task")
    parser.add_argument("--embodiment", type=str, default="franka",
                       help="Robot embodiment identifier")
    parser.add_argument("--compress_images", action="store_true",
                       help="Compress images as JPEG (saves disk space)")
    parser.add_argument("--single_episode", type=str,
                       help="Convert only a single episode (path to episode_#### folder)")
    parser.add_argument("--resolution", type=str, default="240,320",
                       help="Target resolution as 'height,width' (default: 240,320)")
    parser.add_argument("--pad_dim", type=int, default=128,
                       help="Target dimension for padding state and action (default: 128)")
    parser.add_argument("--tactile_freq", type=float, default=None,
                       help="Target frequency (Hz) for tactile data sampling. Default: match low_dim frequency (~30Hz). Can be set between low_dim freq and tactile sensor freq (~100Hz)")
    parser.add_argument("--allow_missing_tactile", action="store_true",
                       help="Allow processing episodes without tactile_data.pkl by using empty 1D arrays as placeholders")

    args = parser.parse_args()

    # Parse resolution
    try:
        h, w = map(int, args.resolution.split(','))
        target_resolution = (h, w)
    except:
        print(f"Error: Invalid resolution format '{args.resolution}'. Use 'height,width' (e.g., 240,320)")
        return

    if args.single_episode:
        episode_dir = Path(args.single_episode)
        input_dir = episode_dir.parent
        output_dir = input_dir / "processed"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"{episode_dir.name}.hdf5"

        convert_episode_to_hdf5(
            episode_dir,
            output_path,
            task_description=args.task_description,
            embodiment=args.embodiment,
            compress_images=args.compress_images,
            target_resolution=target_resolution,
            pad_dim=args.pad_dim,
            tactile_freq=args.tactile_freq,
            allow_missing_tactile=args.allow_missing_tactile
        )
        print(f"\n✓ Single episode converted successfully: {output_path}")
    else:
        convert_all_episodes(
            args.input_dir,
            task_description=args.task_description,
            embodiment=args.embodiment,
            compress_images=args.compress_images,
            pad_dim=args.pad_dim,
            tactile_freq=args.tactile_freq,
            allow_missing_tactile=args.allow_missing_tactile
        )


if __name__ == "__main__":
    main()
