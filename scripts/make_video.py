import h5py
import numpy as np
import cv2
import os
import random

data_dir = "tactile_benchmark/data/recordings/processed/pnp_tennis_ball_v2_unwrapped"
episodes = [f for f in os.listdir(data_dir) if f.endswith(".hdf5")]
episode = random.choice(episodes)
print(f"Generating video for {episode}")

with h5py.File(os.path.join(data_dir, episode), "r") as f:
    cam0 = f["observation.image.cam0"][:]
    cam1 = f["observation.image.cam1"][:]

out = cv2.VideoWriter("v2_episode_sample.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (cam0.shape[2] + cam1.shape[2], cam0.shape[1]))

for i in range(len(cam0)):
    frame0 = cv2.cvtColor(cam0[i], cv2.COLOR_RGB2BGR)
    frame1 = cv2.cvtColor(cam1[i], cv2.COLOR_RGB2BGR)
    frame = np.concatenate((frame0, frame1), axis=1)
    out.write(frame)
    
out.release()
print("Saved to v2_episode_sample.mp4")
