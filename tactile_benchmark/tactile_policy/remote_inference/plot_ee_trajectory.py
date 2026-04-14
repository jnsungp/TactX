import h5py
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import argparse
import websocket_client_policy


def franka_forward_kinematics(q):
    """
    Compute forward kinematics for Franka Panda robot.

    Args:
        q: Joint angles (7,) - only first 7 values used if longer

    Returns:
        ee_pos: End-effector position (x, y, z)
    """
    q = q[:7]  # Only use first 7 joints

    # Franka Panda DH parameters (modified DH convention)
    d = [0.333, 0, 0.316, 0, 0.384, 0, 0.107]
    a = [0, 0, 0.0825, -0.0825, 0, 0.088, 0]
    alpha = [0, -np.pi/2, np.pi/2, np.pi/2, -np.pi/2, np.pi/2, np.pi/2]

    def dh_transform(theta, d, a, alpha):
        """Compute transformation matrix using modified DH parameters."""
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)
        return np.array([
            [ct, -st, 0, a],
            [st*ca, ct*ca, -sa, -d*sa],
            [st*sa, ct*ca, ca, d*ca],
            [0, 0, 0, 1]
        ])

    T = np.eye(4)
    for i in range(7):
        T = T @ dh_transform(q[i], d[i], a[i], alpha[i])

    # Flange to EE offset (default Franka hand)
    T_flange_ee = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0.1034],
        [0, 0, 0, 1]
    ])
    T = T @ T_flange_ee

    return T[:3, 3]


def load_tactile_data(data, tactile_names, tactile_modality):
    """Load tactile data from HDF5 file based on modality."""
    tactile_images = None
    tactile_arrays = None

    if not tactile_names:
        return tactile_images, tactile_arrays

    if tactile_modality == 'image':
        tactile_img_list = []
        for tac_name in tactile_names:
            key = f'observation.image.{tac_name}'
            if key in data:
                tac_imgs = np.array(data[key])
                if len(tac_imgs.shape) == 2:
                    decoded = []
                    for i in range(tac_imgs.shape[0]):
                        img = cv2.imdecode(tac_imgs[i], cv2.IMREAD_COLOR)
                        decoded.append(img.transpose((2, 0, 1)))
                    tac_imgs = np.stack(decoded, axis=0)
                elif len(tac_imgs.shape) == 4 and tac_imgs.shape[-1] == 3:
                    tac_imgs = tac_imgs.transpose((0, 3, 1, 2))
                tactile_img_list.append(tac_imgs)
        if tactile_img_list:
            tactile_images = np.stack(tactile_img_list, axis=1)

    elif tactile_modality in ['array', 'state']:
        tactile_arr_list = []
        for tac_name in tactile_names:
            key = f'observation.{tac_name}'
            if key in data:
                tac_arr = np.array(data[key])
                tactile_arr_list.append(tac_arr)
        if tactile_arr_list:
            tactile_arrays = np.stack(tactile_arr_list, axis=1)

    return tactile_images, tactile_arrays


def plot_ee_trajectories(gt_ee_positions, pred_ee_positions, save_path="ee_trajectory_3d.png"):
    """
    Plot 3D trajectories of ground truth and predicted end-effector positions.

    Args:
        gt_ee_positions: Ground truth EE positions (N, 3)
        pred_ee_positions: Predicted EE positions (N, 3)
        save_path: Path to save the figure
    """
    gt_ee_positions = np.array(gt_ee_positions)
    pred_ee_positions = np.array(pred_ee_positions)

    fig = plt.figure(figsize=(14, 5))

    # 3D trajectory plot
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot(gt_ee_positions[:, 0], gt_ee_positions[:, 1], gt_ee_positions[:, 2],
             'b-', label='Ground Truth', linewidth=2)
    ax1.plot(pred_ee_positions[:, 0], pred_ee_positions[:, 1], pred_ee_positions[:, 2],
             'r--', label='Predicted', linewidth=2, alpha=0.8)
    ax1.scatter(gt_ee_positions[0, 0], gt_ee_positions[0, 1], gt_ee_positions[0, 2],
                c='green', s=100, marker='o', label='Start')
    ax1.scatter(gt_ee_positions[-1, 0], gt_ee_positions[-1, 1], gt_ee_positions[-1, 2],
                c='purple', s=100, marker='x', label='End (GT)')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D End-Effector Trajectory')
    ax1.legend()

    # XY projection
    ax2 = fig.add_subplot(132)
    ax2.plot(gt_ee_positions[:, 0], gt_ee_positions[:, 1], 'b-', label='Ground Truth', linewidth=2)
    ax2.plot(pred_ee_positions[:, 0], pred_ee_positions[:, 1], 'r--', label='Predicted', linewidth=2, alpha=0.8)
    ax2.scatter(gt_ee_positions[0, 0], gt_ee_positions[0, 1], c='green', s=100, marker='o')
    ax2.scatter(gt_ee_positions[-1, 0], gt_ee_positions[-1, 1], c='purple', s=100, marker='x')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('XY Projection')
    ax2.legend()
    ax2.axis('equal')
    ax2.grid(True, alpha=0.3)

    # XZ projection
    ax3 = fig.add_subplot(133)
    ax3.plot(gt_ee_positions[:, 0], gt_ee_positions[:, 2], 'b-', label='Ground Truth', linewidth=2)
    ax3.plot(pred_ee_positions[:, 0], pred_ee_positions[:, 2], 'r--', label='Predicted', linewidth=2, alpha=0.8)
    ax3.scatter(gt_ee_positions[0, 0], gt_ee_positions[0, 2], c='green', s=100, marker='o')
    ax3.scatter(gt_ee_positions[-1, 0], gt_ee_positions[-1, 2], c='purple', s=100, marker='x')
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Z (m)')
    ax3.set_title('XZ Projection (Side View)')
    ax3.legend()
    ax3.axis('equal')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved trajectory plot to {save_path}")

    # Also plot position error over time
    fig2, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Per-axis comparison
    labels = ['X', 'Y', 'Z']
    for i, (ax, label) in enumerate(zip(axes.flat[:3], labels)):
        ax.plot(gt_ee_positions[:, i], 'b-', label='Ground Truth', linewidth=2)
        ax.plot(pred_ee_positions[:, i], 'r--', label='Predicted', linewidth=2, alpha=0.8)
        ax.set_xlabel('Time Step')
        ax.set_ylabel(f'{label} Position (m)')
        ax.set_title(f'{label} Position over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Position error
    error = np.linalg.norm(gt_ee_positions - pred_ee_positions, axis=1)
    axes[1, 1].plot(error * 1000, 'g-', linewidth=2)  # Convert to mm
    axes[1, 1].set_xlabel('Time Step')
    axes[1, 1].set_ylabel('Position Error (mm)')
    axes[1, 1].set_title(f'EE Position Error (Mean: {np.mean(error)*1000:.2f} mm)')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    error_save_path = save_path.replace('.png', '_error.png')
    plt.savefig(error_save_path, dpi=150)
    print(f"Saved error plot to {error_save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Plot 3D EE trajectory: predicted vs ground truth')
    parser.add_argument('--hdf_file_path', type=str, required=True, help='HDF5 file path')
    parser.add_argument('--chunk_size', type=int, default=32, help='Chunk size')
    parser.add_argument('--output', type=str, default='ee_trajectory_3d.png', help='Output file path')
    parser.add_argument('--tactile_names', type=str, nargs='*', default=['tac0', 'tac1'])
    parser.add_argument('--tactile_modality', type=str, default='none',
                        choices=['none', 'image', 'array', 'state'])
    parser.add_argument('--host', type=str, default='localhost', help='Websocket host')
    parser.add_argument('--port', type=int, default=8000, help='Websocket port')

    args = parser.parse_args()

    client = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)

    with h5py.File(args.hdf_file_path, 'r') as data:
        actions = np.array(data['action'])
        left_imgs = np.array(data['observation.image.cam0'])
        right_imgs = np.array(data['observation.image.cam1'])
        states = np.array(data['observation.state'])
        plain_task_text = data.attrs['description']

        print(f"Actions shape: {actions.shape}")
        print(f"States shape: {states.shape}")

        if len(left_imgs.shape) == 2:
            left_img_list = []
            right_img_list = []
            for i in range(left_imgs.shape[0]):
                left_img = cv2.imdecode(left_imgs[i], cv2.IMREAD_COLOR)
                right_img = cv2.imdecode(right_imgs[i], cv2.IMREAD_COLOR)
                left_img_list.append(left_img.transpose((2, 0, 1)))
                right_img_list.append(right_img.transpose((2, 0, 1)))
            left_imgs = np.stack(left_img_list, axis=0)
            right_imgs = np.stack(right_img_list, axis=0)
        elif len(left_imgs.shape) == 4 and left_imgs.shape[-1] == 3:
            left_imgs = left_imgs.transpose((0, 3, 1, 2))
            right_imgs = right_imgs.transpose((0, 3, 1, 2))

        all_tactile_images, all_tactile_arrays = load_tactile_data(
            data, args.tactile_names, args.tactile_modality)

    # Run inference and collect EE positions
    output = None
    act_index = 0
    chunk_size = args.chunk_size

    gt_ee_positions = []
    pred_ee_positions = []

    for t in tqdm(range(states.shape[0]), desc="Running inference"):
        cur_state = states[t]
        cur_action = actions[t]
        cur_left_img = left_imgs[t]
        cur_right_img = right_imgs[t]
        cur_img = np.stack([cur_left_img, cur_right_img], axis=0)

        if output is None or act_index == chunk_size - 10:
            obs = {
                "image": cur_img.astype(np.uint8)[None, ...],
                "qpos": cur_state.astype(np.float32).reshape(1, -1),
                'cond_dict': {
                    "plain_text": [plain_task_text]
                }
            }

            if all_tactile_images is not None:
                obs['tactile_images'] = all_tactile_images[t:t+1].astype(np.uint8)
            if all_tactile_arrays is not None:
                obs['tactile_arrays'] = all_tactile_arrays[t:t+1].astype(np.float32)

            output = client.infer(obs)['action'][0]
            act_index = 0

        pred_action = output[act_index]
        act_index += 1

        # Compute EE positions using forward kinematics
        gt_ee = franka_forward_kinematics(cur_action)
        pred_ee = franka_forward_kinematics(pred_action)

        gt_ee_positions.append(gt_ee)
        pred_ee_positions.append(pred_ee)

    # Plot the trajectories
    plot_ee_trajectories(gt_ee_positions, pred_ee_positions, args.output)
