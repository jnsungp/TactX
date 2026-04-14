import h5py
import pickle
import torch
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import websocket_client_policy
import yaml


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
                    # Compressed JPEG
                    decoded = []
                    for i in range(tac_imgs.shape[0]):
                        img = cv2.imdecode(tac_imgs[i], cv2.IMREAD_COLOR)
                        decoded.append(img.transpose((2, 0, 1)))
                    tac_imgs = np.stack(decoded, axis=0)
                elif len(tac_imgs.shape) == 4 and tac_imgs.shape[-1] == 3:
                    tac_imgs = tac_imgs.transpose((0, 3, 1, 2))
                tactile_img_list.append(tac_imgs)
        if tactile_img_list:
            # Stack along sensor dimension: (T, num_sensors, C, H, W)
            tactile_images = np.stack(tactile_img_list, axis=1)

    elif tactile_modality in ['array', 'state']:
        tactile_arr_list = []
        for tac_name in tactile_names:
            key = f'observation.{tac_name}'
            if key in data:
                tac_arr = np.array(data[key])
                tactile_arr_list.append(tac_arr)
        if tactile_arr_list:
            # Stack along sensor dimension: (T, num_sensors, tactile_dim)
            tactile_arrays = np.stack(tactile_arr_list, axis=1)

    return tactile_images, tactile_arrays


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Eval policy and plot figures before deploy', add_help=False)
    parser.add_argument('--hdf_file_path', type=str, help='hdf file path', required=True)
    parser.add_argument('--chunk_size', type=int, help='chunk size', default=32)
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--tactile_names', type=str, nargs='*', default=['tac0', 'tac1'], help='tactile sensor names (e.g., tac0 tac1)')
    parser.add_argument('--tactile_modality', type=str, default='none', choices=['none', 'image', 'array', 'state'], help='tactile modality')

    # NOTE: you should change the host and port to the one you are using.
    client = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8000)

    args = vars(parser.parse_args())

    chunk_size = args['chunk_size']
    tactile_names = args['tactile_names']
    tactile_modality = args['tactile_modality']

    with h5py.File(args['hdf_file_path'], 'r') as data:
        actions = np.array(data['action'])
        print(actions.shape)
        print(actions[:5])
        left_imgs = np.array(data['observation.image.cam0'])
        right_imgs = np.array(data['observation.image.cam1'])
        states = np.array(data['observation.state'])
        print(states.shape)
        print(states[:5])
        plain_task_text = data.attrs['description']

        if len(left_imgs.shape) == 2:
            # Compressed JPEG images
            assert len(right_imgs.shape) == 2
            assert left_imgs.shape[0] == right_imgs.shape[0]
            # Decompress
            left_img_list = []
            right_img_list = []
            for i in range(left_imgs.shape[0]):
                left_img = cv2.imdecode(left_imgs[i], cv2.IMREAD_COLOR)
                right_img = cv2.imdecode(right_imgs[i], cv2.IMREAD_COLOR)
                left_img_list.append(left_img.transpose((2, 0, 1)))  # HWC -> CHW
                right_img_list.append(right_img.transpose((2, 0, 1)))
            left_imgs = np.stack(left_img_list, axis=0)
            right_imgs = np.stack(right_img_list, axis=0)
        elif len(left_imgs.shape) == 4 and left_imgs.shape[-1] == 3:
            # Raw images in [T, H, W, C] format - transpose to [T, C, H, W]
            left_imgs = left_imgs.transpose((0, 3, 1, 2))  # [T, H, W, C] -> [T, C, H, W]
            right_imgs = right_imgs.transpose((0, 3, 1, 2))

        # Load tactile data
        all_tactile_images, all_tactile_arrays = load_tactile_data(data, tactile_names, tactile_modality)
        if all_tactile_images is not None:
            print(f"Loaded tactile images: {all_tactile_images.shape}")
        if all_tactile_arrays is not None:
            print(f"Loaded tactile arrays: {all_tactile_arrays.shape}")
        
    output = None
    act = None
    act_index = 0

    if args['plot']:
        predicted_list = []
        gt_list = []
        record_list = []

    for t in tqdm(range(states.shape[0])):
            print("step", t)
            t_start = t
            # Select offseted episodes
            cur_action = actions[t_start]
            cur_left_img = left_imgs[t_start]
            cur_right_img = right_imgs[t_start]
            cur_img = np.stack([cur_left_img, cur_right_img], axis=0)  # (2, 3, H, W)

            cur_state = states[t_start]

            if output is None or act_index == chunk_size - 10:

                obs = {
                    "image": cur_img.astype(np.uint8)[None, ...], # [0, 255]
                    "qpos": cur_state.astype(np.float32).reshape(1, -1),  # (1, 128)
                    'cond_dict': {
                        "plain_text": [plain_task_text]
                    }
                }

                # Add tactile data to observation if available
                if all_tactile_images is not None:
                    obs['tactile_images'] = all_tactile_images[t_start:t_start+1].astype(np.uint8)
                if all_tactile_arrays is not None:
                    obs['tactile_arrays'] = all_tactile_arrays[t_start:t_start+1].astype(np.float32)

                print("Predicted Chunk exhausted at", t_start)
                output = client.infer(obs)['action'][0] # (chunk_size,action_dim)
                act_index = 0

            act = output[act_index]
            act_index += 1

            if args['plot']:
                indices = np.arange(0, 3)  # Plot first 3 action dimensions

                predicted_list.append(act[indices])
                gt_list.append(cur_action[indices])
                

    if args['plot']:
        print("plotting")
        plt.figure()

        for i in range(3):  # x, y, z
            plt.plot([x[i] for x in predicted_list], label=f'Predicted {i}', linestyle='--')  
            plt.plot([x[i] for x in gt_list], label=f'Ground Truth {i}') 

        plt.legend()
        plt.title("Predicted vs Ground Truth")
        plt.xlabel("Time Steps")
        plt.ylabel("Values")

        plt.savefig("predicted_vs_gt.png")
        # plt.show()