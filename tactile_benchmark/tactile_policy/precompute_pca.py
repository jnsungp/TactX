import argparse
import os
import pickle
import numpy as np
import h5py
from sklearn.decomposition import IncrementalPCA
from pathlib import Path
from skimage.transform import resize
from tqdm import tqdm

def gather_hdf_paths(base_dir, task_names):
    hdf_paths = []
    for task_name in task_names:
        task_dir = os.path.join(base_dir, task_name)
        if not os.path.exists(task_dir):
            print(f"Warning: {task_dir} does not exist, skipping")
            continue
        for file in sorted(os.listdir(task_dir)):
            if file.endswith('.hdf5'):
                hdf_paths.append(os.path.join(task_dir, file))
    return hdf_paths

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Base directory with task folders')
    parser.add_argument('--task_names', nargs='+', required=True, help='List of task names to include')
    parser.add_argument('--tactile_names', nargs='+', default=['tac0', 'tac1'], help='Tactile sensor names')
    parser.add_argument('--n_components', type=int, default=256, help='Number of PCA components')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for IncrementalPCA')
    parser.add_argument('--image_size', type=int, default=None, help='Resize images to image_size x image_size before PCA (e.g. 64)')
    parser.add_argument('--output_path', type=str, required=True, help='Output path for PCA transformers')
    args = parser.parse_args()

    hdf_paths = gather_hdf_paths(args.data_dir, args.task_names)
    print(f"Found {len(hdf_paths)} episodes across {len(args.task_names)} tasks")

    pca_transformers = {}
    for tactile_name in args.tactile_names:
        print(f"Fitting PCA for {tactile_name}...")
        pca = IncrementalPCA(n_components=args.n_components, batch_size=args.batch_size)
        total_images = 0
        buffer = []
        buffer_size = 0

        for hdf_path in tqdm(hdf_paths, desc=tactile_name):
            with h5py.File(hdf_path, 'r') as root:
                if f'observation.image.{tactile_name}' not in root:
                    continue
                images = root[f'observation.image.{tactile_name}'][()]
                if len(images.shape) == 4 and images.shape[-1] == 3:
                    images = images.transpose(0, 3, 1, 2)
                N, C, H, W = images.shape
                if args.image_size is not None and (H != args.image_size or W != args.image_size):
                    images = np.stack([
                        resize(images[i].transpose(1, 2, 0), (args.image_size, args.image_size),
                               anti_aliasing=True).transpose(2, 0, 1)
                        for i in range(N)
                    ]).astype(np.float32)
                buffer.append(images.reshape(N, -1).astype(np.float32))
                buffer_size += N
                total_images += N

                if buffer_size >= args.batch_size:
                    pca.partial_fit(np.concatenate(buffer, axis=0))
                    buffer = []
                    buffer_size = 0

        if buffer:
            pca.partial_fit(np.concatenate(buffer, axis=0))

        if total_images == 0:
            print(f"Warning: No data found for {tactile_name}")
            continue

        pca_transformers[tactile_name] = pca
        explained_var = np.sum(pca.explained_variance_ratio_)
        print(f"  {tactile_name}: {total_images} images, {args.n_components} components explain {explained_var*100:.2f}% variance")

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'wb') as f:
        pickle.dump(pca_transformers, f)

    print(f"\nSaved PCA transformers to {args.output_path}")

if __name__ == '__main__':
    main()
