"""Custom batch samplers for contrastive learning."""

import numpy as np
from torch.utils.data import Sampler


def _get_episode_ids_for_dataset(dataset):
    """
    Return list of episode_ids, one per sample index of `dataset`.
    Works for CrossModalDataset directly or wrapped in torch.utils.data.Subset.
    """
    # Subset: dataset.dataset is the base, dataset.indices are the subset's indices
    if hasattr(dataset, 'indices') and hasattr(dataset, 'dataset'):
        base = dataset.dataset
        return [base.episode_ids[i] for i in dataset.indices]
    return list(dataset.episode_ids)


class EpisodeBatchSampler(Sampler):
    """
    Yields batches where each batch contains:
      - n_episodes_per_batch distinct episodes (sampled with replacement across batches)
      - n_frames_per_episode frames per episode (sampled without replacement within an episode)

    This forces NT-Xent to see many within-episode negatives, which are much harder
    than random cross-episode negatives — especially useful when hold-phase frames
    dominate the dataset.

    Total batch size = n_episodes_per_batch * n_frames_per_episode.

    Args:
        dataset: CrossModalDataset or Subset wrapping one. Must expose `episode_ids`.
        n_episodes_per_batch: Number of distinct episodes per batch.
        n_frames_per_episode: Number of frames to draw from each episode.
        n_batches_per_epoch: Total batches per epoch. If None, defaults to
            total_samples // batch_size (matches a standard shuffled DataLoader).
    """

    def __init__(self, dataset, n_episodes_per_batch, n_frames_per_episode,
                 n_batches_per_epoch=None):
        self.n_eps = n_episodes_per_batch
        self.n_frames = n_frames_per_episode

        ep_per_idx = _get_episode_ids_for_dataset(dataset)

        # Group sample indices by episode
        self.episode_to_indices = {}
        for idx, ep in enumerate(ep_per_idx):
            self.episode_to_indices.setdefault(ep, []).append(idx)

        # Only use episodes that have at least n_frames_per_episode samples
        self.valid_episodes = [
            ep for ep, idxs in self.episode_to_indices.items()
            if len(idxs) >= n_frames_per_episode
        ]
        if not self.valid_episodes:
            raise ValueError(
                f"No episodes have >= {n_frames_per_episode} frames. "
                f"Max episode length: {max(len(v) for v in self.episode_to_indices.values())}. "
                f"Reduce n_frames_per_episode or loosen thinning."
            )

        if n_batches_per_epoch is None:
            total = sum(len(v) for v in self.episode_to_indices.values())
            bs = n_episodes_per_batch * n_frames_per_episode
            n_batches_per_epoch = max(1, total // bs)
        self.n_batches = n_batches_per_epoch

        print(f"  EpisodeBatchSampler: {n_episodes_per_batch} eps x {n_frames_per_episode} frames "
              f"= {n_episodes_per_batch * n_frames_per_episode} batch size, "
              f"{self.n_batches} batches/epoch, "
              f"{len(self.valid_episodes)}/{len(self.episode_to_indices)} episodes usable")

    def __iter__(self):
        for _ in range(self.n_batches):
            eps = np.random.choice(self.valid_episodes, self.n_eps, replace=True)
            batch = []
            for ep in eps:
                indices = self.episode_to_indices[ep]
                sampled = np.random.choice(indices, self.n_frames, replace=False)
                batch.extend(sampled.tolist())
            yield batch

    def __len__(self):
        return self.n_batches
