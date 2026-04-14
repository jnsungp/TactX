from collections.abc import Iterator, Sequence
import multiprocessing
import os
import typing
from typing import Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
import torch

import tactile_policy.openpi.models.model as _model
import tactile_policy.openpi.training.config as _config
import tactile_policy.openpi.models.tokenizer as _tokenizer

T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    # def data_config(self) -> _config.DataConfig:
    #     """Get the data config for this data loader."""
    #     raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class ConvertToTrainableData():
    def __init__(self, tokenizer: _tokenizer.PaligemmaTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        image_data, qpos_data, action_data, is_pad, conditioning_dict = batch
        prompt = conditioning_dict["plain_text"].tolist()
        tokenized_prompt = []
        tokenized_prompt_mask = []
        for i in range(len(prompt)):
            tokens, token_masks = self.tokenizer.tokenize(prompt[i].lower())
            tokenized_prompt.append(tokens)
            tokenized_prompt_mask.append(token_masks)
        tokenized_prompt = np.stack(tokenized_prompt, axis=0)
        tokenized_prompt_mask = np.stack(tokenized_prompt_mask, axis=0)
        image_dict = {
            "cam_left": image_data[:, 0].transpose(0, 2, 3, 1),
            "cam_right": image_data[:, 1].transpose(0, 2, 3, 1),
        }
        image_mask_dict = {
            "cam_left": np.ones(image_data.shape[0], dtype=bool),
            "cam_right": np.ones(image_data.shape[0], dtype=bool),
        }
        data_dict = {
            "image": image_dict,
            "image_mask": image_mask_dict,
            "state": qpos_data,
            "actions": action_data,
            "tokenized_prompt": tokenized_prompt,
            "tokenized_prompt_mask": tokenized_prompt_mask,
            "is_pad": is_pad,
        }
        return data_dict

def create_data_loader(
    config: _config.TrainConfig,
    sharding: jax.sharding.Sharding | None = None,
    torch_dataset: torch.utils.data.Dataset | None = None,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
    """

    data_loader = TorchDataLoader(
        torch_dataset,
        local_batch_size=config.batch_size // jax.process_count(),
        sharding=sharding,
        shuffle=shuffle,
        num_workers=num_workers,
        seed=config.seed,
        convert_to_trainable_data=ConvertToTrainableData(_tokenizer.PaligemmaTokenizer(config.max_token_len)))

    class DataLoaderImpl(DataLoader):
        def __init__(self, data_loader: TorchDataLoader):
            self._data_loader = data_loader

        def __iter__(self):
            for batch in self._data_loader:
                yield _model.Observation.from_dict(batch), batch["actions"], batch["is_pad"]

    return DataLoaderImpl(data_loader)


class TorchDataLoader:
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        local_batch_size: int,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        num_workers: int = 0,
        seed: int = 0,
        convert_to_trainable_data: ConvertToTrainableData | None = None,
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        self.convert_to_trainable_data = convert_to_trainable_data
        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        # self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            pin_memory=True,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                batch = self.convert_to_trainable_data(batch)
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *x: np.stack(np.asarray(x), axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
