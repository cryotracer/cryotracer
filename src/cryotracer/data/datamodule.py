from itertools import count, islice

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset, Sampler, default_collate

from cryotracer.data.synthetic import SyntheticFilamentDataset


def collate_samples(samples):
    shapes = {tuple(sample["pixel_values"].shape) for sample in samples}
    if len(shapes) != 1:
        raise ValueError(
            f"Mixed image sizes within a batch are unsupported: {sorted(shapes)}. "
            "Use equally sized images or --batch-size 1."
        )
    return default_collate(samples)


class EpochDataset(Dataset):
    """Seed augmentation by logical epoch and sample, independently of workers."""

    def __init__(self, dataset, seed):
        self.dataset = dataset
        self.seed = seed

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, epoch_and_index):
        epoch, index = epoch_and_index
        if isinstance(self.dataset, SyntheticFilamentDataset):
            return self.dataset[epoch, index]
        with torch.random.fork_rng(devices=[]):
            # Torch's CPU RNG repeats when only the upper seed bits change.
            torch.manual_seed(self.seed + epoch * len(self.dataset) + index)
            return self.dataset[index]


class TrainingSampler(Sampler):
    """Shuffled dataset passes for the full optimizer-step budget.

    Runs stop at max_steps. Keeping accumulation continuous across data passes
    makes validation intervals refer exactly to optimizer steps.
    """

    def __init__(self, dataset, seed, num_samples):
        self.size = len(dataset)
        if self.size == 0:
            raise ValueError("Training dataset is empty")
        self.seed = seed
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        def indices():
            for epoch in count():
                order = torch.randperm(
                    self.size,
                    generator=torch.Generator().manual_seed(self.seed + epoch),
                ).tolist()
                yield from ((epoch, index) for index in order)

        return islice(indices(), self.num_samples)


class DataModule(L.LightningDataModule):
    def __init__(
        self,
        *,
        train_dataset=None,
        val_dataset=None,
        predict_dataset=None,
        batch_size=8,
        num_workers=4,
        seed=42,
        max_steps=5000,
        accumulate_grad_batches=1,
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.predict_dataset = predict_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.train_batches = max_steps * accumulate_grad_batches

    def _loader(self, dataset, *, training=False):
        if dataset is None or len(dataset) == 0:
            raise ValueError("Dataset must contain at least one micrograph")
        if training:
            dataset = EpochDataset(dataset, self.seed)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=TrainingSampler(
                dataset, self.seed, self.train_batches * self.batch_size
            )
            if training
            else None,
            num_workers=self.num_workers,
            collate_fn=collate_samples,
            multiprocessing_context="spawn" if self.num_workers else None,
            persistent_workers=self.num_workers > 0,
            pin_memory=torch.cuda.is_available(),
            generator=torch.Generator().manual_seed(self.seed),
        )

    def train_dataloader(self):
        return self._loader(self.train_dataset, training=True)

    def val_dataloader(self):
        return self._loader(self.val_dataset)

    def predict_dataloader(self):
        return self._loader(self.predict_dataset)
