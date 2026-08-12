"""Reproducible DataLoader orchestration, with no model or training-loop concerns."""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, cast

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

from formulalite.processor import FormulaImageProcessor, PreprocessSpec

from .collator import FormulaCollator
from .dataset import ManifestFormulaDataset
from .policy import FormulaDataPolicy, OverlengthPolicy, UnknownTokenPolicy
from .schema import FormulaBatch, FormulaSample
from .streaming import ManifestStreamingDataset
from .tokenizer import FormulaTokenizer


def seed_data_worker(_worker_id: int) -> None:
    """Seed Python and NumPy from the deterministic PyTorch worker seed."""

    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


@dataclass(frozen=True)
class FormulaDataConfig:
    name: str = "default"
    train_manifest: str | Path | None = None
    val_manifest: str | Path | None = None
    test_manifest: str | Path | None = None
    batch_size: int = 16
    num_workers: int = 4
    max_sequence_length: int = 1027
    unknown_policy: str = UnknownTokenPolicy.ERROR.value
    overlength_policy: str = OverlengthPolicy.ERROR.value
    shuffle: bool = True
    streaming: bool = False
    shuffle_buffer_size: int = 1
    seed: int = 42
    pin_memory: bool = False
    image_size: int = 384

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")
        UnknownTokenPolicy(self.unknown_policy)
        OverlengthPolicy(self.overlength_policy)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> FormulaDataConfig:
        allowed = {item.name for item in fields(cls)}
        return cls(**{key: item for key, item in value.items() if key in allowed})


FormulaDatasetType = Dataset[FormulaSample] | IterableDataset[FormulaSample]


class FormulaDataModule(L.LightningDataModule):
    """Construct canonical datasets and deterministic train/val/test DataLoaders."""

    def __init__(
        self,
        config: FormulaDataConfig | Mapping[str, Any],
        *,
        tokenizer: FormulaTokenizer | None = None,
        image_processor: FormulaImageProcessor | None = None,
    ) -> None:
        super().__init__()
        self.config = (
            config
            if isinstance(config, FormulaDataConfig)
            else FormulaDataConfig.from_mapping(config)
        )
        self.tokenizer = tokenizer or FormulaTokenizer.build()
        self.image_processor = image_processor or FormulaImageProcessor(
            PreprocessSpec(image_width=self.config.image_size, image_height=self.config.image_size)
        )
        self.collator = FormulaCollator(
            self.tokenizer,
            self.image_processor,
            max_sequence_length=self.config.max_sequence_length,
        )
        self.train_dataset: FormulaDatasetType | None = None
        self.val_dataset: FormulaDatasetType | None = None
        self.test_dataset: FormulaDatasetType | None = None

    def _policy(self) -> FormulaDataPolicy:
        return FormulaDataPolicy(
            tokenizer=self.tokenizer,
            max_sequence_length=self.config.max_sequence_length,
            unknown_policy=UnknownTokenPolicy(self.config.unknown_policy),
            overlength_policy=OverlengthPolicy(self.config.overlength_policy),
        )

    def _dataset(self, path: str | Path | None, *, train: bool) -> FormulaDatasetType:
        if path is None:
            raise ValueError("the requested DataModule stage has no manifest configured")
        if train and self.config.streaming:
            return ManifestStreamingDataset(
                path,
                policy=self._policy(),
                shuffle_buffer_size=self.config.shuffle_buffer_size,
                seed=self.config.seed,
            )
        return ManifestFormulaDataset(path, policy=self._policy())

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            if self.train_dataset is None:
                self.train_dataset = self._dataset(self.config.train_manifest, train=True)
            if self.val_dataset is None:
                self.val_dataset = self._dataset(self.config.val_manifest, train=False)
        elif stage == "validate" and self.val_dataset is None:
            self.val_dataset = self._dataset(self.config.val_manifest, train=False)
        if stage in (None, "test") and self.test_dataset is None:
            self.test_dataset = self._dataset(self.config.test_manifest, train=False)

    def _loader(self, dataset: FormulaDatasetType, *, train: bool) -> DataLoader[FormulaBatch]:
        iterable = isinstance(dataset, IterableDataset)
        generator = torch.Generator().manual_seed(self.config.seed)
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=train and self.config.shuffle and not iterable,
            num_workers=self.config.num_workers,
            collate_fn=self.collator,
            worker_init_fn=seed_data_worker,
            generator=generator,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.num_workers > 0,
        )
        return cast(DataLoader[FormulaBatch], loader)

    def train_dataloader(self) -> DataLoader[FormulaBatch]:
        if self.train_dataset is None:
            raise RuntimeError("call setup('fit') before train_dataloader()")
        return self._loader(self.train_dataset, train=True)

    def val_dataloader(self) -> DataLoader[FormulaBatch]:
        if self.val_dataset is None:
            raise RuntimeError("call setup('fit') or setup('validate') before val_dataloader()")
        return self._loader(self.val_dataset, train=False)

    def test_dataloader(self) -> DataLoader[FormulaBatch]:
        if self.test_dataset is None:
            raise RuntimeError("call setup('test') before test_dataloader()")
        return self._loader(self.test_dataset, train=False)
