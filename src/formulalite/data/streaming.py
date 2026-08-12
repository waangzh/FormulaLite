"""Non-materializing iterable FormulaLite datasets with bounded shuffling."""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable, Iterator
from functools import partial
from pathlib import Path

from torch.utils.data import IterableDataset, get_worker_info

from .dataset import decode_record, iter_manifest_records
from .normalizer import NORMALIZER_VERSION, normalize
from .policy import FormulaDataPolicy, FormulaDataStatistics, tokenizer_vocab_sha256
from .schema import FormulaRecord, FormulaSample


class StreamingFormulaDataset(IterableDataset[FormulaSample]):
    """Sequential record adapter with worker sharding and bounded-buffer shuffle."""

    def __init__(
        self,
        record_factory: Callable[[], Iterable[FormulaRecord]],
        *,
        policy: FormulaDataPolicy,
        source: str = "stream",
        shuffle_buffer_size: int = 1,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if shuffle_buffer_size < 1:
            raise ValueError("shuffle_buffer_size must be at least 1")
        self.record_factory = record_factory
        self.policy = policy
        self.source = source
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed
        self.epoch = 0
        self.statistics = self._new_statistics()

    def _new_statistics(self) -> FormulaDataStatistics:
        return FormulaDataStatistics(
            dataset_source=self.source,
            tokenizer_sha256=tokenizer_vocab_sha256(self.policy.tokenizer),
        )

    def reset_statistics(self) -> None:
        self.statistics = self._new_statistics()

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = epoch

    def _worker_records(self) -> Iterator[FormulaRecord]:
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        worker_count = worker.num_workers if worker else 1
        for index, record in enumerate(self.record_factory()):
            if index % worker_count == worker_id:
                yield record

    def _accepted_samples(self) -> Iterator[FormulaSample]:
        for record in self._worker_records():
            normalized = normalize(record.raw_latex)
            policy_sample = FormulaSample(
                id=record.id,
                image=record.image_ref,
                raw_latex=record.raw_latex,
                normalized_latex=normalized,
                subset=record.subset,
                source=record.source,
                normalizer_version=NORMALIZER_VERSION,
            )
            if self.policy.evaluate(policy_sample, self.statistics):
                yield decode_record(record, normalized)

    def __iter__(self) -> Iterator[FormulaSample]:
        samples = self._accepted_samples()
        if self.shuffle_buffer_size == 1:
            yield from samples
            return

        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        rng = random.Random(self.seed + self.epoch * 1_000_003 + worker_id)
        buffer: list[FormulaSample] = []
        for sample in samples:
            if len(buffer) < self.shuffle_buffer_size:
                buffer.append(sample)
                continue
            selected = rng.randrange(len(buffer))
            yield buffer[selected]
            buffer[selected] = sample
        while buffer:
            yield buffer.pop(rng.randrange(len(buffer)))


class ManifestStreamingDataset(StreamingFormulaDataset):
    """Streaming adapter that reopens and scans a JSON Lines manifest per iterator."""

    def __init__(
        self,
        path: str | Path,
        *,
        policy: FormulaDataPolicy,
        shuffle_buffer_size: int = 1,
        seed: int = 0,
    ) -> None:
        manifest = Path(path)
        super().__init__(
            partial(iter_manifest_records, manifest),
            policy=policy,
            source=str(manifest),
            shuffle_buffer_size=shuffle_buffer_size,
            seed=seed,
        )
