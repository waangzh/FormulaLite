from pathlib import Path

from torch.utils.data import DataLoader

from formulalite.data.policy import FormulaDataPolicy, OverlengthPolicy, UnknownTokenPolicy
from formulalite.data.schema import FormulaRecord
from formulalite.data.streaming import ManifestStreamingDataset, StreamingFormulaDataset
from formulalite.data.tokenizer import FormulaTokenizer

FIXTURE_ROOT = Path(__file__).parent / "fixtures"
MANIFEST = FIXTURE_ROOT / "dataset" / "manifest.jsonl"
IMAGE = FIXTURE_ROOT / "preprocess" / "images" / "simple_black_formula.png"


def keep_policy(max_length: int = 128) -> FormulaDataPolicy:
    return FormulaDataPolicy(
        FormulaTokenizer.build(),
        max_sequence_length=max_length,
        unknown_policy=UnknownTokenPolicy.KEEP,
        overlength_policy=OverlengthPolicy.ERROR,
    )


def test_streaming_does_not_materialize_source() -> None:
    yielded = 0

    def records():
        nonlocal yielded
        for index in range(100):
            yielded += 1
            yield FormulaRecord(str(index), IMAGE, "x", "SPE", "lazy")

    dataset = StreamingFormulaDataset(records, policy=keep_policy(), shuffle_buffer_size=1)
    first = next(iter(dataset))
    assert first.id == "0"
    assert yielded == 1


def test_bounded_shuffle_consumes_only_buffer_plus_one() -> None:
    yielded = 0

    def records():
        nonlocal yielded
        for index in range(100):
            yielded += 1
            yield FormulaRecord(str(index), IMAGE, "x")

    dataset = StreamingFormulaDataset(
        records, policy=keep_policy(), shuffle_buffer_size=4, seed=9
    )
    next(iter(dataset))
    assert yielded == 5


def test_streaming_shuffle_is_reproducible() -> None:
    first = ManifestStreamingDataset(MANIFEST, policy=keep_policy(), shuffle_buffer_size=5, seed=7)
    second = ManifestStreamingDataset(MANIFEST, policy=keep_policy(), shuffle_buffer_size=5, seed=7)
    third = ManifestStreamingDataset(MANIFEST, policy=keep_policy(), shuffle_buffer_size=5, seed=8)
    first_ids = [sample.id for sample in first]
    assert first_ids == [sample.id for sample in second]
    assert first_ids != [sample.id for sample in third]


def test_streaming_multi_worker_shards_without_duplicates() -> None:
    dataset = ManifestStreamingDataset(MANIFEST, policy=keep_policy())
    ids = list(
        DataLoader(dataset, batch_size=None, num_workers=2, multiprocessing_context="spawn")
    )
    sample_ids = [sample.id for sample in ids]
    assert len(sample_ids) == 12
    assert len(set(sample_ids)) == 12
