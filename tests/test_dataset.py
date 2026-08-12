from pathlib import Path

from PIL import Image

from formulalite.data.dataset import ManifestFormulaDataset, iter_manifest_records
from formulalite.data.normalizer import NORMALIZER_VERSION, normalize
from formulalite.data.policy import FormulaDataPolicy, OverlengthPolicy, UnknownTokenPolicy
from formulalite.data.schema import FormulaRecord, FormulaSample
from formulalite.data.tokenizer import FormulaTokenizer

FIXTURE = Path(__file__).parent / "fixtures" / "dataset" / "manifest.jsonl"


def policy() -> FormulaDataPolicy:
    return FormulaDataPolicy(
        FormulaTokenizer.build(),
        max_sequence_length=32,
        unknown_policy=UnknownTokenPolicy.DROP,
        overlength_policy=OverlengthPolicy.DROP,
    )


def test_canonical_schema() -> None:
    record = FormulaRecord("sample", Path("image.png"), r"x+y", "SPE", "unit")
    sample = FormulaSample(
        id=record.id,
        image=record.image_ref,
        raw_latex=record.raw_latex,
        normalized_latex=normalize(record.raw_latex),
        subset=record.subset,
        source=record.source,
        normalizer_version=NORMALIZER_VERSION,
    )
    assert sample.id == "sample"
    assert sample.raw_latex == r"x+y"
    assert sample.normalized_latex == "x + y"
    assert sample.subset == "SPE"


def test_manifest_records_are_backend_isolated() -> None:
    records = tuple(iter_manifest_records(FIXTURE))
    assert len(records) == 12
    assert all(isinstance(record, FormulaRecord) for record in records)
    assert records[0].image_ref == FIXTURE.parent / "../preprocess/images/simple_black_formula.png"


def test_map_dataset_preserves_raw_and_normalized_labels() -> None:
    dataset = ManifestFormulaDataset(FIXTURE, policy=policy())
    assert len(dataset) == 10
    sample = next(dataset[index] for index in range(len(dataset)) if dataset[index].id == "fraction")
    assert isinstance(sample.image, Image.Image)
    assert sample.raw_latex == r"\frac{a+b}{c-d}"
    assert sample.normalized_latex == r"\frac { a + b } { c - d }"
    assert sample.normalizer_version == NORMALIZER_VERSION
    assert sample.subset == "CPE"
    assert sample.source == "fixture"


def test_map_dataset_statistics_include_dropped_samples() -> None:
    dataset = ManifestFormulaDataset(FIXTURE, policy=policy())
    report = dataset.statistics.to_report()
    assert report["total_samples"] == 12
    assert report["normalized_samples"] == 12
    assert report["unknown_samples"] == 1
    assert report["dropped_unknown_samples"] == 1
    assert report["overlength_samples"] == 1
    assert report["dropped_overlength_samples"] == 1
    assert report["subset_distribution"] == {"CPE": 3, "HWE": 3, "SCE": 3, "SPE": 3}
