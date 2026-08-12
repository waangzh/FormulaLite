from pathlib import Path

import pytest

from formulalite.data.dataset import ManifestFormulaDataset
from formulalite.data.policy import (
    FormulaDataPolicy,
    OverlengthError,
    OverlengthPolicy,
    UnknownTokenError,
    UnknownTokenPolicy,
)
from formulalite.data.tokenizer import FormulaTokenizer

MANIFEST = Path(__file__).parent / "fixtures" / "dataset" / "manifest.jsonl"


def make_policy(unknown: UnknownTokenPolicy, overlength: OverlengthPolicy, limit: int = 32):
    return FormulaDataPolicy(
        FormulaTokenizer.build(),
        max_sequence_length=limit,
        unknown_policy=unknown,
        overlength_policy=overlength,
    )


def test_unknown_error_identifies_sample() -> None:
    with pytest.raises(UnknownTokenError, match="unknown-matrix"):
        ManifestFormulaDataset(
            MANIFEST,
            policy=make_policy(UnknownTokenPolicy.ERROR, OverlengthPolicy.DROP),
        )


def test_unknown_drop_is_explicit_and_counted() -> None:
    dataset = ManifestFormulaDataset(
        MANIFEST,
        policy=make_policy(UnknownTokenPolicy.DROP, OverlengthPolicy.DROP),
    )
    assert "unknown-matrix" not in {dataset[index].id for index in range(len(dataset))}
    assert dataset.statistics.dropped_unknown_samples == 1
    assert dataset.statistics.unknown_token_count > 0


def test_unknown_keep_is_available_for_diagnostics() -> None:
    dataset = ManifestFormulaDataset(
        MANIFEST,
        policy=make_policy(UnknownTokenPolicy.KEEP, OverlengthPolicy.DROP),
    )
    assert "unknown-matrix" in {dataset[index].id for index in range(len(dataset))}
    assert dataset.statistics.unknown_samples == 1
    assert dataset.statistics.dropped_unknown_samples == 0


def test_overlength_error_identifies_sample() -> None:
    with pytest.raises(OverlengthError, match="overlength"):
        ManifestFormulaDataset(
            MANIFEST,
            policy=make_policy(UnknownTokenPolicy.KEEP, OverlengthPolicy.ERROR),
        )


def test_no_silent_truncation_and_report_output(tmp_path: Path) -> None:
    dataset = ManifestFormulaDataset(
        MANIFEST,
        policy=make_policy(UnknownTokenPolicy.KEEP, OverlengthPolicy.DROP),
    )
    assert "overlength" not in {dataset[index].id for index in range(len(dataset))}
    stats = dataset.statistics.sequence_length_statistics()
    maximum = stats["max"]
    assert isinstance(maximum, int)
    assert maximum > 32
    assert stats["overlength_sample_count"] == 1
    assert stats["p50"] is not None
    output = tmp_path / "coverage.json"
    dataset.statistics.write_json(output)
    assert output.read_text(encoding="utf-8").endswith("\n")
    assert '"tokenizer_sha256"' in output.read_text(encoding="utf-8")
