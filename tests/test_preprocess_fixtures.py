import hashlib
import json
from pathlib import Path

import numpy as np

from formulalite.processor import FormulaImageProcessor, PreprocessSpec

PROJECT_ROOT = Path(__file__).parents[1]
FIXTURE_ROOT = PROJECT_ROOT / "tests" / "fixtures" / "preprocess"
CONFIG_PATH = PROJECT_ROOT / "artifacts" / "preprocessor" / "preprocess_config.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_golden_fixtures_match_python_reference() -> None:
    fixture_set = json.loads((FIXTURE_ROOT / "manifest.json").read_text(encoding="utf-8"))
    spec = PreprocessSpec.load(CONFIG_PATH)
    processor = FormulaImageProcessor(spec)

    assert fixture_set["preprocess_spec_version"] == spec.spec_version
    assert fixture_set["preprocess_config_sha256"] == sha256(CONFIG_PATH)
    assert len(fixture_set["fixtures"]) == 8

    for entry in fixture_set["fixtures"]:
        record_path = FIXTURE_ROOT / entry["manifest"]
        record = json.loads(record_path.read_text(encoding="utf-8"))
        image_path = FIXTURE_ROOT / record["source"]["path"]
        gray_path = FIXTURE_ROOT / record["expected_image_uint8"]["path"]
        tensor_path = FIXTURE_ROOT / record["expected_tensor"]["path"]
        assert sha256(record_path) == entry["manifest_sha256"]
        assert sha256(image_path) == record["source"]["sha256"]
        assert sha256(gray_path) == record["expected_image_uint8"]["sha256"]
        assert sha256(tensor_path) == record["expected_tensor"]["sha256"]

        actual = processor.preprocess_with_metadata(image_path)
        expected_gray = np.frombuffer(gray_path.read_bytes(), dtype=np.uint8).reshape(384, 384)
        expected_tensor = np.frombuffer(tensor_path.read_bytes(), dtype="<f4").reshape(3, 384, 384)
        assert actual.metadata.to_dict() == record["metadata"]
        assert np.array_equal(actual.image_uint8, expected_gray)
        assert np.array_equal(actual.pixel_values, expected_tensor)
