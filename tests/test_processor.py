import json
from pathlib import Path

import numpy as np
from PIL import Image

from formulalite.processor import FormulaImageProcessor, PreprocessSpec


def test_preprocess_spec_serialization_round_trip(tmp_path: Path) -> None:
    spec = PreprocessSpec()
    path = tmp_path / "preprocess_config.json"
    spec.save(path)
    assert PreprocessSpec.load(path) == spec
    value = json.loads(path.read_text(encoding="utf-8"))
    assert value["bbox"]["coordinate_convention"] == "half_open"
    assert value["resize"]["size_rounding"] == "round_half_up"
    assert value["padding"]["odd_remainder"] == "right_and_bottom"


def test_python_processor_shape_dtype_and_metadata() -> None:
    image = np.full((20, 40, 4), 255, dtype=np.uint8)
    image[8:12, 10:30, :3] = 0
    processor = FormulaImageProcessor()
    result = processor.preprocess_with_metadata(Image.fromarray(image, "RGBA"))

    assert result.pixel_values.shape == (3, 384, 384)
    assert result.pixel_values.dtype == np.float32
    assert result.image_uint8.shape == (384, 384)
    assert result.metadata.bbox.x0 == 10
    assert result.metadata.bbox.x1 == 30
    assert result.metadata.bbox.y0 == 8
    assert result.metadata.bbox.y1 == 12
    assert result.metadata.input_size == (40, 20)
    assert result.metadata.crop_size == (20, 4)
    assert result.metadata.inverted is False


def test_white_on_black_is_inverted() -> None:
    image = np.zeros((12, 20, 3), dtype=np.uint8)
    image[4:8, 5:15] = 255
    result = FormulaImageProcessor().preprocess_with_metadata(Image.fromarray(image, "RGB"))
    assert result.metadata.inverted is True
    assert result.metadata.bbox == result.metadata.bbox.__class__(5, 4, 15, 8)


def test_transparent_image_uses_white_background_and_is_empty() -> None:
    image = np.zeros((7, 9, 4), dtype=np.uint8)
    result = FormulaImageProcessor().preprocess_with_metadata(Image.fromarray(image, "RGBA"))
    assert result.metadata.empty is True
    assert result.metadata.bbox == result.metadata.bbox.__class__(0, 0, 9, 7)
    top = result.metadata.padding.top
    height = result.metadata.resize_size[1]
    assert np.all(result.image_uint8[top : top + height] == 255)
    assert np.all(result.image_uint8[:top] == 0)


def test_preprocess_is_deterministic() -> None:
    image = Image.new("RGB", (17, 11), "white")
    pixels = np.asarray(image).copy()
    pixels[3:8, 4:13] = 0
    processor = FormulaImageProcessor()
    first = processor.preprocess(Image.fromarray(pixels))
    second = processor.preprocess(Image.fromarray(pixels))
    assert np.array_equal(first, second)
