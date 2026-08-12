"""Generate deterministic PNG and Python-reference preprocessing golden fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from formulalite.processor import DEFAULT_PREPROCESS_SPEC, FormulaImageProcessor

FIXTURE_VERSION = "1.0.0"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _rgb(width: int, height: int, value: int = 255) -> np.ndarray:
    return np.full((height, width, 3), value, dtype=np.uint8)


def _rgba(width: int, height: int) -> np.ndarray:
    return np.zeros((height, width, 4), dtype=np.uint8)


def _simple_black_formula() -> Image.Image:
    data = _rgb(64, 32)
    data[9:12, 8:24] = 0
    data[19:22, 8:24] = 0
    data[9:22, 15:18] = 0
    data[14:17, 31:47] = 0
    data[9:22, 38:41] = 0
    return Image.fromarray(data, "RGB")


def _wide_formula() -> Image.Image:
    data = _rgb(160, 30)
    data[11:14, 5:155] = 0
    data[7:19, 18:22] = 0
    data[7:19, 78:82] = 0
    data[7:19, 138:142] = 0
    return Image.fromarray(data, "RGB")


def _tall_formula() -> Image.Image:
    data = _rgb(30, 160)
    data[5:155, 13:16] = 0
    data[20:24, 7:23] = 0
    data[78:82, 7:23] = 0
    data[136:140, 7:23] = 0
    return Image.fromarray(data, "RGB")


def _transparent_background() -> Image.Image:
    data = _rgba(80, 40)
    data[10:14, 12:68, 3] = 255
    data[10:14, 12:68, :3] = 0
    data[24:28, 12:68, 3] = 255
    data[24:28, 12:68, :3] = 0
    data[10:28, 38:42, 3] = 255
    data[10:28, 38:42, :3] = 0
    # Hidden RGB in transparent pixels verifies that alpha composition is explicit.
    data[0:5, 0:5, :3] = [255, 0, 0]
    return Image.fromarray(data, "RGBA")


def _white_on_black() -> Image.Image:
    data = _rgb(64, 32, 0)
    data[9:12, 8:56] = 255
    data[20:23, 8:56] = 255
    data[9:23, 30:34] = 255
    return Image.fromarray(data, "RGB")


def _near_threshold() -> Image.Image:
    data = _rgb(16, 8)
    data[3, 3:8] = [0, 0, 0]
    data[4, 3] = [199, 199, 199]
    data[4, 4] = [200, 200, 200]
    data[4, 5] = [201, 201, 201]
    return Image.fromarray(data, "RGB")


def _one_pixel_foreground() -> Image.Image:
    data = _rgb(9, 7)
    data[3, 4] = 0
    return Image.fromarray(data, "RGB")


def _empty_image() -> Image.Image:
    return Image.fromarray(_rgb(32, 20), "RGB")


FIXTURES: dict[str, Callable[[], Image.Image]] = {
    "simple_black_formula": _simple_black_formula,
    "wide_formula": _wide_formula,
    "tall_formula": _tall_formula,
    "transparent_background": _transparent_background,
    "white_on_black": _white_on_black,
    "near_threshold": _near_threshold,
    "one_pixel_foreground": _one_pixel_foreground,
    "empty_image": _empty_image,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture-dir",
        type=Path,
        default=Path("tests/fixtures/preprocess"),
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=Path("artifacts/preprocessor/preprocess_config.json"),
    )
    return parser.parse_args()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def main() -> None:
    args = parse_args()
    image_dir = args.fixture_dir / "images"
    expected_dir = args.fixture_dir / "expected"
    image_dir.mkdir(parents=True, exist_ok=True)
    expected_dir.mkdir(parents=True, exist_ok=True)
    DEFAULT_PREPROCESS_SPEC.save(args.config_path)
    processor = FormulaImageProcessor(DEFAULT_PREPROCESS_SPEC)
    entries: list[dict[str, Any]] = []

    for name, factory in FIXTURES.items():
        image_path = image_dir / f"{name}.png"
        gray_path = expected_dir / f"{name}.gray.bin"
        tensor_path = expected_dir / f"{name}.f32.bin"
        metadata_path = expected_dir / f"{name}.json"

        factory().save(image_path, format="PNG", compress_level=9, optimize=False)
        result = processor.preprocess_with_metadata(image_path)
        gray_payload = result.image_uint8.tobytes(order="C")
        tensor_payload = result.pixel_values.astype("<f4", copy=False).tobytes(order="C")
        gray_path.write_bytes(gray_payload)
        tensor_path.write_bytes(tensor_payload)

        record: dict[str, Any] = {
            "name": name,
            "fixture_version": FIXTURE_VERSION,
            "preprocess_spec_version": DEFAULT_PREPROCESS_SPEC.spec_version,
            "source": {
                "path": f"images/{image_path.name}",
                "sha256": _sha256(image_path.read_bytes()),
            },
            "expected_image_uint8": {
                "path": f"expected/{gray_path.name}",
                "shape": [
                    DEFAULT_PREPROCESS_SPEC.image_height,
                    DEFAULT_PREPROCESS_SPEC.image_width,
                ],
                "sha256": _sha256(gray_payload),
            },
            "expected_tensor": {
                "path": f"expected/{tensor_path.name}",
                "dtype": "float32-le",
                "layout": "CHW",
                "shape": [
                    DEFAULT_PREPROCESS_SPEC.channels,
                    DEFAULT_PREPROCESS_SPEC.image_height,
                    DEFAULT_PREPROCESS_SPEC.image_width,
                ],
                "sha256": _sha256(tensor_payload),
            },
            "metadata": result.metadata.to_dict(),
        }
        record["fixture_sha256"] = _sha256(
            image_path.read_bytes()
            + gray_payload
            + tensor_payload
            + _canonical_json(record["metadata"])
        )
        metadata_path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        entries.append(
            {
                "name": name,
                "manifest": f"expected/{metadata_path.name}",
                "manifest_sha256": _sha256(metadata_path.read_bytes()),
                "fixture_sha256": record["fixture_sha256"],
            }
        )

    manifest = {
        "fixture_version": FIXTURE_VERSION,
        "preprocess_spec_version": DEFAULT_PREPROCESS_SPEC.spec_version,
        "preprocess_config_sha256": _sha256(args.config_path.read_bytes()),
        "fixtures": entries,
    }
    manifest["fixture_set_sha256"] = _sha256(_canonical_json(manifest))
    (args.fixture_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
