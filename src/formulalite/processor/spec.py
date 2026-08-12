"""Serializable, cross-runtime preprocessing contract."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

PREPROCESS_SPEC_VERSION: Final = "1.0.0"


@dataclass(frozen=True)
class PreprocessSpec:
    """All values and rounding conventions needed for deterministic preprocessing."""

    spec_version: str = PREPROCESS_SPEC_VERSION
    uint8_max: int = 255
    image_width: int = 384
    image_height: int = 384
    foreground_threshold: int = 200
    mean: tuple[float, float, float] = (0.7931, 0.7931, 0.7931)
    std: tuple[float, float, float] = (0.1738, 0.1738, 0.1738)
    output_dtype: str = "float32"
    output_layout: str = "CHW"
    channels: int = 3
    alpha_background: tuple[int, int, int] = (255, 255, 255)
    polarity_border_threshold: float = 127.5
    resize_interpolation: str = "bilinear"
    padding_value: int = 0

    def __post_init__(self) -> None:
        if self.spec_version != PREPROCESS_SPEC_VERSION:
            msg = f"unsupported preprocess spec version: {self.spec_version}"
            raise ValueError(msg)
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("image dimensions must be positive")
        if self.uint8_max != 255:
            raise ValueError("the Phase 2 image contract requires 8-bit samples")
        if not 0 <= self.foreground_threshold <= 255:
            raise ValueError("foreground threshold must be uint8")
        if len(self.mean) != self.channels or len(self.std) != self.channels:
            raise ValueError("normalization vectors must match channel count")
        if any(value <= 0 for value in self.std):
            raise ValueError("normalization standard deviations must be positive")
        if self.output_dtype != "float32" or self.output_layout != "CHW":
            raise ValueError("the Phase 2 contract requires float32 CHW output")
        if self.channels != 3:
            raise ValueError("the Phase 2 contract requires three replicated channels")
        if self.resize_interpolation != "bilinear":
            raise ValueError("the reference implementation only supports bilinear resize")

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec_version": self.spec_version,
            "uint8_max": self.uint8_max,
            "image_size": {"width": self.image_width, "height": self.image_height},
            "foreground_threshold": self.foreground_threshold,
            "grayscale": {
                "method": "bt601_integer",
                "coefficients": [299, 587, 114],
                "divisor": 1000,
                "rounding": "round_half_up",
            },
            "mean": list(self.mean),
            "std": list(self.std),
            "output_dtype": self.output_dtype,
            "output_layout": self.output_layout,
            "channels": self.channels,
            "alpha": {
                "mode": "straight_alpha_composite",
                "background_rgb": list(self.alpha_background),
                "divisor": 255,
                "rounding": "round_half_up",
                "fully_transparent_behavior": "background",
            },
            "polarity": {
                "mode": "border_mean",
                "border_pixels": "unique_perimeter",
                "invert_when_mean_below": self.polarity_border_threshold,
                "inversion": "255_minus_value",
            },
            "dynamic_range": {
                "mode": "min_max_uint8",
                "formula": "round_half_up((value-min)*255/(max-min))",
                "constant_image_behavior": "unchanged",
            },
            "bbox": {
                "coordinate_convention": "half_open",
                "axes": {"x": "[x0,x1)", "y": "[y0,y1)"},
                "foreground_rule": "pixel < foreground_threshold",
                "empty_image_behavior": "full_input_bbox",
            },
            "resize": {
                "mode": "fit",
                "preserve_aspect_ratio": True,
                "interpolation": self.resize_interpolation,
                "coordinate_transform": "half_pixel",
                "boundary": "clamp",
                "size_rounding": "round_half_up",
                "sample_rounding": "round_half_up_to_uint8",
            },
            "padding": {
                "mode": "constant",
                "value": self.padding_value,
                "placement": "center",
                "odd_remainder": "right_and_bottom",
            },
            "pipeline": [
                "decode",
                "alpha_composite",
                "grayscale",
                "polarity_normalization",
                "dynamic_range_normalization",
                "foreground_bbox",
                "crop",
                "aspect_ratio_resize",
                "center_padding",
                "mean_std_normalization",
                "replicate_channels",
                "float32_chw",
            ],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PreprocessSpec:
        size = value["image_size"]
        alpha = value["alpha"]
        polarity = value["polarity"]
        resize = value["resize"]
        padding = value["padding"]
        return cls(
            spec_version=str(value["spec_version"]),
            uint8_max=int(value["uint8_max"]),
            image_width=int(size["width"]),
            image_height=int(size["height"]),
            foreground_threshold=int(value["foreground_threshold"]),
            mean=tuple(float(item) for item in value["mean"]),  # type: ignore[arg-type]
            std=tuple(float(item) for item in value["std"]),  # type: ignore[arg-type]
            output_dtype=str(value["output_dtype"]),
            output_layout=str(value["output_layout"]),
            channels=int(value["channels"]),
            alpha_background=tuple(int(item) for item in alpha["background_rgb"]),  # type: ignore[arg-type]
            polarity_border_threshold=float(polarity["invert_when_mean_below"]),
            resize_interpolation=str(resize["interpolation"]),
            padding_value=int(padding["value"]),
        )

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> PreprocessSpec:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("preprocess config must be a JSON object")
        return cls.from_dict(value)


DEFAULT_PREPROCESS_SPEC: Final = PreprocessSpec()
