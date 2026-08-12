"""Deterministic Python reference implementation of the preprocessing contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import BinaryIO, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from .spec import DEFAULT_PREPROCESS_SPEC, PreprocessSpec

UInt8Image: TypeAlias = NDArray[np.uint8]
Float32Tensor: TypeAlias = NDArray[np.float32]
ImageInput: TypeAlias = Image.Image | str | Path | bytes | bytearray | BinaryIO


@dataclass(frozen=True)
class BBox:
    x0: int
    y0: int
    x1: int
    y1: int


@dataclass(frozen=True)
class Padding:
    left: int
    top: int
    right: int
    bottom: int


@dataclass(frozen=True)
class PreprocessMetadata:
    bbox: BBox
    input_size: tuple[int, int]
    crop_size: tuple[int, int]
    resize_size: tuple[int, int]
    padding: Padding
    inverted: bool
    empty: bool

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        for key in ("input_size", "crop_size", "resize_size"):
            result[key] = list(result[key])
        return result


@dataclass(frozen=True)
class PreprocessResult:
    pixel_values: Float32Tensor
    image_uint8: UInt8Image
    metadata: PreprocessMetadata


class FormulaImageProcessor:
    """Reference-only eval/inference processor; no augmentation is performed."""

    def __init__(self, spec: PreprocessSpec = DEFAULT_PREPROCESS_SPEC) -> None:
        self.spec = spec

    def preprocess(self, image: ImageInput) -> Float32Tensor:
        return self.preprocess_with_metadata(image).pixel_values

    def preprocess_with_metadata(self, image: ImageInput) -> PreprocessResult:
        rgba = self._decode(image)
        rgb = self._alpha_composite(rgba)
        grayscale = self._grayscale(rgb)
        grayscale, inverted = self._normalize_polarity(grayscale)
        grayscale = self._normalize_dynamic_range(grayscale)
        bbox, empty = self._foreground_bbox(grayscale)
        cropped = grayscale[bbox.y0 : bbox.y1, bbox.x0 : bbox.x1]
        resize_width, resize_height = self._resize_dimensions(cropped.shape[1], cropped.shape[0])
        resized = self._resize_bilinear(cropped, resize_width, resize_height)
        padded, padding = self._center_pad(resized)
        tensor = self._normalize_tensor(padded)
        metadata = PreprocessMetadata(
            bbox=bbox,
            input_size=(rgba.shape[1], rgba.shape[0]),
            crop_size=(cropped.shape[1], cropped.shape[0]),
            resize_size=(resize_width, resize_height),
            padding=padding,
            inverted=inverted,
            empty=empty,
        )
        return PreprocessResult(pixel_values=tensor, image_uint8=padded, metadata=metadata)

    @staticmethod
    def _decode(image: ImageInput) -> UInt8Image:
        source: Image.Image
        if isinstance(image, Image.Image):
            source = image
        elif isinstance(image, (bytes, bytearray)):
            source = Image.open(BytesIO(image))
        elif isinstance(image, (str, Path)):
            source = Image.open(image)
        elif hasattr(image, "read"):
            source = Image.open(cast(BinaryIO, image))
        else:
            raise TypeError("image must be a PIL image, path, encoded bytes, or binary stream")
        source.load()
        return np.asarray(source.convert("RGBA"), dtype=np.uint8)

    def _alpha_composite(self, rgba: UInt8Image) -> UInt8Image:
        source = rgba[..., :3].astype(np.uint32)
        alpha = rgba[..., 3:4].astype(np.uint32)
        background = np.asarray(self.spec.alpha_background, dtype=np.uint32)
        maximum = self.spec.uint8_max
        composite = (
            source * alpha + background * (maximum - alpha) + maximum // 2
        ) // maximum
        return composite.astype(np.uint8)

    def _grayscale(self, rgb: UInt8Image) -> UInt8Image:
        values = rgb.astype(np.uint32)
        config = self.spec.to_dict()["grayscale"]
        coefficients = np.asarray(config["coefficients"], dtype=np.uint32)  # type: ignore[index]
        divisor = int(config["divisor"])  # type: ignore[index]
        gray = (np.sum(values * coefficients, axis=2) + divisor // 2) // divisor
        return gray.astype(np.uint8)

    def _normalize_polarity(self, gray: UInt8Image) -> tuple[UInt8Image, bool]:
        height, width = gray.shape
        mask = np.zeros_like(gray, dtype=np.bool_)
        mask[0, :] = True
        mask[height - 1, :] = True
        mask[:, 0] = True
        mask[:, width - 1] = True
        border_mean = float(np.mean(gray[mask], dtype=np.float64))
        inverted = border_mean < self.spec.polarity_border_threshold
        if inverted:
            return np.subtract(self.spec.uint8_max, gray, dtype=np.uint8), True
        return gray, False

    def _normalize_dynamic_range(self, gray: UInt8Image) -> UInt8Image:
        minimum = int(gray.min())
        maximum = int(gray.max())
        if minimum == maximum:
            return gray.copy()
        value_range = maximum - minimum
        values = gray.astype(np.uint32) - minimum
        maximum_value = self.spec.uint8_max
        return ((values * maximum_value + value_range // 2) // value_range).astype(np.uint8)

    def _foreground_bbox(self, gray: UInt8Image) -> tuple[BBox, bool]:
        ys, xs = np.nonzero(gray < self.spec.foreground_threshold)
        if len(xs) == 0:
            return BBox(0, 0, gray.shape[1], gray.shape[0]), True
        return BBox(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1), False

    def _resize_dimensions(self, width: int, height: int) -> tuple[int, int]:
        scale = min(self.spec.image_width / width, self.spec.image_height / height)
        resized_width = min(self.spec.image_width, max(1, int(np.floor(width * scale + 0.5))))
        resized_height = min(self.spec.image_height, max(1, int(np.floor(height * scale + 0.5))))
        return resized_width, resized_height

    @staticmethod
    def _resize_bilinear(image: UInt8Image, width: int, height: int) -> UInt8Image:
        source_height, source_width = image.shape
        if source_width == width and source_height == height:
            return image.copy()

        x = (np.arange(width, dtype=np.float64) + 0.5) * source_width / width - 0.5
        y = (np.arange(height, dtype=np.float64) + 0.5) * source_height / height - 0.5
        x_floor = np.floor(x).astype(np.int64)
        y_floor = np.floor(y).astype(np.int64)
        x_weight = x - x_floor
        y_weight = y - y_floor
        x0 = np.clip(x_floor, 0, source_width - 1)
        x1 = np.clip(x_floor + 1, 0, source_width - 1)
        y0 = np.clip(y_floor, 0, source_height - 1)
        y1 = np.clip(y_floor + 1, 0, source_height - 1)

        top = (
            image[y0[:, None], x0[None, :]].astype(np.float64) * (1 - x_weight)[None, :]
            + image[y0[:, None], x1[None, :]].astype(np.float64) * x_weight[None, :]
        )
        bottom = (
            image[y1[:, None], x0[None, :]].astype(np.float64) * (1 - x_weight)[None, :]
            + image[y1[:, None], x1[None, :]].astype(np.float64) * x_weight[None, :]
        )
        values = top * (1 - y_weight)[:, None] + bottom * y_weight[:, None]
        return np.floor(values + 0.5).clip(0, 255).astype(np.uint8)

    def _center_pad(self, image: UInt8Image) -> tuple[UInt8Image, Padding]:
        height, width = image.shape
        delta_width = self.spec.image_width - width
        delta_height = self.spec.image_height - height
        left = delta_width // 2
        top = delta_height // 2
        padding = Padding(
            left=left,
            top=top,
            right=delta_width - left,
            bottom=delta_height - top,
        )
        canvas = np.full(
            (self.spec.image_height, self.spec.image_width),
            self.spec.padding_value,
            dtype=np.uint8,
        )
        canvas[top : top + height, left : left + width] = image
        return canvas, padding

    def _normalize_tensor(self, image: UInt8Image) -> Float32Tensor:
        values = image.astype(np.float32) / np.float32(self.spec.uint8_max)
        channels = [
            (values - np.float32(mean)) / np.float32(std)
            for mean, std in zip(self.spec.mean, self.spec.std, strict=True)
        ]
        return np.stack(channels, axis=0).astype(np.float32, copy=False)
