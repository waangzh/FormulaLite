"""Compare FormulaLite and a reference HGNetV2 implementation by stage."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from torch import nn

from formulalite.model import HGNetV2Model
from formulalite.model.state_mapping import map_encoder_state_dict


def _load_reference(reference_repo: Path) -> nn.Module:
    sys.path.insert(0, str(reference_repo.resolve() / "src"))
    module = importlib.import_module("texo.model.hgnet2")
    config = module.HGNetv2Config(
        stem_channels=[3, 32, 48],
        stage_config={
            "stage1": [48, 48, 128, 1, 6, 3, False, False],
            "stage2": [128, 96, 512, 1, 6, 3, True, False],
            "stage3": [512, 192, 1024, 3, 6, 5, True, True],
            "stage4": [1024, 384, 2048, 1, 6, 5, True, True],
        },
        hidden_size=2048,
    )
    return module.HGNetv2(config)


def _checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    payload: Any = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        payload = dict(payload)
    for wrapper in ("state_dict", "model_state_dict"):
        nested = payload.get(wrapper)
        if isinstance(nested, dict):
            return nested
    return payload


def _load_input(path: Path | None, image_size: int, seed: int) -> torch.Tensor:
    if path is None:
        generator = torch.Generator().manual_seed(seed)
        return torch.rand((1, 3, image_size, image_size), generator=generator)
    if path.suffix == ".npy":
        import numpy as np

        return torch.from_numpy(np.load(path))
    tensor = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("input file must contain one tensor")
    return tensor


def _capture(model: nn.Module, pixel_values: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
    outputs: OrderedDict[str, torch.Tensor] = OrderedDict()
    handles: list[Any] = []

    def save(name: str):
        def hook(_: nn.Module, __: tuple[Any, ...], output: torch.Tensor) -> None:
            outputs[name] = output.detach()

        return hook

    handles.append(model.stem.register_forward_hook(save("stem")))  # type: ignore[attr-defined]
    for index, stage in enumerate(model.stages, start=1):  # type: ignore[attr-defined]
        handles.append(stage.register_forward_hook(save(f"stage{index}")))
    try:
        with torch.no_grad():
            result = model(pixel_values)
        outputs["last_hidden_state"] = result.last_hidden_state.detach()
    finally:
        for handle in handles:
            handle.remove()
    return outputs


def _metrics(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, Any]:
    difference = (actual - reference).abs()
    denominator = max(float(torch.linalg.vector_norm(reference)), 1e-12)
    return {
        "shape": list(actual.shape),
        "max_abs_error": float(difference.max()),
        "mean_abs_error": float(difference.mean()),
        "relative_error": float(torch.linalg.vector_norm(difference)) / denominator,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formulalite", type=Path, required=True)
    parser.add_argument("--reference-checkpoint", type=Path, required=True)
    parser.add_argument("--reference-repo", type=Path, required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    formula_model = HGNetV2Model.from_pretrained(args.formulalite).eval()
    reference_model = _load_reference(args.reference_repo).eval()
    checkpoint = _checkpoint_state(args.reference_checkpoint)
    reference_state = map_encoder_state_dict(checkpoint, reference_model.state_dict())
    reference_model.load_state_dict(reference_state, strict=True)
    pixel_values = _load_input(args.input, formula_model.config.image_size, args.seed)
    reference = _capture(reference_model, pixel_values)
    actual = _capture(formula_model, pixel_values)
    report = {name: _metrics(reference[name], actual[name]) for name in reference}
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
