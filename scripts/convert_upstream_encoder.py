"""Convert a strictly matching checkpoint into a FormulaLite HF artifact."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from safetensors.torch import load_file

from formulalite.model import (
    FormulaLiteConfig,
    FormulaLiteForImageToLatex,
    HGNetV2Config,
    HGNetV2Model,
    formulalite_config_from_mapping,
    hgnetv2_config_from_mapping,
)
from formulalite.model.state_mapping import (
    map_checkpoint_state_dict,
    map_encoder_state_dict,
)


def _load_config(path: Path, scope: str) -> HGNetV2Config | FormulaLiteConfig:
    config_type = FormulaLiteConfig if scope == "model" else HGNetV2Config
    if path.is_dir():
        return config_type.from_pretrained(path)
    if path.suffix == ".json":
        return config_type.from_json_file(path)
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(raw, Mapping):
        raise TypeError("model YAML must contain a mapping")
    if scope == "model":
        return formulalite_config_from_mapping(raw)
    return hgnetv2_config_from_mapping(raw)


def _load_state(path: Path) -> Mapping[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        return load_file(path, device="cpu")
    payload: Any = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise TypeError("source checkpoint must contain a state mapping")
    for wrapper in ("state_dict", "model_state_dict"):
        nested = payload.get(wrapper)
        if isinstance(nested, Mapping):
            payload = nested
            break
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("config/model/baseline.yaml")
    )
    parser.add_argument(
        "--scope",
        choices=("encoder", "model"),
        default="encoder",
        help="Convert the standalone encoder or the complete encoder-decoder model",
    )
    parser.add_argument(
        "--source-prefix",
        default=None,
        help="Explicit prefix such as encoder.; by default common prefixes are detected",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _load_config(args.config, args.scope)
    if isinstance(config, FormulaLiteConfig):
        model: HGNetV2Model | FormulaLiteForImageToLatex = FormulaLiteForImageToLatex(
            config
        )
    else:
        model = HGNetV2Model(config)
    source_state = _load_state(args.source)
    mapping = (
        map_checkpoint_state_dict
        if isinstance(model, FormulaLiteForImageToLatex)
        else map_encoder_state_dict
    )
    mapped = mapping(source_state, model.state_dict(), source_prefix=args.source_prefix)
    model.load_state_dict(mapped, strict=True)
    model.save_pretrained(args.output)
    print(
        f"converted {len(mapped)} state entries; "
        f"parameters={sum(parameter.numel() for parameter in model.parameters())}; "
        f"output={args.output}"
    )


if __name__ == "__main__":
    main()
