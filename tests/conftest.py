from pathlib import Path
from typing import Any, cast

import pytest
from omegaconf import OmegaConf

from formulalite.model import (
    FormulaLiteConfig,
    HGNetV2Config,
    formulalite_config_from_mapping,
    hgnetv2_config_from_mapping,
)

CONFIG_DIR = Path(__file__).parents[1] / "config" / "model"


def _load_model_config(name: str) -> HGNetV2Config:
    raw = _load_model_mapping(name)
    return hgnetv2_config_from_mapping(raw)


def _load_model_mapping(name: str) -> dict[str, Any]:
    raw = OmegaConf.to_container(OmegaConf.load(CONFIG_DIR / f"{name}.yaml"), resolve=True)
    if not isinstance(raw, dict):
        raise TypeError("test model config must be a mapping")
    return cast(dict[str, Any], raw)


@pytest.fixture
def tiny_encoder_config() -> HGNetV2Config:
    return _load_model_config("tiny")


@pytest.fixture
def baseline_encoder_config() -> HGNetV2Config:
    return _load_model_config("baseline")


@pytest.fixture
def tiny_model_config() -> FormulaLiteConfig:
    return formulalite_config_from_mapping(_load_model_mapping("tiny"))


@pytest.fixture
def baseline_model_config() -> FormulaLiteConfig:
    return formulalite_config_from_mapping(_load_model_mapping("baseline"))
