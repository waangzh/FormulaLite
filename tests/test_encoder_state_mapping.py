from collections import OrderedDict

import pytest
import torch

from formulalite.model import HGNetV2Config, HGNetV2Model
from formulalite.model.state_mapping import (
    StateMappingError,
    map_encoder_state_dict,
)


def _prefixed_state(model: HGNetV2Model) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        (f"encoder.{name}", torch.full_like(value, index % 7))
        for index, (name, value) in enumerate(model.state_dict().items())
    )


def test_strict_mapping_strips_encoder_prefix_and_preserves_order(
    tiny_encoder_config: HGNetV2Config,
) -> None:
    model = HGNetV2Model(tiny_encoder_config)
    source = _prefixed_state(model)
    source["decoder.fake.weight"] = torch.zeros(1)

    mapped = map_encoder_state_dict(source, model.state_dict())

    assert list(mapped) == list(model.state_dict())
    model.load_state_dict(mapped, strict=True)


def test_strict_mapping_rejects_missing_key(tiny_encoder_config: HGNetV2Config) -> None:
    model = HGNetV2Model(tiny_encoder_config)
    source = _prefixed_state(model)
    source.pop(next(iter(source)))

    with pytest.raises(StateMappingError, match="missing"):
        map_encoder_state_dict(source, model.state_dict())


def test_strict_mapping_rejects_unexpected_encoder_key(
    tiny_encoder_config: HGNetV2Config,
) -> None:
    model = HGNetV2Model(tiny_encoder_config)
    source = _prefixed_state(model)
    source["encoder.unexpected.weight"] = torch.zeros(1)

    with pytest.raises(StateMappingError, match="unexpected"):
        map_encoder_state_dict(source, model.state_dict())


def test_strict_mapping_rejects_shape_mismatch(
    tiny_encoder_config: HGNetV2Config,
) -> None:
    model = HGNetV2Model(tiny_encoder_config)
    source = _prefixed_state(model)
    first_name = next(iter(source))
    source[first_name] = torch.zeros(1)

    with pytest.raises(StateMappingError, match="shape mismatch"):
        map_encoder_state_dict(source, model.state_dict())
