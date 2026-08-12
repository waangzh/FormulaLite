import json

import pytest

from formulalite.model import HGNetV2Config


def test_config_serialization_round_trip(
    tiny_encoder_config: HGNetV2Config, tmp_path
) -> None:
    tiny_encoder_config.save_pretrained(tmp_path)
    restored = HGNetV2Config.from_pretrained(tmp_path)

    assert restored.model_type == "formulalite_hgnetv2"
    assert restored.architecture_dict() == tiny_encoder_config.architecture_dict()
    assert restored.architecture_sha256() == tiny_encoder_config.architecture_sha256()
    serialized = json.loads((tmp_path / "config.json").read_text())
    assert serialized["stage_config"]["stage3"]["use_light_conv"] is True


def test_hydra_conversion_returns_stable_config_object(
    baseline_encoder_config: HGNetV2Config,
) -> None:
    assert type(baseline_encoder_config) is HGNetV2Config
    assert baseline_encoder_config.hidden_size == 2048
    assert baseline_encoder_config.stage_config["stage3"] == {
        "in_channels": 512,
        "mid_channels": 192,
        "out_channels": 1024,
        "num_blocks": 3,
        "num_layers": 6,
        "kernel_size": 5,
        "downsample": True,
        "use_light_conv": True,
    }


def test_config_rejects_inconsistent_channel_chain(
    tiny_encoder_config: HGNetV2Config,
) -> None:
    invalid = tiny_encoder_config.architecture_dict()
    invalid["stage_config"]["stage2"]["in_channels"] = 999

    with pytest.raises(ValueError, match="preceding output"):
        HGNetV2Config(**invalid)
