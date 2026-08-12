from formulalite.model import HGNetV2Config, HGNetV2Model

BASELINE_ENCODER_PARAMETER_COUNT = 13_553_376


def test_baseline_encoder_parameter_count(
    baseline_encoder_config: HGNetV2Config,
) -> None:
    model = HGNetV2Model(baseline_encoder_config)

    assert sum(parameter.numel() for parameter in model.parameters()) == (
        BASELINE_ENCODER_PARAMETER_COUNT
    )
