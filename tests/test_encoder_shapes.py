import torch

from formulalite.model import HGNetV2Config, HGNetV2Model


def test_baseline_stage_shapes_are_exact(
    baseline_encoder_config: HGNetV2Config,
) -> None:
    model = HGNetV2Model(baseline_encoder_config).eval()

    shapes = model.inspect_stage_shapes(torch.zeros(1, 3, 384, 384))

    assert shapes == {
        "stem": (1, 48, 96, 96),
        "stage1": (1, 128, 96, 96),
        "stage2": (1, 512, 48, 48),
        "stage3": (1, 1024, 24, 24),
        "stage4": (1, 2048, 12, 12),
        "flatten": (1, 144, 2048),
    }
