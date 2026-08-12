import torch

from formulalite.model import HGNetV2Config, HGNetV2Model


def test_save_and_load_round_trip(
    tiny_encoder_config: HGNetV2Config, tmp_path
) -> None:
    torch.manual_seed(9)
    model = HGNetV2Model(tiny_encoder_config).eval()
    pixel_values = torch.randn(2, 3, 64, 64)
    expected = model(pixel_values).last_hidden_state

    model.save_pretrained(tmp_path)
    restored = HGNetV2Model.from_pretrained(tmp_path).eval()
    actual = restored(pixel_values).last_hidden_state

    assert restored.config.architecture_dict() == model.config.architecture_dict()
    assert restored.state_dict().keys() == model.state_dict().keys()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
