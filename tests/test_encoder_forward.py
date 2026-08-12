import torch
from transformers.modeling_outputs import BaseModelOutput

from formulalite.model import HGNetV2Config, HGNetV2Model


def test_tiny_forward_batch_and_return_dict_contract(
    tiny_encoder_config: HGNetV2Config,
) -> None:
    model = HGNetV2Model(tiny_encoder_config).eval()
    pixel_values = torch.randn(3, 3, 64, 64, dtype=torch.float32)

    output = model(pixel_values)
    tuple_output = model(pixel_values, return_dict=False)

    assert isinstance(output, BaseModelOutput)
    assert output.last_hidden_state is not None
    assert output.last_hidden_state.shape == (3, 4, 64)
    assert output.last_hidden_state.dtype == torch.float32
    assert len(tuple_output) == 1
    torch.testing.assert_close(tuple_output[0], output.last_hidden_state)


def test_tiny_forward_batch_one(tiny_encoder_config: HGNetV2Config) -> None:
    model = HGNetV2Model(tiny_encoder_config).eval()

    output = model(torch.zeros(1, 3, 64, 64))

    assert output.last_hidden_state.shape == (1, 4, 64)


def test_eval_forward_is_deterministic(tiny_encoder_config: HGNetV2Config) -> None:
    torch.manual_seed(123)
    model = HGNetV2Model(tiny_encoder_config).eval()
    pixel_values = torch.randn(2, 3, 64, 64)

    first = model(pixel_values).last_hidden_state
    second = model(pixel_values).last_hidden_state

    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_seeded_model_construction_is_reproducible(
    tiny_encoder_config: HGNetV2Config,
) -> None:
    torch.manual_seed(77)
    first = HGNetV2Model(tiny_encoder_config).eval()
    torch.manual_seed(77)
    second = HGNetV2Model(tiny_encoder_config).eval()
    pixel_values = torch.randn(1, 3, 64, 64)

    for first_value, second_value in zip(
        first.state_dict().values(), second.state_dict().values(), strict=True
    ):
        torch.testing.assert_close(first_value, second_value, rtol=0, atol=0)
    torch.testing.assert_close(
        first(pixel_values).last_hidden_state,
        second(pixel_values).last_hidden_state,
        rtol=0,
        atol=0,
    )
