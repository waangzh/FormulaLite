import math

import torch
from torch import nn

from formulalite.model import ConvBNAct, HGBlock, LightConv, Stem
from formulalite.model.layers import pad_right_bottom


def test_asymmetric_right_bottom_padding_is_exact() -> None:
    tensor = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])

    actual = pad_right_bottom(tensor)
    expected = torch.tensor([[[[1.0, 2.0, 0.0], [3.0, 4.0, 0.0], [0.0, 0.0, 0.0]]]])

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_stem_uses_two_explicit_asymmetric_padding_steps() -> None:
    stem = Stem(3, 8, 16).eval()
    observed: dict[str, tuple[int, ...]] = {}
    handles = [
        stem.stem1.register_forward_hook(
            lambda _module, _args, output: observed.__setitem__("stem1", tuple(output.shape))
        ),
        stem.stem2a.register_forward_pre_hook(
            lambda _module, args: observed.__setitem__("stem2a_input", tuple(args[0].shape))
        ),
        stem.stem2b.register_forward_pre_hook(
            lambda _module, args: observed.__setitem__("stem2b_input", tuple(args[0].shape))
        ),
    ]
    try:
        output = stem(torch.zeros(1, 3, 64, 64))
    finally:
        for handle in handles:
            handle.remove()

    assert observed == {
        "stem1": (1, 8, 32, 32),
        "stem2a_input": (1, 8, 33, 33),
        "stem2b_input": (1, 4, 33, 33),
    }
    assert output.shape == (1, 16, 16, 16)


def test_light_conv_and_hg_block_contracts() -> None:
    light = LightConv(8, 12, kernel_size=5)
    assert light.conv2.conv.groups == 12
    block = HGBlock(
        12,
        8,
        12,
        num_layers=2,
        kernel_size=5,
        residual=True,
        use_light_conv=True,
    ).eval()

    output = block(torch.zeros(2, 12, 7, 7))

    assert output.shape == (2, 12, 7, 7)


def test_pytorch_default_initialization_contract() -> None:
    layer = ConvBNAct(3, 8, kernel_size=3)
    fan_in = 3 * 3 * 3
    expected_bound = 1 / math.sqrt(fan_in)

    assert float(layer.conv.weight.detach().abs().max()) <= expected_bound
    torch.testing.assert_close(layer.bn.weight, torch.ones(8), rtol=0, atol=0)
    torch.testing.assert_close(layer.bn.bias, torch.zeros(8), rtol=0, atol=0)
    assert isinstance(layer.act, nn.ReLU)
