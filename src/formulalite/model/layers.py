"""Independently rewritten HGNetV2 building blocks."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
from torch.nn import functional as F


def pad_right_bottom(
    tensor: torch.Tensor, *, right: int = 1, bottom: int = 1
) -> torch.Tensor:
    """Apply the explicit asymmetric zero padding used by the Paddle-style stem."""

    if right < 0 or bottom < 0:
        raise ValueError("padding values must be non-negative")
    return F.pad(tensor, (0, right, 0, bottom))


def _activation(name: str) -> Callable[[], nn.Module]:
    if name == "relu":
        return nn.ReLU
    raise ValueError(f"unsupported activation: {name}")


class ConvBNAct(nn.Module):
    """Convolution, BatchNorm, and optional activation with explicit padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        use_activation: bool = True,
        activation: str = "relu",
        batch_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if kernel_size <= 0:
            raise ValueError("kernel_size must be positive")
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels, eps=batch_norm_eps)
        self.act = _activation(activation)() if use_activation else nn.Identity()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(tensor)))


class LightConv(nn.Module):
    """Pointwise projection followed by an activated depthwise convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        activation: str = "relu",
        batch_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.conv1 = ConvBNAct(
            in_channels,
            out_channels,
            kernel_size=1,
            use_activation=False,
            activation=activation,
            batch_norm_eps=batch_norm_eps,
        )
        self.conv2 = ConvBNAct(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            groups=out_channels,
            activation=activation,
            batch_norm_eps=batch_norm_eps,
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1(tensor))


class HGBlock(nn.Module):
    """Dense HG feature aggregation block with an optional residual connection."""

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        *,
        num_layers: int,
        kernel_size: int,
        residual: bool,
        use_light_conv: bool,
        activation: str = "relu",
        batch_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if residual and in_channels != out_channels:
            raise ValueError("residual HGBlock requires matching input and output channels")
        layers: list[nn.Module] = []
        for index in range(num_layers):
            layer_in = in_channels if index == 0 else mid_channels
            if use_light_conv:
                layer: nn.Module = LightConv(
                    layer_in,
                    mid_channels,
                    kernel_size=kernel_size,
                    activation=activation,
                    batch_norm_eps=batch_norm_eps,
                )
            else:
                layer = ConvBNAct(
                    layer_in,
                    mid_channels,
                    kernel_size=kernel_size,
                    activation=activation,
                    batch_norm_eps=batch_norm_eps,
                )
            layers.append(layer)
        self.layers = nn.ModuleList(layers)
        total_channels = in_channels + num_layers * mid_channels
        self.aggregation_squeeze_conv = ConvBNAct(
            total_channels,
            out_channels // 2,
            kernel_size=1,
            activation=activation,
            batch_norm_eps=batch_norm_eps,
        )
        self.aggregation_excitation_conv = ConvBNAct(
            out_channels // 2,
            out_channels,
            kernel_size=1,
            activation=activation,
            batch_norm_eps=batch_norm_eps,
        )
        self.residual = residual

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        identity = tensor
        outputs = [tensor]
        for layer in self.layers:
            tensor = layer(tensor)
            outputs.append(tensor)
        tensor = torch.cat(outputs, dim=1)
        tensor = self.aggregation_squeeze_conv(tensor)
        tensor = self.aggregation_excitation_conv(tensor)
        if self.residual:
            tensor = tensor + identity
        return tensor


class Stem(nn.Module):
    """Paddle-compatible HGNetV2 stem with explicit right/bottom padding."""

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        *,
        activation: str = "relu",
        batch_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.stem1 = ConvBNAct(
            in_channels,
            mid_channels,
            kernel_size=3,
            stride=2,
            activation=activation,
            batch_norm_eps=batch_norm_eps,
        )
        self.stem2a = ConvBNAct(
            mid_channels,
            mid_channels // 2,
            kernel_size=2,
            activation=activation,
            batch_norm_eps=batch_norm_eps,
        )
        self.stem2b = ConvBNAct(
            mid_channels // 2,
            mid_channels,
            kernel_size=2,
            activation=activation,
            batch_norm_eps=batch_norm_eps,
        )
        self.stem3 = ConvBNAct(
            mid_channels * 2,
            mid_channels,
            kernel_size=3,
            stride=2,
            activation=activation,
            batch_norm_eps=batch_norm_eps,
        )
        self.stem4 = ConvBNAct(
            mid_channels,
            out_channels,
            kernel_size=1,
            activation=activation,
            batch_norm_eps=batch_norm_eps,
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, ceil_mode=True)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = self.stem1(tensor)
        tensor = pad_right_bottom(tensor)
        branch_conv = self.stem2a(tensor)
        branch_conv = pad_right_bottom(branch_conv)
        branch_conv = self.stem2b(branch_conv)
        branch_pool = self.pool(tensor)
        tensor = torch.cat((branch_pool, branch_conv), dim=1)
        return self.stem4(self.stem3(tensor))
