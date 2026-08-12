"""Standalone Hugging Face-compatible FormulaLite HGNetV2 encoder."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from typing import Any, cast

import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput

from .config import HGNetV2Config, StageParameters
from .layers import ConvBNAct, HGBlock, Stem


class HGStage(nn.Module):
    """One optional depthwise downsample followed by one or more HG blocks."""

    def __init__(
        self,
        parameters: StageParameters,
        *,
        activation: str,
        batch_norm_eps: float,
    ) -> None:
        super().__init__()
        in_channels = int(parameters["in_channels"])
        self.downsample: nn.Module
        if bool(parameters["downsample"]):
            self.downsample = ConvBNAct(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=2,
                groups=in_channels,
                use_activation=False,
                activation=activation,
                batch_norm_eps=batch_norm_eps,
            )
        else:
            self.downsample = nn.Identity()

        out_channels = int(parameters["out_channels"])
        blocks: list[nn.Module] = []
        for index in range(int(parameters["num_blocks"])):
            blocks.append(
                HGBlock(
                    in_channels if index == 0 else out_channels,
                    int(parameters["mid_channels"]),
                    out_channels,
                    num_layers=int(parameters["num_layers"]),
                    kernel_size=int(parameters["kernel_size"]),
                    residual=index > 0,
                    use_light_conv=bool(parameters["use_light_conv"]),
                    activation=activation,
                    batch_norm_eps=batch_norm_eps,
                )
            )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.downsample(tensor))


class HGNetV2Model(PreTrainedModel):
    """FormulaLite's independent numerical rewrite of the HGNetV2-B4-style backbone."""

    config_class = HGNetV2Config
    base_model_prefix = "formulalite_hgnetv2"
    main_input_name = "pixel_values"

    def __init__(self, config: HGNetV2Config) -> None:
        super().__init__(config)
        self.stem = Stem(
            *config.stem_channels,
            activation=config.activation,
            batch_norm_eps=config.batch_norm_eps,
        )
        self.stages = nn.ModuleList(
            HGStage(
                parameters,
                activation=config.activation,
                batch_norm_eps=config.batch_norm_eps,
            )
            for parameters in config.stage_config.values()
        )
        # Keep the native PyTorch Conv2d and BatchNorm2d initialization contract.

    def _validate_pixel_values(self, pixel_values: torch.Tensor) -> None:
        if pixel_values.ndim != 4:
            raise ValueError("pixel_values must have shape [batch, channels, height, width]")
        expected = (self.config.input_channels, self.config.image_size, self.config.image_size)
        if tuple(pixel_values.shape[1:]) != expected:
            raise ValueError(
                f"pixel_values must have trailing shape {expected}, got "
                f"{tuple(pixel_values.shape[1:])}"
            )
        if not pixel_values.is_floating_point():
            raise TypeError("pixel_values must use a floating-point dtype")

    def _feature_maps(self, pixel_values: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        self._validate_pixel_values(pixel_values)
        outputs: OrderedDict[str, torch.Tensor] = OrderedDict()
        tensor = self.stem(pixel_values)
        outputs["stem"] = tensor
        for index, stage in enumerate(self.stages, start=1):
            tensor = stage(tensor)
            outputs[f"stage{index}"] = tensor
        outputs["flatten"] = tensor.flatten(2).transpose(1, 2)
        return outputs

    @torch.no_grad()
    def inspect_stage_shapes(
        self, pixel_values: torch.Tensor
    ) -> Mapping[str, tuple[int, ...]]:
        """Return debug-only stage shapes without changing the production forward API."""

        return {
            name: tuple(value.shape)
            for name, value in self._feature_maps(pixel_values).items()
        }

    def forward(
        self,
        pixel_values: torch.Tensor,
        return_dict: bool | None = None,
        **_: Any,
    ) -> BaseModelOutput | tuple[torch.Tensor]:
        return_dict = self.config.use_return_dict if return_dict is None else return_dict
        last_hidden_state = self._feature_maps(pixel_values)["flatten"]
        if not return_dict:
            return (last_hidden_state,)
        return BaseModelOutput(last_hidden_state=cast(Any, last_hidden_state))


FormulaLiteHGNetV2Model = HGNetV2Model
