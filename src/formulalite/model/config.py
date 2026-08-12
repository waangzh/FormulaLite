"""Stable model configuration independent of Hydra and OmegaConf."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any, TypeAlias, cast

from transformers import MBartConfig, PretrainedConfig

StageValue: TypeAlias = int | bool
StageParameters: TypeAlias = dict[str, StageValue]

_STAGE_FIELDS = (
    "in_channels",
    "mid_channels",
    "out_channels",
    "num_blocks",
    "num_layers",
    "kernel_size",
    "downsample",
    "use_light_conv",
)


class HGNetV2Config(PretrainedConfig):
    """Serializable architecture configuration for the standalone encoder."""

    model_type = "formulalite_hgnetv2"
    has_no_defaults_at_init = True

    def __init__(
        self,
        *,
        image_size: int = 384,
        input_channels: int = 3,
        stem_channels: Sequence[int] = (3, 32, 48),
        stage_config: Mapping[str, Mapping[str, StageValue]] | None = None,
        hidden_size: int = 2048,
        activation: str = "relu",
        batch_norm_eps: float = 1e-5,
        initialization: str = "pytorch_default",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if stage_config is None:
            raise ValueError("stage_config must explicitly define stage1 through stage4")

        self.image_size = int(image_size)
        self.input_channels = int(input_channels)
        self.stem_channels = [int(channel) for channel in stem_channels]
        self.stage_config = _normalize_stage_config(stage_config)
        self.hidden_size = int(hidden_size)
        self.activation = activation
        self.batch_norm_eps = float(batch_norm_eps)
        self.initialization = initialization
        self._validate_architecture()

    def _validate_architecture(self) -> None:
        if self.image_size <= 0 or self.image_size % 32 != 0:
            raise ValueError("image_size must be a positive multiple of 32")
        if len(self.stem_channels) != 3:
            raise ValueError("stem_channels must contain input, middle, and output channels")
        if self.stem_channels[0] != self.input_channels:
            raise ValueError("stem_channels[0] must equal input_channels")
        if self.activation != "relu":
            raise ValueError("the HGNetV2 baseline currently supports only relu activation")
        if self.initialization != "pytorch_default":
            raise ValueError("the supported initialization is pytorch_default")
        if self.batch_norm_eps <= 0:
            raise ValueError("batch_norm_eps must be positive")

        expected_names = [f"stage{index}" for index in range(1, 5)]
        if list(self.stage_config) != expected_names:
            raise ValueError("stage_config must contain ordered stage1, stage2, stage3, stage4")

        previous_channels = self.stem_channels[-1]
        for name, stage in self.stage_config.items():
            if stage["in_channels"] != previous_channels:
                raise ValueError(f"{name}.in_channels does not match the preceding output")
            for field in (
                "in_channels",
                "mid_channels",
                "out_channels",
                "num_blocks",
                "num_layers",
                "kernel_size",
            ):
                if int(stage[field]) <= 0:
                    raise ValueError(f"{name}.{field} must be positive")
            if int(stage["kernel_size"]) % 2 == 0:
                raise ValueError(f"{name}.kernel_size must be odd")
            previous_channels = int(stage["out_channels"])

        if previous_channels != self.hidden_size:
            raise ValueError("hidden_size must equal the final stage output channels")

    def architecture_dict(self) -> dict[str, Any]:
        """Return the stable fields used to fingerprint numerical architecture."""

        return {
            "activation": self.activation,
            "batch_norm_eps": self.batch_norm_eps,
            "hidden_size": self.hidden_size,
            "image_size": self.image_size,
            "initialization": self.initialization,
            "input_channels": self.input_channels,
            "stage_config": self.stage_config,
            "stem_channels": self.stem_channels,
        }

    def architecture_sha256(self) -> str:
        payload = json.dumps(
            self.architecture_dict(), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(payload).hexdigest()


FormulaLiteHGNetV2Config = HGNetV2Config


class FormulaLiteConfig(PretrainedConfig):
    """Serializable configuration for the complete image-to-LaTeX model."""

    model_type = "formulalite"
    has_no_defaults_at_init = True

    def __init__(
        self,
        *,
        encoder_config: HGNetV2Config | Mapping[str, Any] | None = None,
        vocab_size: int = 687,
        decoder_hidden_size: int = 384,
        decoder_layers: int = 2,
        decoder_attention_heads: int = 16,
        decoder_ffn_dim: int = 1536,
        max_position_embeddings: int = 1027,
        bos_token_id: int = 0,
        pad_token_id: int = 1,
        eos_token_id: int = 2,
        unk_token_id: int = 3,
        decoder_start_token_id: int = 0,
        decoder_dropout: float = 0.1,
        decoder_attention_dropout: float = 0.0,
        decoder_activation_dropout: float = 0.0,
        decoder_layer_norm_eps: float = 1e-5,
        decoder_scale_embedding: bool = True,
        tie_word_embeddings: bool = False,
        **kwargs: Any,
    ) -> None:
        if encoder_config is None:
            raise ValueError("encoder_config must explicitly define the visual encoder")
        encoder = (
            encoder_config
            if isinstance(encoder_config, HGNetV2Config)
            else HGNetV2Config(**dict(encoder_config))
        )
        kwargs.pop("is_encoder_decoder", None)
        super().__init__(
            bos_token_id=bos_token_id,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            decoder_start_token_id=decoder_start_token_id,
            is_encoder_decoder=True,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.encoder_config = encoder.architecture_dict()
        self.vocab_size = int(vocab_size)
        self.decoder_hidden_size = int(decoder_hidden_size)
        self.decoder_layers = int(decoder_layers)
        self.decoder_attention_heads = int(decoder_attention_heads)
        self.decoder_ffn_dim = int(decoder_ffn_dim)
        self.max_position_embeddings = int(max_position_embeddings)
        self.unk_token_id = int(unk_token_id)
        self.decoder_dropout = float(decoder_dropout)
        self.decoder_attention_dropout = float(decoder_attention_dropout)
        self.decoder_activation_dropout = float(decoder_activation_dropout)
        self.decoder_layer_norm_eps = float(decoder_layer_norm_eps)
        self.decoder_scale_embedding = bool(decoder_scale_embedding)
        self._validate_complete_model()

    @property
    def encoder_hidden_size(self) -> int:
        return int(self.encoder_config["hidden_size"])

    def encoder_model_config(self) -> HGNetV2Config:
        return HGNetV2Config(**self.encoder_config)

    def decoder_model_config(self) -> MBartConfig:
        """Build the Hugging Face decoder config at the model boundary."""

        return MBartConfig(
            vocab_size=self.vocab_size,
            max_position_embeddings=self.max_position_embeddings,
            d_model=self.decoder_hidden_size,
            decoder_layers=self.decoder_layers,
            decoder_attention_heads=self.decoder_attention_heads,
            decoder_ffn_dim=self.decoder_ffn_dim,
            bos_token_id=cast(int, self.bos_token_id),
            pad_token_id=cast(int, self.pad_token_id),
            eos_token_id=cast(int, self.eos_token_id),
            decoder_start_token_id=self.decoder_start_token_id,
            forced_eos_token_id=cast(int, self.eos_token_id),
            is_decoder=True,
            is_encoder_decoder=False,
            add_cross_attention=True,
            scale_embedding=self.decoder_scale_embedding,
            tie_word_embeddings=self.tie_word_embeddings,
            dropout=self.decoder_dropout,
            attention_dropout=self.decoder_attention_dropout,
            activation_dropout=self.decoder_activation_dropout,
            layer_norm_eps=self.decoder_layer_norm_eps,
        )

    def get_text_config(
        self, decoder: bool | None = None, encoder: bool | None = None
    ) -> PretrainedConfig:
        """Expose the decoder config to Hugging Face generation utilities."""

        if decoder:
            return self.decoder_model_config()
        if encoder:
            return self.encoder_model_config()
        return self

    def _validate_complete_model(self) -> None:
        positive_fields = {
            "vocab_size": self.vocab_size,
            "decoder_hidden_size": self.decoder_hidden_size,
            "decoder_layers": self.decoder_layers,
            "decoder_attention_heads": self.decoder_attention_heads,
            "decoder_ffn_dim": self.decoder_ffn_dim,
            "max_position_embeddings": self.max_position_embeddings,
        }
        for name, value in positive_fields.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.decoder_hidden_size % self.decoder_attention_heads != 0:
            raise ValueError("decoder_hidden_size must be divisible by decoder_attention_heads")
        token_ids = {
            "bos_token_id": self.bos_token_id,
            "pad_token_id": self.pad_token_id,
            "eos_token_id": self.eos_token_id,
            "unk_token_id": self.unk_token_id,
            "decoder_start_token_id": self.decoder_start_token_id,
        }
        invalid_token_id = any(
            token_id is None or not 0 <= token_id < self.vocab_size
            for token_id in token_ids.values()
        )
        if invalid_token_id:
            raise ValueError("all special-token IDs must be inside the vocabulary")
        special_token_ids = {
            self.bos_token_id,
            self.pad_token_id,
            self.eos_token_id,
            self.unk_token_id,
        }
        if len(special_token_ids) != 4:
            raise ValueError("BOS, PAD, EOS, and UNK IDs must be distinct")
        if self.decoder_start_token_id != self.bos_token_id:
            raise ValueError("decoder_start_token_id must equal bos_token_id")
        if self.tie_word_embeddings:
            raise ValueError("FormulaLite uses an untied decoder LM head")


def _normalize_stage_config(
    stage_config: Mapping[str, Mapping[str, StageValue]],
) -> dict[str, StageParameters]:
    normalized: dict[str, StageParameters] = {}
    for name, raw_stage in stage_config.items():
        missing = set(_STAGE_FIELDS) - set(raw_stage)
        unexpected = set(raw_stage) - set(_STAGE_FIELDS)
        if missing or unexpected:
            raise ValueError(
                f"invalid fields for {name}: missing={sorted(missing)}, "
                f"unexpected={sorted(unexpected)}"
            )
        normalized[name] = {
            "in_channels": int(raw_stage["in_channels"]),
            "mid_channels": int(raw_stage["mid_channels"]),
            "out_channels": int(raw_stage["out_channels"]),
            "num_blocks": int(raw_stage["num_blocks"]),
            "num_layers": int(raw_stage["num_layers"]),
            "kernel_size": int(raw_stage["kernel_size"]),
            "downsample": bool(raw_stage["downsample"]),
            "use_light_conv": bool(raw_stage["use_light_conv"]),
        }
    return normalized


def hgnetv2_config_from_mapping(config: Mapping[str, Any]) -> HGNetV2Config:
    """Convert a resolved Hydra-style mapping at the application boundary."""

    return HGNetV2Config(
        image_size=int(config.get("image_size", config["input_size"])),
        input_channels=int(config.get("input_channels", 3)),
        stem_channels=list(config["stem_channels"]),
        stage_config={
            str(name): dict(stage) for name, stage in dict(config["stage_config"]).items()
        },
        hidden_size=int(config.get("hidden_size", config["encoder_hidden_size"])),
        activation=str(config["activation"]),
        batch_norm_eps=float(config.get("batch_norm_eps", 1e-5)),
        initialization=str(config.get("initialization", "pytorch_default")),
    )


def formulalite_config_from_mapping(config: Mapping[str, Any]) -> FormulaLiteConfig:
    """Convert a resolved Hydra-style mapping into the stable complete-model config."""

    return FormulaLiteConfig(
        encoder_config=hgnetv2_config_from_mapping(config),
        vocab_size=int(config["vocab_size"]),
        decoder_hidden_size=int(config["decoder_hidden_size"]),
        decoder_layers=int(config["decoder_layers"]),
        decoder_attention_heads=int(config["decoder_attention_heads"]),
        decoder_ffn_dim=int(config["decoder_ffn_dim"]),
        max_position_embeddings=int(config["max_position_embeddings"]),
        bos_token_id=int(config["bos_token_id"]),
        pad_token_id=int(config["pad_token_id"]),
        eos_token_id=int(config["eos_token_id"]),
        unk_token_id=int(config["unk_token_id"]),
        decoder_start_token_id=int(config["decoder_start_token_id"]),
        decoder_dropout=float(config.get("decoder_dropout", 0.1)),
        decoder_attention_dropout=float(config.get("decoder_attention_dropout", 0.0)),
        decoder_activation_dropout=float(config.get("decoder_activation_dropout", 0.0)),
        decoder_layer_norm_eps=float(config.get("decoder_layer_norm_eps", 1e-5)),
        decoder_scale_embedding=bool(config.get("decoder_scale_embedding", True)),
        tie_word_embeddings=bool(config["tie_word_embeddings"]),
    )
