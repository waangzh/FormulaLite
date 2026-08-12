"""Hydra entry point for FormulaLite training."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import hydra
import lightning as L
from hydra.utils import to_absolute_path
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig, OmegaConf

from formulalite.data import FormulaDataModule, FormulaTokenizer
from formulalite.model import (
    FormulaLiteForImageToLatex,
    formulalite_config_from_mapping,
)
from formulalite.training import FormulaLiteLitModule


def _mapping(config: Any) -> dict[str, Any]:
    value = OmegaConf.to_container(config, resolve=True)
    if not isinstance(value, dict):
        raise TypeError("configuration section must resolve to a mapping")
    return {str(key): item for key, item in value.items()}


def _absolute_optional_path(value: Any) -> str | None:
    return None if value is None else to_absolute_path(str(value))


def _initialization_mode(training: Mapping[str, Any]) -> str:
    configured = str(training["init_mode"])
    pretrained_path = training.get("pretrained_path")
    resume_path = training.get("resume_from_checkpoint")
    if resume_path is not None:
        if configured == "pretrained" or pretrained_path is not None:
            raise ValueError("resume and pretrained initialization cannot be combined")
        if configured not in {"scratch", "resume"}:
            raise ValueError(f"unsupported training.init_mode: {configured}")
        return "resume"
    if configured == "resume":
        raise ValueError("training.init_mode=resume requires resume_from_checkpoint")
    if configured == "pretrained" and pretrained_path is None:
        raise ValueError("training.init_mode=pretrained requires pretrained_path")
    if configured == "scratch" and pretrained_path is not None:
        raise ValueError("pretrained_path is only valid with training.init_mode=pretrained")
    if configured not in {"scratch", "pretrained"}:
        raise ValueError(f"unsupported training.init_mode: {configured}")
    return configured


def _build_model(
    model_config: Mapping[str, Any], training: Mapping[str, Any]
) -> FormulaLiteForImageToLatex:
    mode = _initialization_mode(training)
    if mode == "pretrained":
        path = cast(str, _absolute_optional_path(training["pretrained_path"]))
        return FormulaLiteForImageToLatex.from_pretrained(path)
    return FormulaLiteForImageToLatex(formulalite_config_from_mapping(model_config))


def _checkpoint_callback(training: Mapping[str, Any]) -> ModelCheckpoint:
    validation = cast(Mapping[str, Any], training["validation"])
    checkpoint = cast(Mapping[str, Any], training["checkpoint"])
    generation_metrics = bool(validation["compute_generation_metrics"])
    monitor = "BLEU" if generation_metrics else "val_loss"
    mode = "max" if generation_metrics else "min"
    metric_field = "{BLEU:.4f}" if generation_metrics else "{val_loss:.4f}"
    return ModelCheckpoint(
        dirpath=_absolute_optional_path(checkpoint.get("dirpath")),
        filename=f"best-{{epoch:02d}}-{{step}}-{metric_field}",
        monitor=monitor,
        mode=mode,
        save_top_k=int(checkpoint["save_top_k"]),
        save_last=True,
        auto_insert_metric_name=False,
        save_weights_only=False,
    )


def run(config: DictConfig) -> L.Trainer:
    """Construct the configured training graph and run fit."""

    L.seed_everything(int(config.seed), workers=True)
    model_config = _mapping(config.model)
    training = _mapping(config.training)
    data_config = _mapping(config.data)
    for field in ("train_manifest", "val_manifest", "test_manifest"):
        data_config[field] = _absolute_optional_path(data_config.get(field))

    tokenizer = FormulaTokenizer.build()
    data_module = FormulaDataModule(data_config, tokenizer=tokenizer)
    model = _build_model(model_config, training)
    configured_image_size = int(data_config["image_size"])
    if configured_image_size != model.config.encoder_model_config().image_size:
        raise ValueError("data.image_size must match the selected model encoder image_size")
    lit_module = FormulaLiteLitModule(model, tokenizer, training)

    logging = cast(Mapping[str, Any], training["logging"])
    logger = TensorBoardLogger(
        save_dir=to_absolute_path(str(logging["save_dir"])),
        name=str(logging["name"]),
    )
    trainer_config = _mapping(config.trainer)
    trainer_config.pop("name", None)
    trainer = L.Trainer(
        **trainer_config,
        logger=logger,
        callbacks=[_checkpoint_callback(training)],
    )
    resume_path = _absolute_optional_path(training.get("resume_from_checkpoint"))
    trainer.fit(lit_module, datamodule=data_module, ckpt_path=resume_path)
    return trainer


def main(overrides: list[str] | None = None) -> None:
    """Compose Hydra overrides without depending on Hydra's process argument parser."""

    config_dir = Path(__file__).parents[2] / "config"
    cli_overrides = sys.argv[1:] if overrides is None else overrides
    with hydra.initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        config = hydra.compose(config_name="train", overrides=cli_overrides)
    run(config)


if __name__ == "__main__":
    main()
