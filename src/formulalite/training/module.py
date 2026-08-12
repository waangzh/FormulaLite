"""Lightning orchestration around the Phase 5 model contract."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, cast

import lightning as L
import torch
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR

from formulalite.data import FormulaBatch, FormulaTokenizer
from formulalite.data.normalizer import normalize
from formulalite.metrics import corpus_bleu, normalized_edit_distance
from formulalite.model import FormulaLiteForImageToLatex


class FormulaLiteLitModule(L.LightningModule):
    """Own optimization and logging while delegating modeling to FormulaLite."""

    def __init__(
        self,
        model: FormulaLiteForImageToLatex,
        tokenizer: FormulaTokenizer,
        training_config: Mapping[str, Any],
    ) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.training_config = dict(training_config)
        self._validation_predictions: list[str] = []
        self._validation_references: list[str] = []
        self.save_hyperparameters({"training_config": self.training_config})

        if bool(self.training_config.get("freeze_encoder", False)):
            for parameter in self.model.encoder.parameters():
                parameter.requires_grad = False

    def _forward_loss(self, batch: FormulaBatch) -> torch.Tensor:
        output = self.model(
            pixel_values=batch.pixel_values,
            decoder_input_ids=batch.decoder_input_ids,
            decoder_attention_mask=batch.decoder_attention_mask,
            labels=batch.labels,
        )
        if output.loss is None:
            raise RuntimeError("FormulaLite model did not return a training loss")
        return output.loss

    def training_step(self, batch: FormulaBatch, batch_idx: int) -> torch.Tensor:
        del batch_idx
        loss = self._forward_loss(batch)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            batch_size=batch.pixel_values.shape[0],
        )
        return loss

    def on_validation_epoch_start(self) -> None:
        self._validation_predictions.clear()
        self._validation_references.clear()

    def validation_step(self, batch: FormulaBatch, batch_idx: int) -> torch.Tensor:
        del batch_idx
        loss = self._forward_loss(batch)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch.pixel_values.shape[0],
            sync_dist=True,
        )
        validation = cast(Mapping[str, Any], self.training_config["validation"])
        if bool(validation["compute_generation_metrics"]):
            generation = cast(Mapping[str, Any], self.training_config["generation"])
            generated_ids = self.model.generate(
                batch.pixel_values,
                max_new_tokens=int(generation["max_new_tokens"]),
                num_beams=1,
                do_sample=False,
            )
            self._validation_predictions.extend(
                normalize(self.tokenizer.decode(token_ids)) for token_ids in generated_ids.tolist()
            )
            self._validation_references.extend(batch.normalized_latex)
        return loss

    def on_validation_epoch_end(self) -> None:
        if self._validation_predictions:
            bleu = corpus_bleu(
                self._validation_predictions,
                self._validation_references,
            )
            edit_distance = normalized_edit_distance(
                self._validation_predictions,
                self._validation_references,
            )
            self.log("BLEU", bleu, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(
                "edit_distance",
                edit_distance,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
        self.log("epoch", float(self.current_epoch), on_step=False, on_epoch=True)

    def test_step(self, batch: FormulaBatch, batch_idx: int) -> torch.Tensor:
        del batch_idx
        loss = self._forward_loss(batch)
        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            batch_size=batch.pixel_values.shape[0],
            sync_dist=True,
        )
        return loss

    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
        self.log("lr", optimizer.param_groups[0]["lr"], on_step=True, on_epoch=False)

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        optimizer_config = cast(Mapping[str, Any], self.training_config["optimizer"])
        scheduler_config = cast(Mapping[str, Any], self.training_config["scheduler"])
        learning_rate = float(optimizer_config["lr"])
        min_lr = float(scheduler_config["min_lr"])
        warmup_steps = int(scheduler_config["warmup_steps"])
        total_steps = int(scheduler_config["total_steps"])
        if learning_rate <= 0 or not 0 <= min_lr <= learning_rate:
            raise ValueError(
                "optimizer.lr must be positive and scheduler.min_lr must be in [0, lr]"
            )
        if warmup_steps < 0 or total_steps <= warmup_steps:
            raise ValueError("scheduler.total_steps must be greater than warmup_steps >= 0")

        betas = tuple(float(value) for value in optimizer_config["betas"])
        if len(betas) != 2:
            raise ValueError("optimizer.betas must contain exactly two values")
        optimizer = AdamW(
            (parameter for parameter in self.parameters() if parameter.requires_grad),
            lr=learning_rate,
            weight_decay=float(optimizer_config["weight_decay"]),
            betas=cast(tuple[float, float], betas),
            eps=float(optimizer_config["eps"]),
        )
        minimum_ratio = min_lr / learning_rate

        def lr_multiplier(step: int) -> float:
            if step < warmup_steps:
                return (step + 1) / max(warmup_steps, 1)
            progress = min((step - warmup_steps) / (total_steps - warmup_steps), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return minimum_ratio + (1.0 - minimum_ratio) * cosine

        scheduler = LambdaLR(optimizer, lr_lambda=lr_multiplier)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
