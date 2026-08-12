from typing import Any, cast
from unittest.mock import patch

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers.modeling_outputs import Seq2SeqLMOutput

from formulalite.data import FormulaBatch, FormulaTokenizer
from formulalite.metrics import corpus_bleu, normalized_edit_distance
from formulalite.model import FormulaLiteConfig, FormulaLiteForImageToLatex
from formulalite.training import FormulaLiteLitModule


def training_config(*, generation_metrics: bool = True) -> dict[str, object]:
    return {
        "freeze_encoder": False,
        "optimizer": {
            "lr": 1.0e-3,
            "weight_decay": 0.05,
            "betas": [0.9, 0.999],
            "eps": 1.0e-8,
        },
        "scheduler": {"warmup_steps": 2, "total_steps": 4, "min_lr": 1.0e-5},
        "generation": {"max_new_tokens": 2},
        "validation": {"compute_generation_metrics": generation_metrics},
    }


def tiny_batch() -> FormulaBatch:
    decoder_input_ids = torch.tensor([[0, 4, 5, 2, 1], [0, 6, 7, 2, 1]])
    labels = torch.tensor([[4, 5, 2, -100, -100], [6, 7, 2, -100, -100]])
    return FormulaBatch(
        pixel_values=torch.randn(2, 3, 64, 64),
        decoder_input_ids=decoder_input_ids,
        decoder_attention_mask=torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 0]]),
        labels=labels,
        sample_ids=("one", "two"),
        subsets=(None, None),
        sources=(None, None),
        raw_latex=("a b", "c d"),
        normalized_latex=("a b", "c d"),
    )


def test_training_step_preserves_labels_and_optimizer_steps(
    tiny_model_config: FormulaLiteConfig,
) -> None:
    model = FormulaLiteForImageToLatex(tiny_model_config)
    module = FormulaLiteLitModule(model, FormulaTokenizer.build(), training_config())
    batch = tiny_batch()
    labels_before = batch.labels.clone()

    model_loss = None
    original_forward = model.forward

    def capture_forward(*args: Any, **kwargs: Any) -> Seq2SeqLMOutput:
        nonlocal model_loss
        output = cast(Seq2SeqLMOutput, original_forward(*args, **kwargs))
        model_loss = output.loss
        return output

    with (
        patch.object(model, "forward", side_effect=capture_forward) as forward,
        patch.object(module, "log"),
    ):
        loss = module.training_step(batch, 0)

    assert loss is model_loss
    assert torch.equal(forward.call_args.kwargs["labels"], labels_before)
    assert torch.equal(batch.labels, labels_before)

    configured = module.configure_optimizers()
    optimizer = cast(AdamW, configured["optimizer"])
    scheduler_config = cast(dict[str, Any], configured["lr_scheduler"])
    scheduler = cast(LambdaLR, scheduler_config["scheduler"])
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    initial_lr = optimizer.param_groups[0]["lr"]
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] != initial_lr

    frozen_config = training_config()
    frozen_config["freeze_encoder"] = True
    frozen = FormulaLiteLitModule(
        FormulaLiteForImageToLatex(tiny_model_config),
        FormulaTokenizer.build(),
        frozen_config,
    )
    assert not any(parameter.requires_grad for parameter in frozen.model.encoder.parameters())
    assert all(parameter.requires_grad for parameter in frozen.model.enc_to_dec_proj.parameters())
    assert all(parameter.requires_grad for parameter in frozen.model.decoder.parameters())


def test_metrics_use_complete_prediction_and_reference_sets() -> None:
    predictions = ["a + b", "x"]
    references = ["a + b", "x y"]

    assert 0.0 < corpus_bleu(predictions, references) < 1.0
    assert normalized_edit_distance(predictions, references) == 0.25
