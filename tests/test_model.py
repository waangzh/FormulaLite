from pathlib import Path
from unittest.mock import patch

import torch
from torch.nn import functional as functional

from formulalite.model import (
    FormulaLiteConfig,
    FormulaLiteForImageToLatex,
    map_checkpoint_state_dict,
)


def test_decoder_config_and_shape(tiny_model_config: FormulaLiteConfig) -> None:
    model = FormulaLiteForImageToLatex(tiny_model_config).eval()
    config = model.decoder.config
    decoder_input_ids = torch.tensor([[0, 4, 5, 2, 1], [0, 6, 2, 1, 1]])
    projected_encoder = torch.randn(2, 4, tiny_model_config.decoder_hidden_size)

    output = model.decoder(
        input_ids=decoder_input_ids,
        encoder_hidden_states=projected_encoder,
        return_dict=True,
    )

    assert config.vocab_size == 687
    assert config.d_model == 32
    assert config.decoder_layers == 1
    assert config.decoder_attention_heads == 4
    assert config.decoder_ffn_dim == 64
    assert config.is_decoder is True
    assert config.add_cross_attention is True
    assert config.tie_word_embeddings is False
    assert model.decoder.model.decoder.embed_tokens.weight.data_ptr() != (
        model.decoder.lm_head.weight.data_ptr()
    )
    assert output.logits.shape == (2, 5, 687)


def test_projection_shape(baseline_model_config: FormulaLiteConfig) -> None:
    model = FormulaLiteForImageToLatex(baseline_model_config).eval()
    encoder_hidden_state = torch.randn(1, 144, 2048)

    projected = model.enc_to_dec_proj(encoder_hidden_state)

    assert model.enc_to_dec_proj.in_features == 2048
    assert model.enc_to_dec_proj.out_features == 384
    assert projected.shape == (1, 144, 384)


def test_model_forward(tiny_model_config: FormulaLiteConfig) -> None:
    model = FormulaLiteForImageToLatex(tiny_model_config).eval()
    pixel_values = torch.randn(2, 3, 64, 64)
    decoder_input_ids = torch.tensor([[0, 4, 5, 2, 1], [0, 6, 7, 2, 1]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 0]])
    labels = torch.tensor([[4, 5, 2, -100, -100], [6, 7, 2, -100, -100]])

    output = model(
        pixel_values=pixel_values,
        decoder_input_ids=decoder_input_ids,
        decoder_attention_mask=attention_mask,
        labels=labels,
    )

    assert output.loss is not None and output.loss.ndim == 0
    assert output.logits.shape == (2, 5, 687)
    assert output.encoder_last_hidden_state is not None
    assert output.encoder_last_hidden_state.shape == (2, 4, 64)


def test_model_respects_pre_shifted_labels(tiny_model_config: FormulaLiteConfig) -> None:
    model = FormulaLiteForImageToLatex(tiny_model_config).eval()
    decoder_input_ids = torch.tensor([[0, 4, 5, 2, 1]])
    labels = torch.tensor([[4, 5, 2, -100, -100]])

    with patch.object(model.decoder, "forward", wraps=model.decoder.forward) as decoder_forward:
        output = model(
            pixel_values=torch.randn(1, 3, 64, 64),
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=torch.tensor([[1, 1, 1, 1, 0]]),
            labels=labels,
        )

    forwarded = decoder_forward.call_args.kwargs
    assert torch.equal(forwarded["input_ids"], decoder_input_ids)
    assert forwarded["labels"] is None
    assert labels[labels != -100].tolist() == [4, 5, 2]
    expected_loss = functional.cross_entropy(output.logits[0, :3], labels[0, :3])
    torch.testing.assert_close(output.loss, expected_loss)


def test_model_save_load_and_generate(
    tiny_model_config: FormulaLiteConfig, tmp_path: Path
) -> None:
    torch.manual_seed(13)
    model = FormulaLiteForImageToLatex(tiny_model_config).eval()
    pixel_values = torch.randn(1, 3, 64, 64)
    decoder_input_ids = torch.tensor([[0, 4, 5, 2, 1]])
    expected = model(pixel_values=pixel_values, decoder_input_ids=decoder_input_ids).logits

    model.save_pretrained(tmp_path)
    restored = FormulaLiteForImageToLatex.from_pretrained(tmp_path).eval()
    wrapped_state = {
        f"module.{name}": value for name, value in model.state_dict().items()
    }
    mapped_state = map_checkpoint_state_dict(wrapped_state, restored.state_dict())
    restored.load_state_dict(mapped_state, strict=True)
    actual = restored(pixel_values=pixel_values, decoder_input_ids=decoder_input_ids).logits
    generated = restored.generate(pixel_values, max_new_tokens=4)

    assert (tmp_path / "config.json").is_file()
    assert (tmp_path / "generation_config.json").is_file()
    assert restored.state_dict().keys() == model.state_dict().keys()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert generated.ndim == 2 and generated.shape[0] == 1
    assert 2 <= generated.shape[1] <= 5
    assert generated[0, 0].item() == 0
