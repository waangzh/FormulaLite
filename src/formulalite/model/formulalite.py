"""Complete FormulaLite image-to-LaTeX encoder-decoder model."""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import nn
from torch.nn import functional as functional
from transformers import MBartForCausalLM, PreTrainedModel
from transformers.modeling_outputs import Seq2SeqLMOutput

from .config import FormulaLiteConfig
from .hgnetv2 import HGNetV2Model


class FormulaLiteForImageToLatex(PreTrainedModel):
    """HGNetV2 encoder with a projected compact MBART causal decoder."""

    config_class = FormulaLiteConfig
    base_model_prefix = "formulalite"
    main_input_name = "pixel_values"

    def __init__(self, config: FormulaLiteConfig) -> None:
        super().__init__(config)
        self.encoder = HGNetV2Model(config.encoder_model_config())
        self.enc_to_dec_proj = nn.Linear(
            config.encoder_hidden_size, config.decoder_hidden_size
        )
        self.decoder = MBartForCausalLM(config.decoder_model_config())

    @classmethod
    def can_generate(cls) -> bool:
        return True

    def get_encoder(self) -> HGNetV2Model:
        return self.encoder

    def get_decoder(self) -> MBartForCausalLM:
        return self.decoder

    def get_output_embeddings(self) -> nn.Module:
        return cast(nn.Module, self.decoder.get_output_embeddings())

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        self.decoder.set_output_embeddings(new_embeddings)

    def forward(
        self,
        pixel_values: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        decoder_attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        return_dict: bool | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        **_: Any,
    ) -> Seq2SeqLMOutput | tuple[torch.Tensor, ...]:
        """Predict each supplied label at the same sequence position as its logit."""

        return_dict = self.config.use_return_dict if return_dict is None else return_dict
        encoder_outputs = self.encoder(pixel_values=pixel_values, return_dict=True)
        encoder_hidden_state = encoder_outputs.last_hidden_state
        projected_hidden_state = self.enc_to_dec_proj(encoder_hidden_state)

        # Labels are intentionally omitted here. The data contract already aligns each
        # target with its decoder input position, so the complete model computes loss.
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=projected_hidden_state,
            labels=None,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        logits = decoder_outputs.logits
        loss = None
        if labels is not None:
            if labels.shape != decoder_input_ids.shape:
                raise ValueError("labels must have the same shape as decoder_input_ids")
            loss = functional.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                labels.to(device=logits.device).reshape(-1),
                ignore_index=-100,
            )

        if not return_dict:
            values = (logits, encoder_hidden_state)
            return (loss, *values) if loss is not None else values

        return Seq2SeqLMOutput(
            loss=cast(Any, loss),
            logits=logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=cast(Any, encoder_hidden_state),
        )

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        *,
        max_new_tokens: int = 128,
        num_beams: int = 1,
        do_sample: bool = False,
        **kwargs: Any,
    ) -> torch.LongTensor:
        """Greedily decode from BOS using only the encoded image and a safety limit."""

        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if num_beams != 1 or do_sample:
            raise ValueError("Phase 5 generation supports greedy decoding only")

        encoder_hidden_state = self.encoder(
            pixel_values=pixel_values, return_dict=True
        ).last_hidden_state
        projected_hidden_state = self.enc_to_dec_proj(encoder_hidden_state)
        start_token_id = self.config.decoder_start_token_id
        if start_token_id is None:
            raise RuntimeError("decoder_start_token_id must be configured")
        input_ids = torch.full(
            (pixel_values.shape[0], 1),
            start_token_id,
            dtype=torch.long,
            device=pixel_values.device,
        )
        generated = self.decoder.generate(
            input_ids=input_ids,
            encoder_hidden_states=projected_hidden_state,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=do_sample,
            bos_token_id=self.config.bos_token_id,
            pad_token_id=self.config.pad_token_id,
            eos_token_id=self.config.eos_token_id,
            **kwargs,
        )
        return cast(torch.LongTensor, generated)


FormulaLiteModel = FormulaLiteForImageToLatex
