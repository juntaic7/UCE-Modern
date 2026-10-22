# Adapted from Universal Cell Embeddings (UCE), https://github.com/snap-stanford/UCE
# Copyright (c) 2023 Yanay Rosen, Yusuf Roohani, Jure Leskovec. MIT License.
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel
from transformers.utils import ModelOutput

from .configuration_uce import UCEConfig


def full_block(in_features: int, out_features: int, p_drop: float = 0.1) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.LayerNorm(out_features),
        nn.GELU(),
        nn.Dropout(p=p_drop),
    )


class UCEPositionalEncoding(nn.Module):
    """Fixed sinusoidal encoding; expects sequence-first input [L, B, D]."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1536):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # persistent: present in original checkpoints

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[: x.size(0)])


@dataclass
class UCEModelOutput(ModelOutput):
    """
    last_hidden_state: per-token decoder outputs, [batch, seq_len, output_dim].
    cell_embedding: L2-normalised CLS embedding, [batch, output_dim] (UCE's ``X_uce``).
    """

    last_hidden_state: Optional[torch.FloatTensor] = None
    cell_embedding: Optional[torch.FloatTensor] = None


class UCEPreTrainedModel(PreTrainedModel):
    config_class = UCEConfig
    base_model_prefix = "uce"
    main_input_name = "input_ids"
    _no_split_modules = ["TransformerEncoderLayer"]

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)


class UCEModel(UCEPreTrainedModel):
    """UCE encoder. Module names mirror the original ``TransformerModel`` so
    original state_dicts load with ``strict=True``."""

    def __init__(self, config: UCEConfig):
        super().__init__(config)
        d, out, p = config.d_model, config.output_dim, config.dropout
        self.d_model = d

        # Frozen token table: normalised ESM2 protein embeddings + chromosome tokens.
        self.pe_embedding = nn.Embedding(config.vocab_size, config.token_dim)
        self.pe_embedding.weight.requires_grad_(False)

        self.encoder = nn.Sequential(nn.Linear(config.token_dim, d), nn.GELU(), nn.LayerNorm(d))
        self.pos_encoder = UCEPositionalEncoding(d, p, config.max_position_embeddings)
        layer = nn.TransformerEncoderLayer(d, config.nhead, dim_feedforward=config.d_hid, dropout=p)
        self.transformer_encoder = nn.TransformerEncoder(layer, config.nlayers, enable_nested_tensor=False)
        self.decoder = nn.Sequential(
            full_block(d, 1024, p),
            full_block(1024, out, p),
            full_block(out, out, p),
            nn.Linear(out, out),
        )
        # Expression-prediction head used in pretraining (see ``predict``).
        self.binary_decoder = nn.Sequential(
            full_block(out + d, 2048, p),
            full_block(2048, 512, p),
            full_block(512, 128, p),
            nn.Linear(128, 1),
        )
        self.gene_embedding_layer = nn.Sequential(nn.Linear(config.token_dim, d), nn.GELU(), nn.LayerNorm(d))

        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.pe_embedding

    def set_input_embeddings(self, value: nn.Embedding):
        self.pe_embedding = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        """
        input_ids: [batch, seq_len] token indices (CLS first), as produced by UCE's
            cell-sentence sampler.
        attention_mask: [batch, seq_len], 1 for real tokens and 0 for padding.
        inputs_embeds: optional [batch, seq_len, token_dim] instead of input_ids.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Pass exactly one of input_ids or inputs_embeds.")
        if inputs_embeds is None:
            inputs_embeds = F.normalize(self.pe_embedding(input_ids.long()), dim=-1)
        if attention_mask is None:
            attention_mask = torch.ones(inputs_embeds.shape[:2], device=inputs_embeds.device)

        src = inputs_embeds.transpose(0, 1)  # -> [L, B, token_dim]
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)

        if self.config.legacy_float_padding_mask:
            key_padding_mask = (1 - attention_mask).to(src.dtype)
        else:
            key_padding_mask = attention_mask == 0
        hidden = self.transformer_encoder(src, src_key_padding_mask=key_padding_mask)

        gene_output = self.decoder(hidden)                     # [L, B, output_dim]
        cell_embedding = F.normalize(gene_output[0], dim=1)    # CLS token
        gene_output = gene_output.transpose(0, 1)              # -> [B, L, output_dim]

        if not return_dict:
            return gene_output, cell_embedding
        return UCEModelOutput(last_hidden_state=gene_output, cell_embedding=cell_embedding)

    def predict(self, cell_embedding: torch.Tensor, gene_embeddings: torch.Tensor) -> torch.Tensor:
        """Expression logits for genes given a cell embedding (pretraining head).

        cell_embedding must already be broadcast to gene_embeddings' leading dims.
        """
        gene_embeddings = self.gene_embedding_layer(gene_embeddings)
        return self.binary_decoder(torch.cat((cell_embedding, gene_embeddings), dim=-1))
