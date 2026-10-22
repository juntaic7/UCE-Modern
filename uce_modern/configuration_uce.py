# Adapted from Universal Cell Embeddings (UCE), https://github.com/snap-stanford/UCE
# Copyright (c) 2023 Yanay Rosen, Yusuf Roohani, Jure Leskovec. MIT License.
from transformers import PretrainedConfig


class UCEConfig(PretrainedConfig):
    """Configuration for UCE.

    Defaults match the released 4-layer UCE model. For the 33-layer model set
    ``nlayers=33``.
    """

    model_type = "uce"
    attribute_map = {
        "hidden_size": "d_model",
        "num_hidden_layers": "nlayers",
        "num_attention_heads": "nhead",
        "intermediate_size": "d_hid",
    }

    def __init__(
        self,
        vocab_size: int = 145469,          # protein-embedding tokens + chromosome tokens
        token_dim: int = 5120,             # ESM2 embedding size
        d_model: int = 1280,
        nhead: int = 20,
        d_hid: int = 5120,
        nlayers: int = 4,
        output_dim: int = 1280,
        dropout: float = 0.05,
        max_position_embeddings: int = 1536,
        pad_length: int = 1536,
        cls_token_idx: int = 3,
        chrom_token_offset: int = 143574,
        chrom_token_left_idx: int = 1,
        chrom_token_right_idx: int = 2,
        pad_token_idx: int = 0,
        sample_size: int = 1024,
        legacy_float_padding_mask: bool = True,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.token_dim = token_dim
        self.d_model = d_model
        self.nhead = nhead
        self.d_hid = d_hid
        self.nlayers = nlayers
        self.output_dim = output_dim
        self.dropout = dropout
        self.max_position_embeddings = max_position_embeddings
        self.pad_length = pad_length
        self.cls_token_idx = cls_token_idx
        self.chrom_token_offset = chrom_token_offset
        self.chrom_token_left_idx = chrom_token_left_idx
        self.chrom_token_right_idx = chrom_token_right_idx
        self.pad_token_idx = pad_token_idx
        self.sample_size = sample_size
        # Original UCE passes a float padding mask (1 - mask), which PyTorch adds
        # to the attention logits instead of masking padded keys. Keep True to
        # reproduce the released checkpoints; set False for true masking.
        self.legacy_float_padding_mask = legacy_float_padding_mask
        super().__init__(pad_token_id=pad_token_idx, **kwargs)
