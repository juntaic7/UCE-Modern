"""Convert original UCE checkpoints to transformers format."""
import re
from pathlib import Path
from typing import Optional

import torch

from .configuration_uce import UCEConfig
from .modeling_uce import UCEModel

_LAYER_KEY = re.compile(r"^transformer_encoder\.layers\.(\d+)\.")


def infer_nlayers(state_dict) -> int:
    return 1 + max(int(m.group(1)) for k in state_dict if (m := _LAYER_KEY.match(k)))


def convert_original_checkpoint(
    checkpoint: str | Path, out_dir: Optional[str | Path] = None, push_to_hub: Optional[str] = None
) -> UCEModel:
    """Load an original UCE ``.torch`` state_dict into ``UCEModel``.

    Architecture sizes are read from the checkpoint, so the same call works for
    the 4-layer and 33-layer models.
    """
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    vocab_size, token_dim = state["pe_embedding.weight"].shape
    d_model = state["encoder.0.weight"].shape[0]
    config = UCEConfig(
        nlayers=infer_nlayers(state),
        vocab_size=vocab_size,
        token_dim=token_dim,
        d_model=d_model,
        d_hid=state["transformer_encoder.layers.0.linear1.weight"].shape[0],
        output_dim=state["decoder.3.weight"].shape[0],
    )
    model = UCEModel(config)
    model.load_state_dict(state, strict=True)
    model.eval()

    # Ship the modeling code with the weights so AutoModel works with trust_remote_code.
    UCEConfig.register_for_auto_class()
    UCEModel.register_for_auto_class("AutoModel")
    if out_dir is not None:
        model.save_pretrained(out_dir)
    if push_to_hub:
        model.push_to_hub(push_to_hub)
    return model
