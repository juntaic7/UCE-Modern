"""Convert original UCE checkpoints to transformers format."""
import re
import shutil
from pathlib import Path
from typing import Optional

import torch

from .configuration_uce import UCEConfig
from .modeling_uce import UCEModel
from . import processing_uce
from .processing_uce import CHECKPOINT_FILES, GENE_VOCAB_FILE, build_gene_vocab, download_model_file

_REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_CARDS_DIR = _REPO_ROOT / "model_cards"

# Hub repo name for each released model; the model cards link to each other by these.
HUB_REPO_NAMES = {"uce-100m": "UCE-Modern-100m", "uce-650m": "UCE-Modern-650m"}

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


def build_hub_repo(
    name: str,
    out_dir: str | Path,
    model_files_dir: str | Path = "model_files",
    repo_id: Optional[str] = None,
    download: bool = True,
) -> UCEModel:
    """Assemble a self-contained Hub repo folder for ``name`` (a CHECKPOINT_FILES key).

    Writes the converted weights, config and modeling code, the processor's
    ``gene_vocab.csv`` and ``processing_uce.py``, plus ``model_cards/<name>.md``
    as README.md (with ``{repo_id}`` and ``{owner}`` filled in) and the LICENSE.
    ``repo_id`` must be ``<owner>/<HUB_REPO_NAMES[name]>``.
    """
    if name not in CHECKPOINT_FILES:
        raise ValueError(f"Unknown model {name!r}; choose from {sorted(CHECKPOINT_FILES)}")
    expected = HUB_REPO_NAMES[name]
    if repo_id is not None and repo_id.split("/")[-1].lower() != expected.lower():
        raise ValueError(f"{name} must be pushed to a repo named {expected!r}, got {repo_id!r}")
    out = Path(out_dir)
    root = Path(model_files_dir)
    ckpt = download_model_file(CHECKPOINT_FILES[name], root) if download else root / CHECKPOINT_FILES[name]
    model = convert_original_checkpoint(ckpt, out)

    build_gene_vocab(root, download=download).to_csv(out / GENE_VOCAB_FILE, index=False)
    shutil.copy(processing_uce.__file__, out / "processing_uce.py")
    shutil.copy(_REPO_ROOT / "LICENSE", out / "LICENSE")
    repo_id = repo_id or f"<hf-user>/{expected}"
    card = (MODEL_CARDS_DIR / f"{name}.md").read_text()
    card = card.replace("{repo_id}", repo_id).replace("{owner}", repo_id.split("/")[0])
    (out / "README.md").write_text(card)  # written last: marks a complete build
    return model
