"""End-to-end example: load the 4- and 33-layer UCE models through
``transformers.AutoModel`` and embed a sample of the 10k PBMC dataset that
ships with the original UCE model files (``model_files/10k_pbmcs_proc.h5ad``).

    python example.py --n-cells 300 --models 4 33

The first run converts each checkpoint in ``model_files/`` into ``converted/``.
"""
import argparse
import gc
from pathlib import Path

import numpy as np
import scanpy as sc
import torch
from transformers import AutoModel

import uce_modern  # noqa: F401  (registers "uce" with AutoConfig/AutoModel)
from uce_modern import UCEModel
from uce_modern.convert import convert_original_checkpoint
from uce_modern.processing_uce import CHECKPOINT_FILES, UCEProcessor, download_model_file


LABEL_KEYS = ("cell_type", "celltype", "cell_types", "str_labels", "labels", "leiden", "louvain")


def load_pbmc(path: str):
    """Load the original UCE example dataset (raw counts, gene symbols as var_names)."""
    if not Path(path).exists():
        raise SystemExit(
            f"{path} not found. It is part of the original UCE model files: "
            "https://figshare.com/articles/dataset/Universal_Cell_Embedding_Model_Files/24320806"
        )
    adata = sc.read_h5ad(path)
    # Prefer raw counts if the file also stores processed values.
    if "counts" in adata.layers:
        adata.X = adata.layers["counts"]
    elif adata.raw is not None and adata.raw.n_vars == adata.n_vars:
        adata.X = adata.raw.X
    adata.var_names_make_unique()
    return adata


def find_label_key(adata, requested=None):
    if requested:
        return requested
    return next((k for k in LABEL_KEYS if k in adata.obs), None)


def load_model(nlayers: int, model_files: str, converted_dir: str) -> UCEModel:
    hf_dir = Path(converted_dir) / f"uce-{nlayers}l"
    if not (hf_dir / "config.json").exists():
        ckpt = download_model_file(CHECKPOINT_FILES[nlayers], model_files)
        print(f"Converting {ckpt} -> {hf_dir}")
        convert_original_checkpoint(ckpt, hf_dir)
    model = AutoModel.from_pretrained(hf_dir)  # tests the transformers loading path

    assert isinstance(model, UCEModel), type(model)
    assert model.config.nlayers == nlayers == len(model.transformer_encoder.layers)
    n_params = sum(p.numel() for n, p in model.named_parameters() if not n.startswith("pe_embedding"))
    print(f"Loaded {nlayers}-layer UCE from {hf_dir} ({n_params / 1e6:.0f}M params excl. token table)")
    return model


@torch.inference_mode()
def embed(model, processor, adata, batch_size, seed, device):
    model.to(device).eval()
    out = []
    for i, batch in enumerate(processor.iter_batches(adata, batch_size=batch_size, seed=seed)):
        batch = {k: v.to(device) for k, v in batch.items()}
        out.append(model(**batch).cell_embedding.float().cpu())
        print(f"  batch {i + 1}/{-(-adata.n_obs // batch_size)}", end="\r")
    print()
    return torch.cat(out).numpy()


def check_embeddings(name, emb, labels, n_cells, dim):
    assert emb.shape == (n_cells, dim), emb.shape
    assert np.isfinite(emb).all(), "non-finite values in embeddings"
    norms = np.linalg.norm(emb, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3), (norms.min(), norms.max())

    # Sanity check: cells of the same type should be more similar than cells of different types.
    sim = emb @ emb.T
    same = labels[:, None] == labels[None, :]
    off_diag = ~np.eye(len(labels), dtype=bool)
    within, between = sim[same & off_diag].mean(), sim[~same].mean()
    print(f"{name}: shape={emb.shape}  mean cos within type={within:.3f}  between types={between:.3f}")
    return sim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=int, nargs="+", default=[4, 33], choices=[4, 33])
    ap.add_argument("--n-cells", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model-files", default="model_files")
    ap.add_argument("--converted-dir", default="converted")
    ap.add_argument("--adata", default="model_files/10k_pbmcs_proc.h5ad",
                    help="AnnData of raw counts with gene symbols as var_names")
    ap.add_argument("--label-key", default=None,
                    help="obs column with cell types for the sanity check (auto-detected if omitted)")
    ap.add_argument("--out", default="outputs/pbmc_uce.h5ad")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # 1. Data: filter the full dataset (as the original does), then sample cells.
    pbmc = load_pbmc(args.adata)
    print(f"Loaded {args.adata}: {pbmc.n_obs} cells x {pbmc.n_vars} genes")
    processor = UCEProcessor.from_model_files(args.model_files, species="human")
    adata = processor.prepare_adata(pbmc, filter=True)
    rng = np.random.default_rng(args.seed)
    idx = np.sort(rng.choice(adata.n_obs, size=min(args.n_cells, adata.n_obs), replace=False))
    adata = adata[idx].copy()
    label_key = find_label_key(adata, args.label_key)
    print(f"Cell-type labels for sanity check: {label_key or 'none found (pass --label-key)'}")
    labels = adata.obs[label_key].astype(str).to_numpy() if label_key else np.zeros(adata.n_obs, dtype=str)
    print(f"Embedding {adata.n_obs} cells x {adata.n_vars} genes on {args.device}")

    # 2. Models: same seed -> both models see identical cell sentences.
    sims = {}
    for nlayers in args.models:
        model = load_model(nlayers, args.model_files, args.converted_dir)
        emb = embed(model, processor, adata, args.batch_size, args.seed, args.device)
        key = f"X_uce_{nlayers}l"
        adata.obsm[key] = emb
        sims[key] = check_embeddings(key, emb, labels, adata.n_obs, model.config.output_dim)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(sims) == 2:
        a, b = (s[np.triu_indices_from(s, k=1)] for s in sims.values())
        print(f"Agreement of cell-cell similarities, 4l vs 33l: Pearson r = {np.corrcoef(a, b)[0, 1]:.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(args.out)
    print(f"Wrote {args.out} (embeddings in .obsm: {', '.join(sims)})")


if __name__ == "__main__":
    main()
