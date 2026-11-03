"""Embed preprocessed .h5ad files with UCE-Modern and save the cell embeddings as .npy.

    python tools/embed_dataset.py benchmark/preprocessed_datasets/neurips_2021.h5ad \
        --model uce-650m --out-dir benchmark/embedding
    python tools/embed_dataset.py data/*.h5ad --model jtcao7/UCE-Modern-650m --out-dir outputs

Inputs must be in the UCE input format (raw counts in .X, gene symbols), e.g. the
output of tools/preprocess_dataset.py. Each ``<name>.h5ad`` gives
``<out-dir>/<name>_uce_embs.npy``: float32, [cells, 1280], L2-normalised, one row per
cell in input order, as the original UCE writes it. If ``prepare_adata`` drops cells
(no counts over UCE genes, or fewer than 25 genes), their obs_names are listed in
``<name>_uce_dropped_cells.txt`` and the rows follow the remaining cells.

--precision: fp32 (default) is strict fp32, for numerical comparison with the original
UCE; tf32 runs the fp32 matmuls on tensor cores; bf16 and fp16 autocast the forward
pass (weights stay fp32). Each is faster than the one before and further from fp32.
"""
import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import anndata as ad  # noqa: E402
from transformers import AutoModel  # noqa: E402

import uce_modern  # noqa: E402,F401  (registers "uce" with AutoConfig/AutoModel)
from uce_modern.convert import build_hub_repo  # noqa: E402
from uce_modern.preprocessors import infer_species  # noqa: E402
from uce_modern.processing_uce import CHECKPOINT_FILES, UCEProcessor  # noqa: E402

PRECISIONS = ("fp32", "tf32", "bf16", "fp16")
AUTOCAST_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


class CellSentences(Dataset):
    """One item per batch of cells; seeds follow ``UCEProcessor.iter_batches``, so the
    sentences (and embeddings) do not depend on the number of workers."""

    def __init__(self, processor, adata, batch_size, seed):
        self.processor, self.adata, self.batch_size = processor, adata, batch_size
        rng = np.random.default_rng(seed)
        self.seeds = [int(rng.integers(2**32)) for _ in range(0, adata.n_obs, batch_size)]

    def __len__(self):
        return len(self.seeds)

    def __getitem__(self, i):
        start = i * self.batch_size
        return self.processor(self.adata[start:start + self.batch_size], seed=self.seeds[i])


def resolve_model(model, model_files, converted_dir):
    """A released model name (built in ``converted/<name>`` if missing), a local folder or a Hub repo."""
    if model in CHECKPOINT_FILES:
        hf_dir = Path(converted_dir) / model
        if not (hf_dir / "README.md").exists():  # written last by build_hub_repo
            print(f"Building {model} -> {hf_dir}")
            build_hub_repo(model, hf_dir, model_files)
        return str(hf_dir)
    return model


@torch.inference_mode()
def embed(model, processor, adata, batch_size, seed, device, num_workers, precision="fp32"):
    autocast = torch.autocast(torch.device(device).type, dtype=AUTOCAST_DTYPES.get(precision),
                              enabled=precision in AUTOCAST_DTYPES)
    loader = DataLoader(
        CellSentences(processor, adata, batch_size, seed), batch_size=None, num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
        # fork shares the AnnData with the workers instead of pickling it to each one.
        multiprocessing_context="fork" if num_workers else None,
    )
    out = np.empty((adata.n_obs, model.config.output_dim), dtype=np.float32)
    n_done, t0 = 0, time.time()
    for i, batch in enumerate(loader):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with autocast:
            emb = model(**batch).cell_embedding.float().cpu().numpy()
        out[n_done:n_done + len(emb)] = emb
        n_done += len(emb)
        if (i + 1) % 50 == 0 or n_done == adata.n_obs:
            rate = n_done / (time.time() - t0)
            print(f"  {n_done:,}/{adata.n_obs:,} cells  {rate:.0f} cells/s  "
                  f"ETA {(adata.n_obs - n_done) / rate / 60:.1f} min", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="preprocessed .h5ad files")
    ap.add_argument("--out-dir", required=True, help="folder for <name>_uce_embs.npy")
    ap.add_argument("--model", default="uce-100m",
                    help=f"{' or '.join(CHECKPOINT_FILES)}, a converted model folder or a Hub repo "
                         "(default: uce-100m)")
    ap.add_argument("--species", default=None,
                    help="UCE species (default: from organism metadata, else 'human')")
    ap.add_argument("--batch-size", type=int, default=25, help="cells per batch (original UCE: 25)")
    ap.add_argument("--seed", type=int, default=0, help="seed for sampling the cell sentences")
    ap.add_argument("--precision", default="fp32", choices=PRECISIONS,
                    help="fp32 (strict, default), tf32, bf16 or fp16 (autocast); see above")
    ap.add_argument("--no-filter", action="store_true",
                    help="skip prepare_adata's gene (>= 10 cells) and cell (>= 25 genes) filters")
    ap.add_argument("--num-workers", type=int, default=8, help="processes building cell sentences")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--model-files", default="model_files")
    ap.add_argument("--converted-dir", default="converted")
    ap.add_argument("--overwrite", action="store_true", help="re-embed inputs whose .npy exists")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    jobs = [(Path(p), out_dir / f"{Path(p).stem}_uce_embs.npy") for p in args.inputs]
    if not args.overwrite:
        for inp, out in jobs:
            if out.exists():
                print(f"Skipping {inp}: {out} exists (pass --overwrite to redo)")
        jobs = [(inp, out) for inp, out in jobs if not out.exists()]
    if not jobs:
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    # TF32 only when asked: plain fp32 must not silently use TF32 matmuls.
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = args.precision == "tf32"
    model_path = resolve_model(args.model, args.model_files, args.converted_dir)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(args.device).eval()
    print(f"Loaded {model.config.nlayers}-layer UCE from {model_path} on {args.device} ({args.precision})")

    for inp, out in jobs:
        adata = ad.read_h5ad(inp)
        species = args.species or infer_species(adata) or "human"
        print(f"{inp}: {adata.n_obs:,} cells x {adata.n_vars:,} genes, species={species}")
        processor = UCEProcessor.from_pretrained(model_path, species=species)
        prepared = processor.prepare_adata(adata, filter=not args.no_filter)
        dropped = adata.obs_names.difference(prepared.obs_names, sort=False)
        del adata
        gc.collect()

        emb = embed(model, processor, prepared, args.batch_size, args.seed, args.device, args.num_workers,
                    args.precision)
        np.save(out, emb)
        print(f"Wrote {out}: {emb.shape}")
        if len(dropped):
            dropped_path = out.with_name(f"{inp.stem}_uce_dropped_cells.txt")
            dropped_path.write_text("\n".join(dropped) + "\n")
            print(f"  {len(dropped):,} cells dropped by prepare_adata; obs_names in {dropped_path}")
        del prepared
        gc.collect()


if __name__ == "__main__":
    main()
