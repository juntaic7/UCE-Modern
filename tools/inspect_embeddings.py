"""Summarise .npy embedding files, e.g. the raw outputs of the official UCE implementation.

    python tools/inspect_embeddings.py benchmark/embedding_original
    python tools/inspect_embeddings.py benchmark/embedding_original/neurips_2021_uce_embs.npy \
        --adata-dir benchmark/raw_datasets

For each file, prints shape, dtype and basic statistics (NaN/Inf, all-zero and
duplicate rows, row norms). With --adata-dir, ``<name>_uce_embs.npy`` is matched to
``<name>.h5ad`` and the cell counts are compared.
"""
import argparse
from pathlib import Path

import numpy as np


def summarise(path, chunk=100_000):
    emb = np.load(path, mmap_mode="r")
    print(f"== {path}")
    print(f"  shape: {emb.shape}  dtype: {emb.dtype}  size on disk: {path.stat().st_size / 2**20:.1f} MiB")
    if emb.ndim != 2:
        return emb.shape[0] if emb.ndim else 0

    n, d = emb.shape
    n_nan = n_inf = n_zero = 0
    norms = np.empty(n, dtype=np.float64)
    vmin, vmax, total = np.inf, -np.inf, 0.0
    for start in range(0, n, chunk):
        x = np.asarray(emb[start:start + chunk], dtype=np.float64)
        n_nan += int(np.isnan(x).any(axis=1).sum())
        n_inf += int(np.isinf(x).any(axis=1).sum())
        n_zero += int((x == 0).all(axis=1).sum())
        norms[start:start + len(x)] = np.linalg.norm(x, axis=1)
        vmin, vmax = min(vmin, np.nanmin(x)), max(vmax, np.nanmax(x))
        total += np.nansum(x)

    n_unique = len(np.unique(np.ascontiguousarray(emb).view(np.dtype((np.void, emb.dtype.itemsize * d)))))
    print(f"  cells: {n:,}  embedding dim: {d}")
    print(f"  rows with NaN: {n_nan:,}  with Inf: {n_inf:,}  all-zero: {n_zero:,}  "
          f"duplicate: {n - n_unique:,}")
    print(f"  values: min {vmin:.4g}  max {vmax:.4g}  mean {total / (n * d):.4g}")
    print(f"  row L2 norm: min {np.nanmin(norms):.4g}  median {np.nanmedian(norms):.4g}  "
          f"max {np.nanmax(norms):.4g}")
    print(f"  first row[:5]: {np.asarray(emb[0, :5])}")
    return n


def adata_n_obs(path):
    import anndata as ad
    adata = ad.read_h5ad(path, backed="r")
    n = adata.n_obs
    adata.file.close()
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help=".npy files or folders containing them")
    ap.add_argument("--adata-dir", default=None,
                    help="folder of .h5ad files to compare cell counts against")
    args = ap.parse_args()

    files = []
    for p in map(Path, args.paths):
        files += sorted(p.glob("*.npy")) if p.is_dir() else [p]

    for f in files:
        n = summarise(f)
        if args.adata_dir:
            h5ad = Path(args.adata_dir) / f"{f.stem.removesuffix('_uce_embs')}.h5ad"
            if h5ad.exists():
                n_obs = adata_n_obs(h5ad)
                status = "match" if n == n_obs else f"differ by {n_obs - n:,}"
                print(f"  {h5ad}: {n_obs:,} cells ({status})")
            else:
                print(f"  {h5ad}: not found")
        print()


if __name__ == "__main__":
    main()
