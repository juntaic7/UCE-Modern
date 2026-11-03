"""Read 10x Genomics Cell Ranger output into a UCE-ready AnnData."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import anndata as ad
import scanpy as sc

from .common import to_uce_format

PathLike = str | Path


def _read_one(path: Path) -> ad.AnnData:
    """Indexed by gene id, with symbols (not unique) in var["gene_symbol"]."""
    if path.is_dir():
        # A Cell Ranger outs/ folder: use its filtered matrix.
        for name in ("filtered_feature_bc_matrix.h5", "filtered_feature_bc_matrix"):
            if (path / name).exists():
                return _read_one(path / name)
        adata = sc.read_10x_mtx(path, var_names="gene_ids", gex_only=True)
        adata.var["gene_symbol"] = adata.var.pop("gene_symbols")
    else:
        adata = sc.read_10x_h5(path, gex_only=True)
        adata.var["gene_symbol"] = adata.var_names
        adata.var_names = adata.var.pop("gene_ids")
    return adata


def read_10x(paths: PathLike | Sequence[PathLike], sample_names: Optional[Sequence[str]] = None) -> ad.AnnData:
    """Read Cell Ranger output into a UCE-ready AnnData.

    Args:
        paths: one or more of: an ``.h5`` matrix (``filtered_feature_bc_matrix.h5``,
            ``sample_filtered_feature_bc_matrix.h5`` from ``cellranger multi``, ...);
            a matrix folder (``matrix.mtx.gz``, ``features.tsv.gz`` or ``genes.tsv``,
            ``barcodes.tsv.gz``); or a Cell Ranger ``outs/`` folder. Only Gene
            Expression features are kept.
        sample_names: one per path. Several samples (or any ``sample_names``) are
            concatenated on their shared gene ids, with ``obs["sample"]`` set and
            ``_<sample>`` appended to barcodes to keep them unique.
            Default: ``"0"``, ``"1"``, ...
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    adatas = [_read_one(Path(p)) for p in paths]
    if len(adatas) == 1 and sample_names is None:
        adata = adatas[0]
    else:
        names = list(sample_names) if sample_names is not None else [str(i) for i in range(len(adatas))]
        if len(names) != len(adatas):
            raise ValueError(f"Got {len(names)} sample names for {len(adatas)} paths")
        # merge="first": symbols from the first sample, if references renamed a gene.
        adata = ad.concat(dict(zip(names, adatas)), label="sample", index_unique="_",
                          join="inner", merge="first")
    return to_uce_format(adata, gene_symbol_key="gene_symbol", counts_layer="X")
