"""Bring the NeurIPS 2021 multimodal single-cell benchmark into the UCE input format.

The CITE-seq bone marrow dataset from Open Problems (``uns["dataset_id"]`` is
``openproblems_bmmc_cite``, 90,261 cells), as distributed for the UCE benchmarks:

- ``X`` is normalized; raw counts are in ``layers["counts"]``;
- ``var_names`` are gene symbols (not unique), Ensembl ids are in ``var["gene_id"]``;
- ``var["feature_types"]`` is ``GEX`` for genes and ``ADT`` for the 134 surface
  proteins, whose names (``CD4``, ``CD19``, ...) clash with gene symbols.
"""
from __future__ import annotations

from pathlib import Path

import anndata as ad

from .common import to_uce_format


def from_neurips_2021(adata: ad.AnnData | str | Path, counts_layer: str = "counts") -> ad.AnnData:
    """Return a UCE-ready AnnData from the NeurIPS 2021 ``.h5ad`` (path or AnnData).

    Keeps gene expression (``GEX``) features only, dropping the ADT proteins.
    ``obs`` (``cell_type``, ``batch``, donor metadata, ...) and ``obsm`` (including
    the ADT embeddings) are kept.
    """
    if not isinstance(adata, ad.AnnData):
        adata = ad.read_h5ad(adata)
    if "feature_types" in adata.var:
        adata = adata[:, (adata.var["feature_types"] == "GEX").to_numpy()]
    return to_uce_format(adata, gene_id_key="gene_id", counts_layer=counts_layer)
