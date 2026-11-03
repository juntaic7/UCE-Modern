"""Template preprocessor for a dataset with no dedicated preprocessor.

``from_custom`` works as is for any AnnData with raw counts somewhere (``X``, a
layer or ``adata.raw``): pass the arguments that describe your data. For a
dataset you preprocess often, copy this file (e.g. to ``my_dataset.py``), rename
the function, set the defaults below and fill in the ``EDIT:`` step, as
``neurips_2021.py`` does.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import anndata as ad

from .common import to_uce_format

# EDIT: var column holding gene symbols, or None if var_names already are symbols.
GENE_SYMBOL_KEY: Optional[str] = None
# EDIT: var column holding gene ids (e.g. Ensembl), or None.
GENE_ID_KEY: Optional[str] = None
# EDIT: where the raw counts are: "X", "raw" (adata.raw.X), a layer name, or None to auto-detect.
COUNTS_LAYER: Optional[str] = None
# EDIT: (var column, values to keep) to drop non-RNA features (e.g. ("feature_types", ["GEX"])
# for CITE-seq ADT proteins or multiome ATAC peaks), or None to keep all features.
FEATURE_FILTER: Optional[Tuple[str, Sequence[str]]] = None


def from_custom(
    adata: ad.AnnData | str | Path,
    gene_symbol_key: Optional[str] = GENE_SYMBOL_KEY,
    gene_id_key: Optional[str] = GENE_ID_KEY,
    counts_layer: Optional[str] = COUNTS_LAYER,
    feature_filter: Optional[Tuple[str, Sequence[str]]] = FEATURE_FILTER,
) -> ad.AnnData:
    """Return a UCE-ready AnnData from an ``.h5ad`` (path or AnnData).

    ``gene_symbol_key``, ``gene_id_key`` and ``counts_layer`` are passed to
    ``to_uce_format``. ``feature_filter`` is ``(var column, values)``: only
    features whose column value is in ``values`` are kept.
    """
    if not isinstance(adata, ad.AnnData):
        adata = ad.read_h5ad(adata)

    # EDIT: dataset-specific fixes, e.g. dropping cells
    # (adata = adata[adata.obs["qc_pass"]]) or recovering gene symbols from
    # names like "ENSG00000141510_TP53" into a var column for gene_symbol_key.

    if feature_filter is not None:
        key, keep = feature_filter
        if key not in adata.var:
            raise KeyError(f"var has no column {key!r}; columns: {list(adata.var.columns)}")
        adata = adata[:, adata.var[key].isin(list(keep)).to_numpy()]
    return to_uce_format(adata, gene_symbol_key=gene_symbol_key, gene_id_key=gene_id_key,
                         counts_layer=counts_layer)
