"""Bring an AnnData into the input format ``UCEProcessor.prepare_adata`` expects.

UCE-ready AnnData:

- ``X``: raw integer counts (cells x genes) as a float32 CSR matrix. Not
  normalized, log-transformed or scaled.
- ``var_names``: unique gene symbols (e.g. ``CD3E``), matched case-insensitively
  against the processor's gene vocabulary. Not Ensembl ids.
- ``var["gene_id"]``: the source gene id (e.g. Ensembl), when known. Informational only.
- ``obs``, ``obsm``, ``uns``: passed through unchanged.
"""
from __future__ import annotations

import re
from typing import Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

# Layers checked, in order, for raw counts when adata.X does not hold them.
COUNTS_LAYERS = ("counts", "raw_counts", "umi_counts")

# Checks look at this many stored values: enough to tell counts from normalized data.
_CHECK_VALUES = 1_000_000
_GENE_ID = re.compile(r"^(ENS[A-Z]*G\d{11}(\.\d+)?|\d+)$")  # Ensembl id or numeric index


def _values(X) -> np.ndarray:
    if sparse.issparse(X):
        return np.asarray(X.data[:_CHECK_VALUES])
    return np.asarray(X[: max(1, _CHECK_VALUES // max(1, X.shape[1]))]).ravel()


def is_raw_counts(X) -> bool:
    """True if ``X`` looks like raw counts: non-negative integers (checked on a sample of values)."""
    v = _values(X)
    return bool((v >= 0).all() and np.array_equal(v, np.round(v)))


def _looks_like_gene_ids(names: pd.Index) -> bool:
    sample = names[:1000].astype(str)
    return len(sample) > 0 and np.mean([bool(_GENE_ID.match(n)) for n in sample]) > 0.5


def _source(adata: ad.AnnData, name: str) -> Tuple[object, pd.DataFrame]:
    """Counts matrix and matching var for ``name``: "X", "raw" (adata.raw) or a layer."""
    if name == "X":
        return adata.X, adata.var
    if name == "raw":
        if adata.raw is None:
            raise ValueError("counts_layer='raw' but adata.raw is None")
        # raw may keep more genes than X; bring over the var columns raw.var lacks.
        extra = adata.var[adata.var.columns.difference(adata.raw.var.columns)]
        return adata.raw.X, adata.raw.var.join(extra)
    if name not in adata.layers:
        raise KeyError(f"No layer {name!r}; layers: {list(adata.layers)}")
    return adata.layers[name], adata.var


def _find_counts(adata: ad.AnnData, counts_layer: Optional[str]) -> Tuple[object, pd.DataFrame]:
    if counts_layer is not None:
        return _source(adata, counts_layer)
    names = ["X", *(k for k in COUNTS_LAYERS if k in adata.layers)]
    if adata.raw is not None:
        names.append("raw")
    for name in names:
        X, var = _source(adata, name)
        if is_raw_counts(X):
            if name != "X":
                print(f"UCE: using raw counts from {'adata.raw.X' if name == 'raw' else f'layers[{name!r}]'}")
            return X, var
    raise ValueError(
        f"No raw counts found in {', '.join(names)}: all hold negative or non-integer values. "
        "UCE needs raw counts (e.g. UMI counts from Cell Ranger). Pass counts_layer= to "
        "force a source."
    )


def validate_uce_input(adata: ad.AnnData) -> None:
    """Raise ``ValueError`` if ``adata`` is not in the UCE input format (see module docstring).

    Non-integer counts only print a warning, as in ``UCEProcessor.prepare_adata``:
    some assays (e.g. Smart-seq) report fractional estimated counts.
    """
    if (_values(adata.X) < 0).any():
        raise ValueError("adata.X has negative values; UCE needs raw counts, not scaled data.")
    if not is_raw_counts(adata.X):
        print("UCE: warning: adata.X has non-integer values; UCE expects raw counts. "
              "Normalized or log-transformed data gives wrong embeddings.")
    if not adata.var_names.is_unique:
        raise ValueError("var_names are not unique; call adata.var_names_make_unique().")
    if _looks_like_gene_ids(adata.var_names):
        raise ValueError(
            f"var_names look like gene ids, not symbols (e.g. {adata.var_names[0]!r}). "
            "UCE matches genes by symbol: pass gene_symbol_key=<var column with symbols>."
        )


def to_uce_format(
    adata: ad.AnnData,
    gene_symbol_key: Optional[str] = None,
    gene_id_key: Optional[str] = None,
    counts_layer: Optional[str] = None,
) -> ad.AnnData:
    """Return a UCE-ready AnnData built from ``adata`` (see module docstring).

    Args:
        gene_symbol_key: var column holding gene symbols. Default: ``var_names``
            already are symbols. Genes with no symbol keep their original name.
        gene_id_key: var column holding gene ids (e.g. Ensembl), stored as
            ``var["gene_id"]``. Default: the original ``var_names`` when
            ``gene_symbol_key`` is given.
        counts_layer: where the raw counts are: ``"X"``, ``"raw"`` (``adata.raw.X``)
            or a layer name. Default: the first of X, ``COUNTS_LAYERS``, adata.raw
            that holds non-negative integers.
    """
    X, var = _find_counts(adata, counts_layer)
    var = var.copy()
    if gene_symbol_key is not None:
        if gene_symbol_key not in var:
            raise KeyError(f"var has no column {gene_symbol_key!r}; columns: {list(var.columns)}")
        symbols = var[gene_symbol_key].astype(object)
        missing = symbols.isna() | (symbols.astype(str).str.strip() == "")
        symbols = symbols.where(~missing, var.index.to_series())
        var["gene_id"] = (var[gene_id_key] if gene_id_key else var.index).astype(str)
    else:
        symbols = var.index
        if gene_id_key:
            var["gene_id"] = var[gene_id_key].astype(str)
    var.index = ad.utils.make_index_unique(pd.Index(symbols.astype(str).to_numpy()))

    out = ad.AnnData(
        X=sparse.csr_matrix(X, dtype=np.float32),
        obs=adata.obs.copy(),
        var=var,
        obsm=dict(adata.obsm),
        uns=dict(adata.uns),
    )
    validate_uce_input(out)
    return out
