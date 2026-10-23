"""Bring a CELLxGENE dataset into the UCE input format.

Handles both ways of getting CELLxGENE data:

- an ``.h5ad`` downloaded from CELLxGENE Discover: ``var_names`` are Ensembl ids,
  symbols are in ``var["feature_name"]``, and ``X`` is usually normalized, with
  raw counts in ``adata.raw.X``;
- an AnnData from the Census API (``cellxgene_census.get_anndata``): ``X`` holds
  raw counts, symbols are in ``var["feature_name"]`` and Ensembl ids in
  ``var["feature_id"]``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import anndata as ad

from .common import to_uce_format

# CELLxGENE organism label -> UCEProcessor species.
ORGANISM_TO_SPECIES = {
    "Homo sapiens": "human",
    "Mus musculus": "mouse",
    "Xenopus tropicalis": "frog",
    "Danio rerio": "zebrafish",
    "Microcebus murinus": "mouse_lemur",
    "Sus scrofa": "pig",
    "Macaca fascicularis": "macaca_fascicularis",
    "Macaca mulatta": "macaca_mulatta",
}


def infer_species(adata: ad.AnnData) -> Optional[str]:
    """UCE species from CELLxGENE organism metadata (``obs`` or ``uns["organism"]``).

    Returns None if the organism is missing, mixed, or not supported by UCE.
    """
    organisms = set()
    if "organism" in adata.obs:
        organisms |= set(adata.obs["organism"].astype(str).unique())
    if "organism" in adata.uns:
        organisms.add(str(adata.uns["organism"]))
    species = {ORGANISM_TO_SPECIES.get(o) for o in organisms}
    return species.pop() if len(species) == 1 else None


def from_cellxgene(adata: ad.AnnData | str | Path, counts_layer: Optional[str] = None) -> ad.AnnData:
    """Return a UCE-ready AnnData from a CELLxGENE Discover ``.h5ad`` (path or AnnData)
    or a Census AnnData.

    ``counts_layer`` is passed to ``to_uce_format``; by default raw counts are taken
    from X if it holds them, else from ``adata.raw.X``.
    """
    if not isinstance(adata, ad.AnnData):
        adata = ad.read_h5ad(adata)
    if "feature_name" not in adata.var:
        raise ValueError(
            "var has no 'feature_name' column; this does not look like CELLxGENE data. "
            "For other AnnData use to_uce_format(adata, gene_symbol_key=...)."
        )
    gene_id_key = "feature_id" if "feature_id" in adata.var else None
    return to_uce_format(adata, gene_symbol_key="feature_name", gene_id_key=gene_id_key,
                         counts_layer=counts_layer)
