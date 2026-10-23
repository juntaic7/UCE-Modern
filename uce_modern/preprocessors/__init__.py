"""Dataset preprocessors: turn 10x Genomics or CELLxGENE data into the AnnData
format ``UCEProcessor.prepare_adata`` expects (raw counts in ``X``, gene symbols
as ``var_names``).

    from uce_modern.preprocessors import read_10x, from_cellxgene

    adata = read_10x("sample/outs/filtered_feature_bc_matrix.h5")
    adata = from_cellxgene("cellxgene_download.h5ad")
"""
from .cellxgene import ORGANISM_TO_SPECIES, from_cellxgene, infer_species
from .common import is_raw_counts, to_uce_format, validate_uce_input
from .tenx import read_10x

__all__ = [
    "ORGANISM_TO_SPECIES",
    "from_cellxgene",
    "infer_species",
    "is_raw_counts",
    "read_10x",
    "to_uce_format",
    "validate_uce_input",
]
