"""Dataset preprocessors: turn 10x Genomics, CELLxGENE, NeurIPS 2021 or other data
into the AnnData format ``UCEProcessor.prepare_adata`` expects (raw counts in
``X``, gene symbols as ``var_names``).

    from uce_modern.preprocessors import read_10x, from_cellxgene, from_neurips_2021, from_custom

    adata = read_10x("sample/outs/filtered_feature_bc_matrix.h5")
    adata = from_cellxgene("cellxgene_download.h5ad")
    adata = from_neurips_2021("benchmark/raw_datasets/neurips_2021.h5ad")
    adata = from_custom("other.h5ad", gene_symbol_key="gene_symbols", counts_layer="counts")
"""
from .cellxgene import ORGANISM_TO_SPECIES, from_cellxgene, infer_species
from .common import is_raw_counts, to_uce_format, validate_uce_input
from .custom import from_custom
from .neurips_2021 import from_neurips_2021
from .tenx import read_10x

__all__ = [
    "ORGANISM_TO_SPECIES",
    "from_cellxgene",
    "from_custom",
    "from_neurips_2021",
    "infer_species",
    "is_raw_counts",
    "read_10x",
    "to_uce_format",
    "validate_uce_input",
]
