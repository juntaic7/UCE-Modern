#!/usr/bin/env bash
# Preprocess the raw benchmark datasets into the UCE input format.
#   bash scripts/preprocess.sh
set -euo pipefail

cd "$(dirname "$0")/.."  # repository root
PYTHON=${PYTHON:-.venv/bin/python}
RAW=benchmark/raw_datasets
OUT=benchmark/preprocessed_datasets
mkdir -p "$OUT"

# NeurIPS 2021 BMMC CITE-seq: raw counts in layers["counts"], ADT features dropped
"$PYTHON" tools/preprocess_dataset.py neurips_2021 "$RAW/neurips_2021.h5ad" \
    --out "$OUT/neurips_2021.h5ad"

# NYGC multimodal PBMC (CELLxGENE layout): raw counts in .raw.X, symbols in var["feature_name"]
"$PYTHON" tools/preprocess_dataset.py cellxgene "$RAW/nygc_multimodal_pbmc.h5ad" \
    --out "$OUT/nygc_multimodal_pbmc.h5ad"
