#!/usr/bin/env bash
# Embed the preprocessed benchmark datasets with UCE-Modern-650m (strict fp32), writing
# <name>_uce_embs.npy to match the original UCE outputs in benchmark/embedding_original.
#   bash scripts/embed.sh
set -euo pipefail

cd "$(dirname "$0")/.."  # repository root
PYTHON=${PYTHON:-.venv/bin/python}
IN=benchmark/preprocessed_datasets
OUT=benchmark/embedding

# Batch size 25 and the gene/cell filters as in the original UCE; existing .npy files are skipped.
"$PYTHON" tools/embed_dataset.py "$IN/neurips_2021.h5ad" "$IN/nygc_multimodal_pbmc.h5ad" \
    --model uce-650m --batch-size 25 --out-dir "$OUT"
