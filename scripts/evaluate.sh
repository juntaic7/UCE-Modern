#!/usr/bin/env bash
# Compare every UCE-Modern embedding in benchmark/embedding (<name>_uce_embs_seed<seed>.npy) with the
# original UCE output in benchmark/embedding_original, and each seed other than 0 with seed 0 (the noise
# from gene sampling alone). Reports go to benchmark/evaluation/<name>_seed<seed>_vs_<original|seed0>.json.
#   bash scripts/evaluate.sh
set -euo pipefail
shopt -s nullglob

cd "$(dirname "$0")/.."  # repository root
PYTHON=${PYTHON:-.venv/bin/python}
REF=benchmark/embedding_original
NEW=benchmark/embedding
OUT=benchmark/evaluation

for f in "$NEW"/*_uce_embs_seed*.npy; do
    stem=$(basename "$f" .npy)
    name=${stem%_uce_embs_seed*}
    seed=${stem##*_uce_embs_seed}

    if [[ -f "$REF/${name}_uce_embs.npy" ]]; then
        "$PYTHON" benchmark/evaluation/compare_embeddings.py "$name" \
            --ref-dir "$REF" --new-dir "$NEW" --new-seed "$seed" --out-dir "$OUT"
    else
        echo "$name: no original UCE embedding in $REF, skipping seed$seed vs original"
    fi

    if [[ "$seed" != 0 && -f "$NEW/${name}_uce_embs_seed0.npy" ]]; then
        "$PYTHON" benchmark/evaluation/compare_embeddings.py "$name" \
            --ref-dir "$NEW" --ref-seed 0 --new-dir "$NEW" --new-seed "$seed" --out-dir "$OUT"
    fi
done
