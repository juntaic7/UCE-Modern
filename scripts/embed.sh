#!/usr/bin/env bash
# Embed the preprocessed benchmark datasets with UCE-Modern-650m (TF32), writing
# <name>_uce_embs_seed<SEED>.npy, to compare with original UCE: benchmark/embedding_original.
#   bash scripts/embed.sh            # seed 0
#   SEED=1 bash scripts/embed.sh     # another gene sampling, for the seed-vs-seed baseline
#   LOG=true bash scripts/embed.sh   # append output to <OUT>/<dataset>_<model>_embd_log.txt, not the terminal
set -euo pipefail

cd "$(dirname "$0")/.."  # repository root
PYTHON=${PYTHON:-.venv/bin/python}
SEED=${SEED:-0}
LOG=${LOG:-false}
if [[ "$LOG" != true && "$LOG" != false ]]; then
    echo "LOG must be true or false, got '$LOG'" >&2
    exit 1
fi
IN=benchmark/preprocessed_datasets
OUT=benchmark/embedding
DATASET=neurips_2021
MODEL=uce-650m

# Batch size 25 and the gene/cell filters as in the original UCE; existing .npy files are skipped.
cmd=("$PYTHON" -u tools/embed_dataset.py "$IN/$DATASET.h5ad"
     --model "$MODEL" --precision tf32 --batch-size 25 --seed "$SEED" --out-dir "$OUT")

if [[ "$LOG" == true ]]; then
    mkdir -p "$OUT"
    log="$OUT/${DATASET}_${MODEL}_embd_log.txt"
    echo "Logging to $log"
    echo "=== $(date '+%Y-%m-%d %H:%M:%S')  seed=$SEED  ${cmd[*]}" >> "$log"
    "${cmd[@]}" >> "$log" 2>&1
else
    "${cmd[@]}"
fi
