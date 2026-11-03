# UCE-Modern

A modernized reimplementation of **Universal Cell Embeddings (UCE)**.

> Original repository: **[snap-stanford/UCE](https://github.com/snap-stanford/UCE)**
> by Yanay Rosen, Yusuf Roohani, Jure Leskovec, et al. All credit for the model,
> training, and pretrained weights goes to the original authors.

## Why this reimplementation?

The original UCE code pins exact, now-dated dependencies (e.g. `torch==2.1.1`,
`numpy==1.26`, `requests==2.25.1`, `urllib3==1.26`). That makes it hard to install
alongside a current ML stack or on newer hardware.

**UCE-Modern aims for loose dependency and environment requirements.** It pins
no package versions and makes no assumptions about the CUDA toolkit, so UCE can
run on whichever recent PyTorch build suits your machine: for example a CUDA 13
build for newer NVIDIA GPUs, a CUDA 12 build, or CPU only. The original model is
preserved, and its embeddings are reproduced up to UCE's random gene sampling (so far
tested on one dataset; see [Benchmark](#benchmark)).

It is developed and tested with Python 3.12, PyTorch 2.14 (CUDA 13.0) and
`transformers` 5.20. Older versions may work but have not been tested yet.

It also packages UCE as a Hugging Face `transformers` model, so it can be loaded
with `AutoModel.from_pretrained(...)` and shared on the Hugging Face Hub.

## Status

🚧 Work in progress.

- [x] `transformers`-compatible model (`UCEConfig`, `UCEModel`) for both
      released models: **UCE-Modern-100m** (4 layers) and **UCE-Modern-650m** (33 layers)
- [x] Script to convert original UCE checkpoints and build one self-contained
      Hugging Face repo per model
- [x] AnnData preprocessing / cell-sentence processor (`UCEProcessor`)
- [x] Dataset preprocessors for 10x Genomics, CELLxGENE and NeurIPS 2021 data, plus a
      customizable template for other datasets (`uce_modern/preprocessors/`)
- [x] End-to-end example on 10x PBMCs (`example.py`)
- [x] Statistical parity check against original UCE embeddings: UCE-Modern-650m
      matches the original as closely as two seeds match each other
      (see [Benchmark](#benchmark); one dataset so far)
- [x] Pretrained weights published on the Hugging Face Hub:
      [UCE-Modern-100m](https://huggingface.co/jtcao7/UCE-Modern-100m) and
      [UCE-Modern-650m](https://huggingface.co/jtcao7/UCE-Modern-650m)

## Installation

We recommend using **uv** to manage the Python environment and all packages.
[Install uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

```bash
git clone <this-repo-url>
cd UCE-Modern
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

This installs the default PyTorch build from PyPI. To choose the build for your
CUDA version (or CPU), pass uv's `--torch-backend` option:

```bash
uv pip install -r requirements.txt --torch-backend=cu130   # CUDA 13.0
uv pip install -r requirements.txt --torch-backend=cu128   # CUDA 12.8
uv pip install -r requirements.txt --torch-backend=cpu     # CPU only
uv pip install -r requirements.txt --torch-backend=auto    # detect from the installed driver
```

Keep this environment activated when running the examples below. Use
`uv pip install <package>` to install additional packages in the same environment.

Dependencies are kept minimal and unpinned: `torch`, `einops`, `scanpy`,
`accelerate`, `transformers` (their transitive dependencies cover numpy, pandas,
scipy, anndata, huggingface_hub, safetensors, etc.).

## Usage

### Quick start: embed PBMCs

```bash
python example.py --n-cells 300 --models uce-100m uce-650m
```

This downloads the original UCE model files from figshare into `model_files/`
(several GB), builds a Hub repo folder for each model in `converted/<model>`,
loads them with `AutoModel`, embeds a sample of the original UCE 10k PBMC dataset
(`model_files/10k_pbmcs_proc.h5ad`; use `--adata` for your own raw-count AnnData),
runs sanity checks, and writes `outputs/pbmc_uce.h5ad` with embeddings in
`.obsm["X_uce_100m"]` / `["X_uce_650m"]`.

### Input data format

UCE takes an AnnData of single-cell RNA-seq counts. Before you pass it to the
processor (`processor.prepare_adata`), it **must** have:

| Field | Required | Content |
| --- | --- | --- |
| `adata.X` | **Yes** | **Raw integer counts** (e.g. UMI counts), cells × genes, sparse or dense. Not normalized, log-transformed or scaled. |
| `adata.var_names` | **Yes** | **Gene symbols** of the processor's species (e.g. `CD3E`; matched case-insensitively), unique. Not Ensembl IDs. |
| `adata.var["gene_symbol"]` | No | Each gene's symbol, matched instead of `var_names` when present. The dataset preprocessors set it, so genes that share a symbol (unique `var_names` `MATR3`, `MATR3-1`) are all kept, as separate columns, as in the original UCE. |
| `adata.obs` | No | Any cell metadata. Passed through unchanged. |

Keep all genes; don't subset to highly variable genes. `prepare_adata` drops
genes that have no ESM2 protein embedding or chromosome position.

`prepare_adata` then filters genes (expressed in ≥ 10 cells) and cells (≥ 25
genes), and adds the per-gene columns that the tokenizer (`processor(adata)` /
`processor.iter_batches(adata)`) reads. They are set for you; don't create them by hand:

| Column | Type | Content |
| --- | --- | --- |
| `var["uce_token_id"]` | int | Token id of the gene's ESM2 protein embedding |
| `var["uce_chrom"]` | int | Chromosome code |
| `var["uce_start"]` | int | Genomic start position (orders genes within a chromosome) |

### Prepare 10x Genomics, CELLxGENE or other data

The dataset preprocessors in [`uce_modern/preprocessors/`](uce_modern/preprocessors)
convert common sources into the input format above, then check the result.
They raise an error if only normalized values are found, or if `var_names`
are still gene ids.

| Source | Input | What the preprocessor does |
| --- | --- | --- |
| 10x Genomics Cell Ranger | `filtered_feature_bc_matrix.h5`, a matrix folder (`matrix.mtx.gz`, `features.tsv.gz`, `barcodes.tsv.gz`) or an `outs/` folder | Keeps Gene Expression features only; symbols as `var_names` and `var["gene_symbol"]`, Ensembl IDs in `var["gene_id"]`; several samples are concatenated on Ensembl IDs, with `obs["sample"]` |
| CELLxGENE Discover | Downloaded `.h5ad` | Takes raw counts from `adata.raw.X` (its `X` is normalized); symbols from `var["feature_name"]` |
| CELLxGENE Census | AnnData from `cellxgene_census.get_anndata(...)` | Symbols from `var["feature_name"]`, Ensembl IDs from `var["feature_id"]` |
| NeurIPS 2021 (Open Problems BMMC CITE-seq) | `neurips_2021.h5ad` | Takes raw counts from `layers["counts"]` (its `X` is normalized); drops the ADT protein features; Ensembl IDs from `var["gene_id"]` |
| Any other `.h5ad` | `.h5ad` or AnnData | `from_custom`: you name the symbol and id columns, the counts source and which features to keep. To reuse settings, copy [`custom.py`](uce_modern/preprocessors/custom.py) and edit its `EDIT:` lines |

```python
from uce_modern.preprocessors import (
    from_cellxgene, from_custom, from_neurips_2021, infer_species, read_10x, to_uce_format,
)

adata = read_10x("sample/outs/filtered_feature_bc_matrix.h5")
adata = read_10x(["a/outs", "b/outs"], sample_names=["a", "b"])  # sets obs["sample"]
adata = from_cellxgene("cellxgene_download.h5ad")  # or a Census AnnData
species = infer_species(adata)  # "human", "mouse", ... from CELLxGENE organism metadata
adata = from_neurips_2021("benchmark/raw_datasets/neurips_2021.h5ad")

# Any other .h5ad: name the var column holding gene symbols, where the raw counts
# are ("X", "raw" for adata.raw.X, or a layer name) and, optionally, which features to keep.
adata = from_custom("other.h5ad", gene_symbol_key="gene_symbols", counts_layer="counts",
                    feature_filter=("feature_types", ["Gene Expression"]))

# Or the underlying conversion on an in-memory AnnData.
adata = to_uce_format(other, gene_symbol_key="gene_symbols", counts_layer="counts")
```

Or from the command line, writing a ready-to-use `.h5ad`:

```bash
python tools/preprocess_dataset.py 10x a/outs b/outs --sample-names a b --out data/my_10x.h5ad
python tools/preprocess_dataset.py cellxgene cellxgene_download.h5ad --out data/my_cxg.h5ad
python tools/preprocess_dataset.py neurips_2021 benchmark/raw_datasets/neurips_2021.h5ad \
    --out benchmark/preprocessed_datasets/neurips_2021.h5ad
python tools/preprocess_dataset.py custom other.h5ad --gene-symbol-key gene_symbols \
    --counts-layer counts --out data/other.h5ad
```

Pass the result to `processor.prepare_adata` (see [Load the model](#load-the-model)), with the
`species` that matches your data.

### Load the model

```python
import uce_modern  # registers UCE with AutoConfig / AutoModel
from transformers import AutoModel

model = AutoModel.from_pretrained("converted/uce-100m").eval()

# From the Hub (modeling code is uploaded alongside the weights):
# model = AutoModel.from_pretrained("jtcao7/UCE-Modern-100m", trust_remote_code=True)

from uce_modern import UCEProcessor

# Local folder or Hub repo (e.g. "jtcao7/UCE-Modern-100m"); needs only
# gene_vocab.csv + config.json, not the protein embeddings.
# UCEProcessor.from_model_files("model_files") also works.
processor = UCEProcessor.from_pretrained("converted/uce-100m", species="human")
adata = processor.prepare_adata(adata)   # see "Input data format" above
batch = processor(adata[:16], seed=0)    # {"input_ids", "attention_mask"}

out = model(**batch)
out.cell_embedding      # [batch, 1280]  L2-normalized cell embedding (X_uce)
out.last_hidden_state   # [batch, seq_len, 1280]  per-token outputs
```

`input_ids` are UCE "cell sentences" (CLS token first, then chromosome and gene
tokens), shaped `[batch, seq_len]`, with `attention_mask` set to 1 for real
tokens and 0 for padding.

## Differences from the original

- Batch-first inputs and a `transformers`-style output object; parameter names
  are unchanged, so original state dicts load with `strict=True`.
- **Padding mask:** the original code passes a *float* padding mask (`1 - mask`),
  which PyTorch adds to the attention logits rather than masking padded keys.
  This behavior is kept by default (`UCEConfig(legacy_float_padding_mask=True)`)
  to reproduce the released weights. Set it to `False` for standard masking
  (embeddings will then differ from the original).

## Benchmark

UCE builds each cell's input by randomly sampling 1,024 genes, weighted by expression.
The original code does this with an unseeded `np.random`, so its embeddings change
from run to run and cannot be reproduced exactly. UCE-Modern samples with a seed
(`--seed`, default 0), so its embeddings are reproducible. The benchmark below
therefore asks a statistical question: is UCE-Modern as close to the original as
two UCE-Modern runs with different seeds are to each other?

We embedded the NeurIPS 2021 BMMC dataset (90,261 cells) with UCE-Modern-650m
(TF32, batch size 25) using seeds 0 and 1, and compared both with the embeddings
from the original UCE implementation:

| Metric | Seed 0 vs original | Seed 1 vs original | Seed 1 vs seed 0 |
| --- | --- | --- | --- |
| Cosine similarity, same cell (mean) | 0.927 | 0.926 | 0.927 |
| Mean absolute difference per value | 0.0081 | 0.0081 | 0.0081 |
| Own cell is the nearest of all 90k (5,000 sampled cells) | 60.3% | 60.6% | 61.0% |
| Shared 15 nearest neighbours | 12.4% | 12.4% | 12.3% |
| Linear CKA | 0.978 | 0.978 | 0.978 |
| Correlation of per-dimension means | 0.999998 | 0.999998 | 0.999998 |

All three comparisons agree to three or four significant digits. UCE-Modern differs
from the original exactly as much as it differs from itself under a different seed,
so the remaining difference comes entirely from gene sampling, not from the port or
from TF32. For reference, two unrelated cells have a mean cosine similarity of 0.40.

> **Note:** this has so far been tested on one dataset (NeurIPS 2021 BMMC) and one
> model (UCE-Modern-650m). Other tissues, species and UCE-Modern-100m have not been
> benchmarked yet.

The table also shows that a single UCE embedding carries noticeable sampling noise:
only about 12% of a cell's 15 nearest neighbours stay the same after reseeding.
Analyses that rely on fine-grained neighbourhoods may benefit from averaging the
embeddings of several seeds.

To reproduce, put the original UCE outputs in `benchmark/embedding_original/` as
`<dataset>_uce_embs.npy`, then run:

```bash
bash scripts/preprocess.sh          # raw .h5ad -> UCE input format
bash scripts/embed.sh               # seed 0; SEED=1 bash scripts/embed.sh for seed 1
bash scripts/evaluate.sh            # JSON reports in benchmark/evaluation/
```

## Citation

If you use this code, please cite the original UCE paper:

```bibtex
@article{rosen2026universal,
  title={Universal cell embedding provides a foundation model for cell biology},
  author={Rosen, Yanay and Roohani, Yusuf and Agrawal, Ayush and Samotor{\v{c}}an, Leon and Tabula Sapiens Consortium and Quake, Stephen R and Leskovec, Jure},
  journal={Nature},
  pages={1--9},
  year={2026},
  publisher={Nature Publishing Group UK, London}
}
```

## License

MIT. This project is derived from [UCE](https://github.com/snap-stanford/UCE)
(MIT, Copyright (c) 2023 Yanay Rosen, Yusuf Roohani, Jure Leskovec); see
[LICENSE](LICENSE).
