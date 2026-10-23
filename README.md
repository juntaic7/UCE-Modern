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
preserved, and work toward reproducing its embeddings is ongoing.

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
- [x] Dataset preprocessors for 10x Genomics and CELLxGENE data (`uce_modern/preprocessors/`)
- [x] End-to-end example on 10x PBMCs (`example.py`)
- [ ] Numerical parity check against original UCE embeddings (`X_uce`)
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

### Prepare 10x Genomics or CELLxGENE data

The dataset preprocessors in [`uce_modern/preprocessors/`](uce_modern/preprocessors)
convert common sources into the input format above, then check the result.
They raise an error if only normalized values are found, or if `var_names`
are still gene ids.

| Source | Input | What the preprocessor does |
| --- | --- | --- |
| 10x Genomics Cell Ranger | `filtered_feature_bc_matrix.h5`, a matrix folder (`matrix.mtx.gz`, `features.tsv.gz`, `barcodes.tsv.gz`) or an `outs/` folder | Keeps Gene Expression features only; symbols as `var_names`, Ensembl IDs in `var["gene_id"]`; several samples are concatenated, with `obs["sample"]` |
| CELLxGENE Discover | Downloaded `.h5ad` | Takes raw counts from `adata.raw.X` (its `X` is normalized); symbols from `var["feature_name"]` |
| CELLxGENE Census | AnnData from `cellxgene_census.get_anndata(...)` | Symbols from `var["feature_name"]`, Ensembl IDs from `var["feature_id"]` |

```python
from uce_modern.preprocessors import from_cellxgene, infer_species, read_10x, to_uce_format

adata = read_10x("sample/outs/filtered_feature_bc_matrix.h5")
adata = read_10x(["a/outs", "b/outs"], sample_names=["a", "b"])  # sets obs["sample"]
adata = from_cellxgene("cellxgene_download.h5ad")  # or a Census AnnData
species = infer_species(adata)  # "human", "mouse", ... from CELLxGENE organism metadata

# Any other AnnData: name the var column holding gene symbols and, optionally,
# where the raw counts are ("X", "raw" for adata.raw.X, or a layer name).
adata = to_uce_format(other, gene_symbol_key="gene_symbols", counts_layer="counts")
```

Or from the command line, writing a ready-to-use `.h5ad`:

```bash
python scripts/preprocess_dataset.py 10x a/outs b/outs --sample-names a b --out data/my_10x.h5ad
python scripts/preprocess_dataset.py cellxgene cellxgene_download.h5ad --out data/my_cxg.h5ad
```

Pass the result to `processor.prepare_adata` (see [Load the model](#load-the-model)), with the
`species` that matches your data.

### Build a Hugging Face repo per model

[UCE-Modern-100m](https://huggingface.co/jtcao7/UCE-Modern-100m) and
[UCE-Modern-650m](https://huggingface.co/jtcao7/UCE-Modern-650m) are published as
separate Hub repos, each self-contained and loadable on its own. To rebuild them,
or to publish them under your own account:

```bash
python scripts/build_hub_repo.py --model uce-100m --push <hf-user>/UCE-Modern-100m
python scripts/build_hub_repo.py --model uce-650m --push <hf-user>/UCE-Modern-650m
```

Each build downloads what it needs from the
[UCE model files on figshare](https://figshare.com/articles/dataset/Universal_Cell_Embedding_Model_Files/24320806)
and writes `converted/<model>/`:

| File | Size | Purpose |
| --- | --- | --- |
| `model.safetensors` | 3.4 GB / 5.7 GB | Weights, including the frozen ESM2 token table |
| `config.json` | — | `UCEConfig` with `auto_map` for `trust_remote_code` |
| `configuration_uce.py`, `modeling_uce.py` | — | Model code |
| `processing_uce.py` | — | `UCEProcessor` |
| `gene_vocab.csv` | 5 MB | Per-species gene → token id, chromosome, start |
| `README.md` | — | Model card from `model_cards/<model>.md` |
| `LICENSE` | — | MIT |

Omit `--push` to build locally only. To convert an arbitrary checkpoint
(weights and code only), use `scripts/convert_original_checkpoint.py`.

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
