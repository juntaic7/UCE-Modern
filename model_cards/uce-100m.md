---
license: mit
library_name: transformers
pipeline_tag: feature-extraction
tags:
  - biology
  - single-cell
  - scRNA-seq
  - cell-embedding
  - foundation-model
  - uce
---

# UCE-Modern-100m (Universal Cell Embeddings, 4-layer)

The 4-layer **Universal Cell Embeddings (UCE)** model, packaged for Hugging Face
`transformers` by [UCE-Modern](https://github.com/juntaic7/UCE-Modern). UCE maps a
single cell's raw gene-expression counts to a 1280-dimensional embedding. It
works across species and needs no fine-tuning.

These are the **original pretrained weights** released by
[snap-stanford/UCE](https://github.com/snap-stanford/UCE) (`4layer_model.torch`).
Only the format has changed. All credit for the model and its training goes to
the original authors.

For a larger model with stronger embeddings, see the 33-layer
[UCE-Modern-650m](https://huggingface.co/{owner}/UCE-Modern-650m).

| | |
| --- | --- |
| Transformer layers | 4 |
| Hidden size / heads / FFN | 1280 / 20 / 5120 |
| Parameters | 104M trainable + 745M frozen token table (ESM2 protein embeddings) |
| Output | `cell_embedding` [batch, 1280], L2-normalized (`X_uce`) |
| Weights | `model.safetensors`, float32, ~3.4 GB |
| Species | human, mouse, frog, zebrafish, mouse lemur, pig, *M. fascicularis*, *M. mulatta* |

## Usage

```bash
pip install torch transformers einops scanpy
```

Dependencies are unpinned, and nothing in the model depends on a particular CUDA
version. Any recent PyTorch build should work: CUDA 13, CUDA 12 or CPU only.
Tested with PyTorch 2.14 (CUDA 13.0) and `transformers` 5.20.

```python
import scanpy as sc
import torch
from transformers import AutoModel
from transformers.dynamic_module_utils import get_class_from_dynamic_module

repo = "{repo_id}"
model = AutoModel.from_pretrained(repo, trust_remote_code=True).eval()

# The processor turns raw counts into UCE "cell sentences". Its code ships in this repo.
UCEProcessor = get_class_from_dynamic_module("processing_uce.UCEProcessor", repo)
processor = UCEProcessor.from_pretrained(repo, species="human")

adata = sc.read_h5ad("my_data.h5ad")      # see "Input data format" below
adata = processor.prepare_adata(adata)    # filter genes/cells, map genes to tokens

embeddings = []
with torch.inference_mode():
    for batch in processor.iter_batches(adata, batch_size=25, seed=0):
        embeddings.append(model(**batch).cell_embedding)
adata.obsm["X_uce"] = torch.cat(embeddings).numpy()
```

If you have the [UCE-Modern](https://github.com/juntaic7/UCE-Modern) package installed,
you can use `from uce_modern import UCEProcessor` instead of `get_class_from_dynamic_module`.

## Input data format

Before `processor.prepare_adata`, the AnnData **must** have:

| Field | Required | Content |
| --- | --- | --- |
| `adata.X` | **Yes** | **Raw integer counts** (e.g. UMI counts), cells × genes, sparse or dense. Not normalized, log-transformed or scaled. |
| `adata.var_names` | **Yes** | **Gene symbols** of the processor's species (e.g. `CD3E`; matched case-insensitively), unique. Not Ensembl IDs. |
| `adata.obs` | No | Any cell metadata. Passed through unchanged. |

Keep all genes; don't subset to highly variable genes. `prepare_adata` filters
genes and cells, drops genes without a protein embedding or chromosome position,
and adds the per-gene columns the tokenizer (`processor(adata)` /
`processor.iter_batches(adata)`) reads. They are set for you; don't create them by hand:

| Column | Type | Content |
| --- | --- | --- |
| `var["uce_token_id"]` | int | Token id of the gene's ESM2 protein embedding |
| `var["uce_chrom"]` | int | Chromosome code |
| `var["uce_start"]` | int | Genomic start position (orders genes within a chromosome) |

Getting common sources into this format:

```python
# 10x Genomics Cell Ranger output: symbols are already var_names, X holds UMI counts.
adata = sc.read_10x_h5("filtered_feature_bc_matrix.h5")
adata.var_names_make_unique()

# CELLxGENE (Discover download or Census AnnData): symbols are in var["feature_name"].
adata = sc.read_h5ad("cellxgene_download.h5ad")
if adata.raw is not None:  # Discover files: X is normalized, raw counts are in .raw
    adata.X = adata.raw.X
adata.var_names = adata.var["feature_name"].astype(str).to_numpy()
adata.var_names_make_unique()
```

The [UCE-Modern](https://github.com/juntaic7/UCE-Modern) package includes
dataset preprocessors (`uce_modern.preprocessors.read_10x`, `from_cellxgene`)
that do this conversion, check the result, and handle multi-sample 10x runs.

## Files

| File | Purpose |
| --- | --- |
| `config.json`, `model.safetensors` | Model config and weights |
| `configuration_uce.py`, `modeling_uce.py` | Model code loaded with `trust_remote_code=True` |
| `processing_uce.py` | `UCEProcessor`: AnnData → `input_ids` / `attention_mask` |
| `gene_vocab.csv` | For each species: gene symbol → token id, chromosome code, genomic start |

`gene_vocab.csv` is built from the original `protein_embeddings/`,
`species_offsets.pkl` and `species_chrom.csv`. You do not need to download
those files (~2.6 GB) to run inference. The ESM2 protein embeddings are already
stored in the model's frozen token table.

## Notes

- The model expects **raw integer counts**. Each cell sentence samples 1024
  genes with replacement, weighted by `log1p(counts)`, so embeddings depend on
  the random seed. Pass `seed=` for reproducible results.
- By default, `legacy_float_padding_mask=True` reproduces the original code's
  padding behavior. The released weights were trained with it.

## Citation

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

MIT, following the original [UCE](https://github.com/snap-stanford/UCE)
(Copyright (c) 2023 Yanay Rosen, Yusuf Roohani, Jure Leskovec).
