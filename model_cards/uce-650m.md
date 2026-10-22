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

# UCE-Modern-650m (Universal Cell Embeddings, 33-layer)

The 33-layer **Universal Cell Embeddings (UCE)** model, packaged for Hugging Face
`transformers` by [UCE-Modern](https://github.com/juntaic7/UCE-Modern). UCE maps a
single cell's raw gene-expression counts to a 1280-dimensional embedding. It
works across species and needs no fine-tuning.

These are the **original pretrained weights** released by
[snap-stanford/UCE](https://github.com/snap-stanford/UCE) (`33l_8ep_1024t_1280.torch`).
Only the format has changed. All credit for the model and its training goes to
the original authors.

This is the model the UCE paper uses for its main results. For a smaller,
faster model, see the 4-layer [UCE-Modern-100m](https://huggingface.co/{owner}/UCE-Modern-100m).

| | |
| --- | --- |
| Transformer layers | 33 |
| Hidden size / heads / FFN | 1280 / 20 / 5120 |
| Parameters | 675M trainable + 745M frozen token table (ESM2 protein embeddings) |
| Output | `cell_embedding` [batch, 1280], L2-normalized (`X_uce`) |
| Weights | `model.safetensors`, float32, ~5.7 GB |
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

adata = sc.read_h5ad("my_data.h5ad")      # raw counts in .X, gene symbols as var_names
adata = processor.prepare_adata(adata)    # filter genes/cells, map genes to tokens

embeddings = []
with torch.inference_mode():
    for batch in processor.iter_batches(adata, batch_size=25, seed=0):
        embeddings.append(model(**batch).cell_embedding)
adata.obsm["X_uce"] = torch.cat(embeddings).numpy()
```

If you have the [UCE-Modern](https://github.com/juntaic7/UCE-Modern) package installed,
you can use `from uce_modern import UCEProcessor` instead of `get_class_from_dynamic_module`.

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
