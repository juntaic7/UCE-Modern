# UCE-Modern

A modernized reimplementation of **Universal Cell Embeddings (UCE)**.

> Original repository: **[snap-stanford/UCE](https://github.com/snap-stanford/UCE)**
> by Yanay Rosen, Yusuf Roohani, Jure Leskovec, et al. All credit for the model,
> training, and pretrained weights goes to the original authors.

## Why this fork?

The original UCE code pins older dependencies (e.g. `torch==2.1.1`, `numpy==1.26`,
`requests==2.25.1`, `urllib3==1.26`), which can be hard to install alongside a
current ML stack. **UCE-Modern aims to bring UCE up to date with the most recent
PyTorch release** (developed against PyTorch 2.14 / Python 3.12) and current
versions of its other dependencies, while reproducing the original embeddings.

It also packages UCE as a Hugging Face `transformers` model, so it can be loaded
with `AutoModel.from_pretrained(...)` and shared on the Hugging Face Hub.

## Status

🚧 Work in progress.

- [x] `transformers`-compatible model (`UCEConfig`, `UCEModel`) for both the
      4-layer and 33-layer UCE architectures
- [x] Script to convert original UCE checkpoints
- [ ] Numerical parity check against original UCE embeddings (`X_uce`)
- [ ] AnnData preprocessing / cell-sentence processor
- [ ] Pretrained weights published on the Hugging Face Hub

## Installation

```bash
git clone <this-repo-url> UCE-Modern
cd UCE-Modern
pip install -r requirements.txt
```

Dependencies are kept minimal: `torch`, `einops`, `scanpy`, `accelerate`,
`transformers` (their transitive dependencies cover numpy, pandas, scipy,
anndata, huggingface_hub, safetensors, etc.).

## Usage

### Convert an original checkpoint

Download an original checkpoint (e.g. `4layer_model.torch`) from the
[original repo](https://github.com/snap-stanford/UCE), then:

```bash
# 4-layer model
python scripts/convert_original_checkpoint.py \
    --checkpoint model_files/4layer_model.torch --nlayers 4 --out converted/uce-4l

# 33-layer model, and push to the Hub
python scripts/convert_original_checkpoint.py \
    --checkpoint model_files/33l_8ep_1024t_1280.torch --nlayers 33 \
    --out converted/uce-33l --push <hf-user>/uce-33l
```

### Load the model

```python
import uce_modern  # registers UCE with AutoConfig / AutoModel
from transformers import AutoModel

model = AutoModel.from_pretrained("converted/uce-4l").eval()

# From the Hub (modeling code is uploaded alongside the weights):
# model = AutoModel.from_pretrained("<hf-user>/uce-4l", trust_remote_code=True)

out = model(input_ids=input_ids, attention_mask=attention_mask)
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
@article{rosen2023universal,
  title   = {Universal Cell Embeddings: A Foundation Model for Cell Biology},
  author  = {Rosen, Yanay and Roohani, Yusuf and Agrawal, Ayush and Samotorcan, Leon
             and Tabula Sapiens Consortium and Quake, Stephen R. and Leskovec, Jure},
  journal = {bioRxiv},
  year    = {2023},
  doi     = {10.1101/2023.11.28.568918}
}
```

## License

MIT. This project is derived from [UCE](https://github.com/snap-stanford/UCE)
(MIT, Copyright (c) 2023 Yanay Rosen, Yusuf Roohani, Jure Leskovec); see
[LICENSE](LICENSE).
