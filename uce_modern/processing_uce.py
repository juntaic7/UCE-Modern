# Adapted from Universal Cell Embeddings (UCE), https://github.com/snap-stanford/UCE
# Copyright (c) 2023 Yanay Rosen, Yusuf Roohani, Jure Leskovec. MIT License.
"""Turn an AnnData of raw counts into UCE "cell sentences".

Mirrors the original pipeline (``data_proc/data_utils.py`` + ``eval_data.py``):

1. optional filtering (genes in >= 10 cells, cells with >= 25 genes);
2. keep genes that have an ESM2 protein embedding and a chromosome position;
3. per cell, sample ``sample_size`` genes with replacement, p ~ log1p(counts);
4. group sampled genes by chromosome (chromosome order shuffled), sort by genomic start, and emit ``[CLS] (CHROM_k genes... CHROM_END)* [PAD]...``.
"""
from __future__ import annotations

import json
import pickle
import tarfile
import urllib.request
from pathlib import Path
from typing import Dict, Iterator, Optional, Sequence

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy import sparse

FIGSHARE = "https://figshare.com/ndownloader/files/"
MODEL_FILE_URLS = {
    "species_chrom.csv": FIGSHARE + "42706558",
    "species_offsets.pkl": FIGSHARE + "42706555",
    "protein_embeddings.tar.gz": FIGSHARE + "42715213",
    "4layer_model.torch": FIGSHARE + "42706576",
    "33l_8ep_1024t_1280.torch": FIGSHARE + "43423236",
}

# Released checkpoints by model name: UCE-100M is the 4-layer model, UCE-650M the
# 33-layer one. Filenames as on figshare (and as the original UCE code expects).
CHECKPOINT_FILES = {"uce-100m": "4layer_model.torch", "uce-650m": "33l_8ep_1024t_1280.torch"}

# Compact processor data shipped in each Hub repo instead of the ~2.6 GB protein
# embeddings: per species, gene symbol -> token id, chromosome code, genomic start.
GENE_VOCAB_FILE = "gene_vocab.csv"

# Processor settings read from the model's config.json in ``UCEProcessor.from_pretrained``.
_CONFIG_KEYS = ("sample_size", "pad_length", "pad_token_idx", "chrom_token_right_idx",
                "cls_token_idx", "chrom_token_offset")

SPECIES_TO_PE_FILE = {
    "human": "Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt",
    "mouse": "Mus_musculus.GRCm39.gene_symbol_to_embedding_ESM2.pt",
    "frog": "Xenopus_tropicalis.Xenopus_tropicalis_v9.1.gene_symbol_to_embedding_ESM2.pt",
    "zebrafish": "Danio_rerio.GRCz11.gene_symbol_to_embedding_ESM2.pt",
    "mouse_lemur": "Microcebus_murinus.Mmur_3.0.gene_symbol_to_embedding_ESM2.pt",
    "pig": "Sus_scrofa.Sscrofa11.1.gene_symbol_to_embedding_ESM2.pt",
    "macaca_fascicularis": "Macaca_fascicularis.Macaca_fascicularis_6.0.gene_symbol_to_embedding_ESM2.pt",
    "macaca_mulatta": "Macaca_mulatta.Mmul_10.gene_symbol_to_embedding_ESM2.pt",
}


FIGSHARE_ARTICLE = "https://figshare.com/articles/dataset/Universal_Cell_Embedding_Model_Files/24320806"


def _is_valid(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _fetch(url: str, dest: Path) -> None:
    """Download ``url`` to ``dest`` atomically; refuse empty responses."""
    tmp = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (UCE-Modern)"})
    try:
        with urllib.request.urlopen(req) as resp, open(tmp, "wb") as f:
            if resp.status != 200:
                raise RuntimeError(f"HTTP {resp.status}")
            while chunk := resp.read(1 << 20):
                f.write(chunk)
        if tmp.stat().st_size == 0:
            raise RuntimeError("empty response (figshare may be blocking scripted downloads)")
        tmp.replace(dest)
    except Exception as e:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download {dest.name} from {url}: {e}\n"
            f"Download it manually from {FIGSHARE_ARTICLE} and save it as {dest}"
        ) from e


def download_model_file(name: str, model_files_dir: str | Path = "model_files") -> Path:
    """Download one of MODEL_FILE_URLS into ``model_files_dir`` (no-op if present).

    ``protein_embeddings.tar.gz`` is extracted to ``model_files_dir/protein_embeddings``.
    Empty files left by a failed download are treated as missing.
    """
    root = Path(model_files_dir)
    root.mkdir(parents=True, exist_ok=True)
    if name == "protein_embeddings.tar.gz":
        target = root / "protein_embeddings"
        if target.is_dir() and any(target.iterdir()):
            return target
        archive = root / name
        if not _is_valid(archive):
            print(f"Downloading {name} ...")
            _fetch(MODEL_FILE_URLS[name], archive)
        with tarfile.open(archive) as tar:
            tar.extractall(root, filter="data")
        return target
    path = root / name
    if not _is_valid(path):
        print(f"Downloading {name} -> {path} ...")
        _fetch(MODEL_FILE_URLS[name], path)
    return path


def _load_pt(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:  # older pickles that weights_only cannot read
        return torch.load(path, map_location="cpu", weights_only=False)


def build_gene_vocab(
    model_files_dir: str | Path = "model_files",
    species: Optional[Sequence[str]] = None,
    download: bool = True,
) -> pd.DataFrame:
    """Build the gene vocabulary from the original UCE model files.

    Returns one row per (species, gene) that has both an ESM2 protein embedding
    and a chromosome position, with columns ``species, gene_symbol, token_id,
    chrom_code, start``. ``species=None`` builds all species in SPECIES_TO_PE_FILE.
    """
    root = Path(model_files_dir)
    if download:
        for name in ("species_chrom.csv", "species_offsets.pkl", "protein_embeddings.tar.gz"):
            download_model_file(name, root)
    with open(root / "species_offsets.pkl", "rb") as f:
        offsets = pickle.load(f)

    # Chromosome codes are categories over ALL species, as in the original.
    chrom = pd.read_csv(root / "species_chrom.csv")
    chrom["spec_chrom"] = pd.Categorical(chrom["species"] + "_" + chrom["chromosome"])
    chrom["chrom_code"] = chrom["spec_chrom"].cat.codes.astype(np.int64)
    chrom["gene_symbol"] = chrom["gene_symbol"].astype(str).str.upper()

    frames = []
    for sp in species or SPECIES_TO_PE_FILE:
        # Token ids: position in the species' protein-embedding dict + species offset.
        pe = _load_pt(root / "protein_embeddings" / SPECIES_TO_PE_FILE[sp])
        gene_to_token: Dict[str, int] = {}
        for i, gene in enumerate(pe.keys()):
            gene_to_token.setdefault(gene.upper(), i + offsets[sp])
        del pe
        sp_chrom = chrom[chrom["species"] == sp].drop_duplicates("gene_symbol")
        sp_chrom = sp_chrom[sp_chrom["gene_symbol"].isin(gene_to_token)]
        frames.append(pd.DataFrame({
            "species": sp,
            "gene_symbol": sp_chrom["gene_symbol"].to_numpy(),
            "token_id": sp_chrom["gene_symbol"].map(gene_to_token).to_numpy(np.int64),
            "chrom_code": sp_chrom["chrom_code"].to_numpy(),
            "start": sp_chrom["start"].to_numpy(),
        }))
    return pd.concat(frames, ignore_index=True)


class UCEProcessor:
    """Builds UCE input_ids / attention_mask from an AnnData of raw counts."""

    def __init__(
        self,
        gene_to_token: Dict[str, int],
        gene_to_chrom: pd.DataFrame,
        species: str = "human",
        sample_size: int = 1024,
        pad_length: int = 1536,
        pad_token_idx: int = 0,
        chrom_token_right_idx: int = 2,
        cls_token_idx: int = 3,
        chrom_token_offset: int = 143574,
    ):
        self.gene_to_token = gene_to_token        # UPPER symbol -> global token id
        self.gene_to_chrom = gene_to_chrom        # index: UPPER symbol; cols: chrom_code, start
        self.species = species
        self.sample_size = sample_size
        self.pad_length = pad_length
        self.pad_token_idx = pad_token_idx
        self.chrom_token_right_idx = chrom_token_right_idx
        self.cls_token_idx = cls_token_idx
        self.chrom_token_offset = chrom_token_offset

    @classmethod
    def from_gene_vocab(cls, vocab: pd.DataFrame, species: str = "human", **kwargs) -> "UCEProcessor":
        """Build a processor from a ``build_gene_vocab`` table (or its saved CSV)."""
        vocab = vocab[vocab["species"] == species]
        if vocab.empty:
            raise ValueError(f"No genes for species {species!r}; available: {sorted(SPECIES_TO_PE_FILE)}")
        gene_to_token = dict(zip(vocab["gene_symbol"], vocab["token_id"].astype(int)))
        chrom = vocab.set_index("gene_symbol")[["chrom_code", "start"]]
        return cls(gene_to_token, chrom, species=species, **kwargs)

    @classmethod
    def from_model_files(
        cls,
        model_files_dir: str | Path = "model_files",
        species: str = "human",
        download: bool = True,
        **kwargs,
    ) -> "UCEProcessor":
        """Build a processor from the original UCE model files (downloaded from figshare)."""
        vocab = build_gene_vocab(model_files_dir, species=[species], download=download)
        return cls.from_gene_vocab(vocab, species=species, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        species: str = "human",
        revision: Optional[str] = None,
        **kwargs,
    ) -> "UCEProcessor":
        """Load from a converted model directory or Hub repo (e.g. ``<hf-user>/uce-100m``).

        Reads ``gene_vocab.csv`` and the sampling settings in ``config.json``;
        ``kwargs`` override the config values.
        """
        root = Path(pretrained_model_name_or_path)
        if root.is_dir():
            vocab_path, config_path = root / GENE_VOCAB_FILE, root / "config.json"
        else:
            from huggingface_hub import hf_hub_download

            repo = str(pretrained_model_name_or_path)
            vocab_path = hf_hub_download(repo, GENE_VOCAB_FILE, revision=revision)
            config_path = hf_hub_download(repo, "config.json", revision=revision)
        with open(config_path) as f:
            config = json.load(f)
        kwargs = {**{k: config[k] for k in _CONFIG_KEYS if k in config}, **kwargs}
        # keep_default_na=False: gene symbols such as "NA" must stay strings.
        vocab = pd.read_csv(vocab_path, dtype={"species": str, "gene_symbol": str}, keep_default_na=False)
        return cls.from_gene_vocab(vocab, species=species, **kwargs)

    def prepare_adata(self, adata: sc.AnnData, filter: bool = True) -> sc.AnnData:
        """Filter cells/genes and attach per-gene token, chromosome, and start.

        ``adata.X`` must hold raw counts and ``var_names`` must be gene symbols.
        """
        adata = adata.copy()
        if filter:
            sc.pp.filter_genes(adata, min_cells=10)
            sc.pp.filter_cells(adata, min_genes=25)
        upper = adata.var_names.str.upper()
        keep = upper.isin(list(self.gene_to_token)) & upper.isin(self.gene_to_chrom.index)
        n_drop = int((~keep).sum())
        if n_drop:
            print(f"UCE: dropping {n_drop}/{adata.n_vars} genes without protein embedding or chromosome position")
        adata = adata[:, keep].copy()

        # Sampling weights are log1p(int(counts)), so a cell needs at least one
        # gene with count >= 1 among the kept genes. Cells can lose all their
        # counts here when the input has few genes (e.g. an HVG subset).
        X = adata.X
        if sparse.issparse(X):
            usable = np.asarray((X >= 1).sum(axis=1)).ravel() > 0
            data = X.data
        else:
            usable = (np.asarray(X) >= 1).any(axis=1)
            data = np.asarray(X).ravel()
        if data.size and not np.allclose(data, np.round(data)):
            print("UCE: warning: adata.X has non-integer values; UCE expects raw counts "
                  "(values are truncated to int before sampling).")
        n_empty = int((~usable).sum())
        if n_empty:
            print(f"UCE: dropping {n_empty}/{adata.n_obs} cells with no counts over UCE genes")
            adata = adata[usable].copy()
        upper = adata.var_names.str.upper()
        adata.var["uce_token_id"] = [self.gene_to_token[g] for g in upper]
        adata.var["uce_chrom"] = self.gene_to_chrom.loc[upper, "chrom_code"].to_numpy()
        adata.var["uce_start"] = self.gene_to_chrom.loc[upper, "start"].to_numpy()
        return adata

    def _sentence(self, counts: np.ndarray, tokens, chroms, starts, rng) -> np.ndarray:
        weights = np.log1p(counts.astype(np.int64))  # original casts counts to int64
        total = weights.sum()
        if total <= 0:
            raise ValueError("Cell has no counts over UCE genes; filter it out first.")
        choice = rng.choice(len(weights), size=self.sample_size, p=weights / total, replace=True)
        choice = choice[np.argsort(chroms[choice])]
        new_chrom, chosen_starts = chroms[choice], starts[choice]

        uq_chroms = np.unique(new_chrom)
        rng.shuffle(uq_chroms)
        seq = [self.cls_token_idx]
        for c in uq_chroms:
            loc = np.where(new_chrom == c)[0]
            to_add = choice[loc[np.argsort(chosen_starts[loc])]]
            seq.append(int(c) + self.chrom_token_offset)
            seq.extend(tokens[to_add].tolist())
            seq.append(self.chrom_token_right_idx)
        if len(seq) > self.pad_length:
            raise ValueError(f"Cell sentence length {len(seq)} exceeds pad_length {self.pad_length}.")
        return np.asarray(seq, dtype=np.int64)

    def __call__(self, adata: sc.AnnData, seed: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """Encode all cells of a prepared AnnData into one padded batch."""
        rng = np.random.default_rng(seed)
        tokens = adata.var["uce_token_id"].to_numpy(np.int64)
        chroms = adata.var["uce_chrom"].to_numpy(np.int64)
        starts = adata.var["uce_start"].to_numpy()
        X = adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X)
        seqs = [self._sentence(row, tokens, chroms, starts, rng) for row in X]

        max_len = max(len(s) for s in seqs)
        input_ids = torch.full((len(seqs), max_len), self.pad_token_idx, dtype=torch.long)
        attention_mask = torch.zeros((len(seqs), max_len), dtype=torch.float32)
        for i, s in enumerate(seqs):
            input_ids[i, : len(s)] = torch.from_numpy(s)
            attention_mask[i, : len(s)] = 1.0
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def iter_batches(
        self, adata: sc.AnnData, batch_size: int = 25, seed: Optional[int] = 0
    ) -> Iterator[Dict[str, torch.Tensor]]:
        rng = np.random.default_rng(seed)
        for start in range(0, adata.n_obs, batch_size):
            yield self(adata[start : start + batch_size], seed=int(rng.integers(2**32)))
