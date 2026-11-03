"""Convert 10x Genomics, CELLxGENE, NeurIPS 2021 or other data into the AnnData format UCE expects.

    python tools/preprocess_dataset.py 10x sample_a/outs sample_b/outs \
        --sample-names a b --out data/pbmc.h5ad
    python tools/preprocess_dataset.py cellxgene cellxgene_download.h5ad --out data/lung.h5ad
    python tools/preprocess_dataset.py neurips_2021 benchmark/raw_datasets/neurips_2021.h5ad \
        --out benchmark/preprocessed_datasets/neurips_2021.h5ad
    python tools/preprocess_dataset.py custom other.h5ad --gene-symbol-key gene_symbols \
        --counts-layer counts --out data/other.h5ad

The output has raw counts in .X and gene symbols as var_names. Pass it to
``UCEProcessor.prepare_adata`` (or ``python example.py --adata <out>``).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uce_modern.preprocessors import (  # noqa: E402
    from_cellxgene, from_custom, from_neurips_2021, infer_species, read_10x,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", choices=["10x", "cellxgene", "neurips_2021", "custom"])
    ap.add_argument("inputs", nargs="+",
                    help="10x: Cell Ranger .h5 files, matrix folders or outs/ folders; "
                         "cellxgene: one .h5ad from CELLxGENE Discover or Census; "
                         "neurips_2021, custom: one .h5ad")
    ap.add_argument("--out", required=True, help="Output .h5ad")
    ap.add_argument("--sample-names", nargs="+", default=None, help="10x only: one name per input")
    ap.add_argument("--counts-layer", default=None,
                    help="not 10x: 'X', 'raw' or a layer name (default: auto-detect; "
                         "neurips_2021: 'counts')")
    ap.add_argument("--gene-symbol-key", default=None,
                    help="custom only: var column holding gene symbols (default: var_names)")
    ap.add_argument("--gene-id-key", default=None,
                    help="custom only: var column holding gene ids, e.g. Ensembl")
    args = ap.parse_args()

    if args.source != "custom" and (args.gene_symbol_key or args.gene_id_key):
        ap.error("--gene-symbol-key and --gene-id-key apply to custom only")
    if args.source == "10x":
        if args.counts_layer:
            ap.error("--counts-layer does not apply to 10x")
        adata = read_10x(args.inputs, sample_names=args.sample_names)
    else:
        if len(args.inputs) != 1 or args.sample_names:
            ap.error(f"{args.source} takes one .h5ad and no --sample-names")
        if args.source == "cellxgene":
            adata = from_cellxgene(args.inputs[0], counts_layer=args.counts_layer)
        elif args.source == "neurips_2021":
            adata = from_neurips_2021(args.inputs[0], counts_layer=args.counts_layer or "counts")
        else:
            adata = from_custom(args.inputs[0], gene_symbol_key=args.gene_symbol_key,
                                gene_id_key=args.gene_id_key, counts_layer=args.counts_layer)

    adata.obs.columns = adata.obs.columns.str.replace("/", "_")  # h5ad keys cannot contain '/'
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out)
    print(f"Wrote {out}: {adata.n_obs} cells x {adata.n_vars} genes "
          "(raw counts in .X, gene symbols as var_names)")
    if species := infer_species(adata):
        print(f"Organism metadata says species={species!r}: use UCEProcessor.from_pretrained(..., species={species!r})")


if __name__ == "__main__":
    main()
