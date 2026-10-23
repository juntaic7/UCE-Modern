"""Convert 10x Genomics or CELLxGENE data into the AnnData format UCE expects.

    python scripts/preprocess_dataset.py 10x sample_a/outs sample_b/outs \
        --sample-names a b --out data/pbmc.h5ad
    python scripts/preprocess_dataset.py cellxgene cellxgene_download.h5ad --out data/lung.h5ad

The output has raw counts in .X and gene symbols as var_names. Pass it to
``UCEProcessor.prepare_adata`` (or ``python example.py --adata <out>``).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uce_modern.preprocessors import from_cellxgene, infer_species, read_10x  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", choices=["10x", "cellxgene"])
    ap.add_argument("inputs", nargs="+",
                    help="10x: Cell Ranger .h5 files, matrix folders or outs/ folders; "
                         "cellxgene: one .h5ad from CELLxGENE Discover or Census")
    ap.add_argument("--out", required=True, help="Output .h5ad")
    ap.add_argument("--sample-names", nargs="+", default=None, help="10x only: one name per input")
    ap.add_argument("--counts-layer", default=None,
                    help="cellxgene only: 'X', 'raw' or a layer name (default: auto-detect)")
    args = ap.parse_args()

    if args.source == "10x":
        if args.counts_layer:
            ap.error("--counts-layer applies to cellxgene only")
        adata = read_10x(args.inputs, sample_names=args.sample_names)
    else:
        if len(args.inputs) != 1 or args.sample_names:
            ap.error("cellxgene takes one .h5ad and no --sample-names")
        adata = from_cellxgene(args.inputs[0], counts_layer=args.counts_layer)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out)
    print(f"Wrote {out}: {adata.n_obs} cells x {adata.n_vars} genes "
          "(raw counts in .X, gene symbols as var_names)")
    if species := infer_species(adata):
        print(f"Organism metadata says species={species!r}: use UCEProcessor.from_pretrained(..., species={species!r})")


if __name__ == "__main__":
    main()
