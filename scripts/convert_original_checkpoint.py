"""Convert an original UCE checkpoint (4- or 33-layer) to transformers format.

    python scripts/convert_original_checkpoint.py \
        --checkpoint model_files/4layer_model.torch --out converted/my-uce \
        [--push <hf-user>/my-uce]

The number of layers and hidden sizes are inferred from the checkpoint. Only the
weights, config and modeling code are written; to build the full Hub repo for a
released model (with processor files and model card) use build_hub_repo.py.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uce_modern.convert import convert_original_checkpoint  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--push", default=None, help="Hub repo id, e.g. user/my-uce")
    args = ap.parse_args()
    model = convert_original_checkpoint(args.checkpoint, args.out, args.push)
    print(f"Saved {model.config.nlayers}-layer UCE to {args.out}")
    if args.push:
        print(f"Pushed to https://huggingface.co/{args.push}")


if __name__ == "__main__":
    main()
