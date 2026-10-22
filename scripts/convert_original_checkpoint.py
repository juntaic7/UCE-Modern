"""Convert an original UCE checkpoint (4- or 33-layer) to transformers format.

    python scripts/convert_original_checkpoint.py \
        --checkpoint model_files/4layer_model.torch --out converted/uce-4l \
        [--push <hf-user>/uce-4l]

The number of layers and hidden sizes are inferred from the checkpoint.
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
    ap.add_argument("--push", default=None, help="Hub repo id, e.g. user/uce-4l")
    args = ap.parse_args()
    model = convert_original_checkpoint(args.checkpoint, args.out, args.push)
    print(f"Saved {model.config.nlayers}-layer UCE to {args.out}")
    if args.push:
        print(f"Pushed to https://huggingface.co/{args.push}")


if __name__ == "__main__":
    main()
