"""Convert an original UCE checkpoint (e.g. 4layer_model.torch) to a
transformers-format model, optionally pushing it to the Hugging Face Hub.

    python scripts/convert_original_checkpoint.py \
        --checkpoint model_files/4layer_model.torch --nlayers 4 \
        --out converted/uce-4l [--push <user>/uce-4l]
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uce_modern import UCEConfig, UCEModel  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--nlayers", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--push", default=None, help="Hub repo id, e.g. user/uce-4l")
    args = ap.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    vocab_size, token_dim = state["pe_embedding.weight"].shape
    config = UCEConfig(nlayers=args.nlayers, vocab_size=vocab_size, token_dim=token_dim)

    model = UCEModel(config)
    model.load_state_dict(state, strict=True)
    model.eval()

    # Ship the modeling code with the weights so AutoModel works via trust_remote_code.
    UCEConfig.register_for_auto_class()
    UCEModel.register_for_auto_class("AutoModel")

    model.save_pretrained(args.out)
    print(f"Saved to {args.out}")
    if args.push:
        model.push_to_hub(args.push)
        print(f"Pushed to https://huggingface.co/{args.push}")


if __name__ == "__main__":
    main()
