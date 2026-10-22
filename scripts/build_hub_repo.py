"""Build (and optionally push) the Hugging Face repo for one released UCE model.

    python scripts/build_hub_repo.py --model uce-100m --out converted/uce-100m \
        [--push <hf-user>/uce-100m]

uce-100m is the 4-layer model, uce-650m the 33-layer one. Each repo is
self-contained: weights, config, modeling and processing code, gene vocab,
model card, and license. Original files are downloaded into --model-files.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uce_modern.convert import build_hub_repo  # noqa: E402
from uce_modern.processing_uce import CHECKPOINT_FILES  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(CHECKPOINT_FILES))
    ap.add_argument("--out", default=None, help="Output folder (default: converted/<model>)")
    ap.add_argument("--model-files", default="model_files")
    ap.add_argument("--push", default=None, help="Hub repo id, e.g. user/uce-100m")
    ap.add_argument("--private", action="store_true", help="Create the Hub repo as private")
    args = ap.parse_args()

    out = Path(args.out or f"converted/{args.model}")
    model = build_hub_repo(args.model, out, args.model_files, repo_id=args.push)
    print(f"Built {args.model} ({model.config.nlayers} layers) in {out}:")
    for f in sorted(out.iterdir()):
        print(f"  {f.name:24s} {f.stat().st_size / 1e6:10.1f} MB")

    if args.push:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.push, private=args.private, exist_ok=True)
        api.upload_folder(folder_path=out, repo_id=args.push, commit_message=f"Upload {args.model}")
        print(f"Pushed to https://huggingface.co/{args.push}")


if __name__ == "__main__":
    main()
