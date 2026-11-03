"""Compare UCE-Modern cell embeddings with those of the original UCE, or with another seed.

    # UCE-Modern seed 0 vs original UCE -> neurips_2021_seed0_vs_original.json
    python benchmark/evaluation/compare_embeddings.py neurips_2021
    # UCE-Modern seed 1 vs seed 0 (the noise from gene sampling alone) -> neurips_2021_seed1_vs_seed0.json
    python benchmark/evaluation/compare_embeddings.py neurips_2021 \
        --ref-dir benchmark/embedding --ref-seed 0 --new-seed 1

Loads the reference ``<ref-dir>/<name>_uce_embs.npy`` (original UCE; with --ref-seed,
``<name>_uce_embs_seed<ref-seed>.npy``) and ``<new-dir>/<name>_uce_embs_seed<new-seed>.npy``
(UCE-Modern), whose rows are the same cells in the same order, and writes
``<out-dir>/<name>_seed<new-seed>_vs_<original|seed<ref-seed>>.json``.

UCE samples each cell's genes at random, and the original UCE does so with an unseeded
``np.random``, so its embeddings cannot be reproduced exactly, not even by rerunning it; the
comparison is statistical. Next to element-wise and per-cell differences it reports a baseline
(cosine similarity between unrelated cells), how well each new embedding retrieves its own
reference one among all cells, and how well cell neighbourhoods and the global geometry
(linear CKA) agree. Comparing two UCE-Modern seeds gives the level that sampling alone explains.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PERCENTILES = (1, 5, 25, 50, 75, 95, 99)


def rnd(v, digits=6):
    return float(f"{v:.{digits}g}")


def describe(x):
    """Distribution summary of a 1-D array, rounded for reading."""
    x = np.asarray(x, dtype=np.float64).ravel()
    out = {"mean": x.mean(), "std": x.std(), "min": x.min()}
    for p, v in zip(PERCENTILES, np.percentile(x, PERCENTILES)):
        out["median" if p == 50 else f"p{p}"] = v
    out["max"] = x.max()
    return {k: rnd(v) for k, v in out.items()}


def rel(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def load(path, seed):
    emb = np.load(path)
    if emb.ndim != 2:
        raise ValueError(f"{path}: expected a 2-D array, got shape {emb.shape}")
    info = {
        "path": rel(path),
        "seed": seed,  # null: unseeded gene sampling (original UCE)
        "shape": list(emb.shape),
        "dtype": str(emb.dtype),
        "rows_with_nan": int(np.isnan(emb).any(axis=1).sum()),
        "rows_with_inf": int(np.isinf(emb).any(axis=1).sum()),
        "row_l2_norm": describe(np.linalg.norm(emb.astype(np.float64), axis=1)),
    }
    return emb.astype(np.float32), info


def unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def top_k(sims, rows, k):
    """Indices of the k most similar cells per row, excluding the cell itself."""
    sims[np.arange(len(rows)), rows] = -np.inf
    return np.argpartition(sims, -k, axis=1)[:, -k:]


def compare(ref, new, n_query, k, seed):
    n, d = ref.shape
    rng = np.random.default_rng(seed)
    ref_u, new_u = unit(ref), unit(new)

    # Per cell: the same cell's reference and new embedding.
    cos = (ref_u * new_u).sum(axis=1)
    l2 = np.linalg.norm(new - ref, axis=1)

    # Baseline: each cell against a different, randomly paired cell (a derangement).
    order = rng.permutation(n)
    partner = np.empty(n, dtype=np.int64)
    partner[order] = np.roll(order, 1)
    random_ref = (ref_u * ref_u[partner]).sum(axis=1)
    random_cross = (new_u * ref_u[partner]).sum(axis=1)

    diff = new - ref
    abs_diff = np.abs(diff)
    flat_ref, flat_new = ref.ravel().astype(np.float64), new.ravel().astype(np.float64)
    pearson_all = np.corrcoef(flat_ref, flat_new)[0, 1]
    del flat_ref, flat_new

    # Per dimension, across cells.
    ref_c = ref.astype(np.float64) - ref.mean(axis=0, dtype=np.float64)
    new_c = new.astype(np.float64) - new.mean(axis=0, dtype=np.float64)
    dim_corr = (ref_c * new_c).sum(axis=0) / np.sqrt((ref_c ** 2).sum(axis=0) * (new_c ** 2).sum(axis=0))
    dim_mean_corr = np.corrcoef(ref.mean(axis=0), new.mean(axis=0))[0, 1]
    dim_std_corr = np.corrcoef(ref.std(axis=0), new.std(axis=0))[0, 1]

    # Linear CKA (Kornblith et al., 2019) on all cells.
    cross = new_c.T @ ref_c
    cka = (cross ** 2).sum() / (np.linalg.norm(ref_c.T @ ref_c) * np.linalg.norm(new_c.T @ new_c))
    del ref_c, new_c, cross

    # Retrieval and neighbourhoods for a random sample of query cells, against all cells.
    queries = np.sort(rng.choice(n, size=min(n_query, n), replace=False))
    ranks, overlap = [], []
    for start in range(0, len(queries), 1000):
        q = queries[start:start + 1000]
        sims = new_u[q] @ ref_u.T
        ranks.append((sims > sims[np.arange(len(q)), q][:, None]).sum(axis=1))
        nn_ref = top_k(ref_u[q] @ ref_u.T, q, k)
        nn_new = top_k(new_u[q] @ new_u.T, q, k)
        overlap += [len(np.intersect1d(a, b, assume_unique=True)) / k for a, b in zip(nn_ref, nn_new)]
    ranks = np.concatenate(ranks)
    overlap = np.asarray(overlap)

    summary = {
        "n_cells": n,
        "embedding_dim": d,
        "cosine_similarity_same_cell_mean": rnd(cos.mean()),
        "cosine_similarity_same_cell_median": rnd(np.median(cos)),
        "cosine_similarity_random_pair_mean": rnd(random_cross.mean()),
        "retrieval_top1_accuracy": rnd((ranks == 0).mean()),
        f"knn{k}_overlap_mean": rnd(overlap.mean()),
        "linear_cka": rnd(cka),
        "pearson_all_values": rnd(pearson_all),
    }
    sections = {
        "per_cell_cosine_similarity": {
            "description": "Cosine similarity between each cell's reference and new embedding (1 = same direction).",
            **describe(cos),
            "fraction_at_least": {str(t): rnd((cos >= t).mean()) for t in (0.999, 0.99, 0.95, 0.9, 0.8)},
        },
        "random_pair_baseline": {
            "description": "Cosine similarity between unrelated cells, the level that 'same cell' should clearly exceed. "
                           "reference_vs_reference: two different cells, both reference; "
                           "new_vs_reference: a cell's new embedding vs another cell's reference.",
            "reference_vs_reference": describe(random_ref),
            "new_vs_reference": describe(random_cross),
            "same_cell_mean_minus_random_pair_mean": rnd(cos.mean() - random_cross.mean()),
        },
        "per_cell_l2_distance": {
            "description": "Euclidean distance between each cell's reference and new embedding "
                           "(both are L2-normalised, so 0 = identical and 2 = opposite).",
            **describe(l2),
        },
        "elementwise_difference": {
            "description": "Over all cells x dimensions: |new - reference|, the relative Frobenius error "
                           "||new - reference|| / ||reference||, and the Pearson correlation of all values.",
            "abs_diff": describe(abs_diff),
            "relative_frobenius_error": rnd(np.linalg.norm(diff) / np.linalg.norm(ref)),
            "pearson_all_values": rnd(pearson_all),
            "exactly_equal_fraction": rnd((diff == 0).mean()),
        },
        "per_dimension": {
            "description": "Pearson correlation of each embedding dimension across cells, and agreement of the "
                           "per-dimension means and standard deviations.",
            "pearson_across_cells": describe(dim_corr),
            "pearson_of_dimension_means": rnd(dim_mean_corr),
            "pearson_of_dimension_stds": rnd(dim_std_corr),
        },
        "retrieval": {
            "description": f"For {len(queries):,} random cells: the rank of the cell's own reference embedding among "
                           f"all {n:,} reference embeddings, by cosine similarity to its new embedding (0 = best match).",
            "n_queries": len(queries),
            "top1_accuracy": rnd((ranks == 0).mean()),
            "top10_accuracy": rnd((ranks < 10).mean()),
            "top100_accuracy": rnd((ranks < 100).mean()),
            "rank": describe(ranks),
        },
        "neighbourhood": {
            "description": f"For the same query cells: fraction of the {k} nearest neighbours (cosine, among all "
                           f"cells) shared between the reference and the new embedding space.",
            "k": k,
            "overlap": describe(overlap),
            "chance_level": rnd(k / (n - 1)),
        },
        "global_structure": {
            "description": "Linear CKA between the two embedding matrices (1 = same geometry up to rotation and "
                           "isotropic scaling).",
            "linear_cka": rnd(cka),
        },
    }
    return summary, sections


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="+", help="dataset names, e.g. neurips_2021")
    ap.add_argument("--ref-dir", default=ROOT / "benchmark/embedding_original", type=Path,
                    help="reference embeddings (default: original UCE)")
    ap.add_argument("--ref-seed", type=int, default=None,
                    help="seed suffix of the reference files (default: none, as the original UCE writes them)")
    ap.add_argument("--new-dir", default=ROOT / "benchmark/embedding", type=Path,
                    help="UCE-Modern embeddings")
    ap.add_argument("--new-seed", type=int, default=0, help="seed suffix of the UCE-Modern files")
    ap.add_argument("--out-dir", default=ROOT / "benchmark/evaluation", type=Path)
    ap.add_argument("--n-query", type=int, default=5000, help="cells sampled for retrieval and kNN")
    ap.add_argument("--k", type=int, default=15, help="neighbours for the kNN overlap")
    ap.add_argument("--sample-seed", type=int, default=0, help="seed for the query cells and random pairs")
    args = ap.parse_args()

    ref_label = "original" if args.ref_seed is None else f"seed{args.ref_seed}"
    ref_suffix = "" if args.ref_seed is None else f"_seed{args.ref_seed}"
    new_label = f"seed{args.new_seed}"
    for name in args.names:
        t0 = time.time()
        ref, ref_info = load(args.ref_dir / f"{name}_uce_embs{ref_suffix}.npy", args.ref_seed)
        new, new_info = load(args.new_dir / f"{name}_uce_embs_{new_label}.npy", args.new_seed)
        if ref.shape != new.shape:
            raise ValueError(f"{name}: shapes differ, reference {ref.shape} vs new {new.shape} "
                             f"(cells dropped by prepare_adata? see {args.new_dir}/{name}_uce_dropped_cells.txt)")

        summary, sections = compare(ref, new, args.n_query, args.k, args.sample_seed)
        result = {
            "dataset": name,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "reference": ref_info,
            "new": new_info,
            "sample_seed": args.sample_seed,
            "note": "UCE samples each cell's genes at random"
                    + (" (the original UCE with an unseeded RNG)" if args.ref_seed is None else "")
                    + ", so an exact match is impossible; compare 'same cell' scores with the random-pair "
                      "baseline and chance levels, and with a seed-vs-seed comparison of UCE-Modern.",
            "summary": summary,
            **sections,
        }
        args.out_dir.mkdir(parents=True, exist_ok=True)
        out = args.out_dir / f"{name}_{new_label}_vs_{ref_label}.json"
        out.write_text(json.dumps(result, indent=2) + "\n")
        print(f"{name}: wrote {rel(out)} ({time.time() - t0:.0f} s)")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
