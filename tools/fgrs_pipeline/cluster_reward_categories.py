#!/usr/bin/env python3
"""
Cluster reward categories hierarchically to construct a reward tree.

Reads JSONL at tools/fgrs_pipeline/output/reward_categories_by_id.jsonl, builds
co-occurrence sets for reward categories, computes cosine distances, and
performs average-linkage agglomerative clustering without external deps.

Outputs a JSON tree to tools/fgrs_pipeline/output/reward_hierarchy.json by default.

Usage:
  python tools/fgrs_pipeline/cluster_reward_categories.py \
    --input tools/fgrs_pipeline/output/reward_categories_by_id.jsonl \
    --mode all \
    --output tools/fgrs_pipeline/output/reward_hierarchy.json

Modes:
  - all:   use union of good_reward_categories and bad_reward_categories
  - good:  use only good_reward_categories
  - bad:   use only bad_reward_categories

Threshold mode:
  Provide --threshold FLOAT to merge categories whose cosine distance is < threshold.
  In threshold mode, outputs a flat clustering JSON with cluster members.

This script is dependency-free (pure Python) and suitable for environments
without SciPy/sklearn.
"""

from __future__ import annotations

import argparse
import json
import os
import math
from typing import Dict, List, Set, Tuple
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(iterable, total=None, desc=None, unit=None):
        return iterable


def load_jsonl(path: str) -> List[dict]:
    records: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # Be tolerant to trailing commas or minor formatting issues
                # Attempt a fallback: remove trailing commas
                cleaned = line.rstrip(",")
                records.append(json.loads(cleaned))
    return records


def build_category_sets(records: List[dict], mode: str) -> Dict[str, Set[str]]:
    """Build mapping: category -> set of item ids where it appears.

    mode in {"all", "good", "bad"}
    """
    cat2items: Dict[str, Set[str]] = {}
    for rec in tqdm(records, total=len(records), desc="records"):
        item_id = rec.get("id")
        if not item_id:
            # Skip records without id
            continue
        good = rec.get("good_reward_categories", [])
        bad = rec.get("bad_reward_categories", [])
        if mode == "all":
            cats = list(set(good) | set(bad))
        elif mode == "good":
            cats = list(set(good))
        elif mode == "bad":
            cats = list(set(bad))
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        for c in cats:
            if not c:
                continue
            s = cat2items.setdefault(c, set())
            s.add(item_id)
    return cat2items


def cosine_distance(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    denom = (len(a) * len(b)) ** 0.5
    if denom == 0:
        return 1.0
    cos_sim = inter / denom
    return 1.0 - cos_sim


def average_linkage_hclust(dist_init: Dict[Tuple[int, int], float], sizes_init: Dict[int, int]) -> List[Tuple[int, int, float, int]]:
    """Perform average-linkage hierarchical clustering.

    Args:
        dist_init: initial distances between singleton cluster ids (i<j) -> distance
        sizes_init: initial sizes for each singleton cluster id

    Returns:
        linkage: list of merges as (id_i, id_j, distance, new_size)
    """
    # Active cluster ids
    active = set(sizes_init.keys())
    sizes = dict(sizes_init)
    dist = dict(dist_init)  # mutable copy

    next_id = max(active) + 1 if active else 0
    linkage: List[Tuple[int, int, float, int]] = []
    init_count = len(active)
    pbar = tqdm(total=max(init_count - 1, 0), desc="merges")

    def get_pairs():
        return [(i, j, d) for (i, j), d in dist.items() if i in active and j in active]

    while len(active) > 1:
        # Find closest pair
        pairs = get_pairs()
        if not pairs:
            # Happens if distances are degenerate; break to avoid infinite loop
            break
        i, j, dmin = min(pairs, key=lambda t: t[2])
        # Merge i and j into new_id
        new_id = next_id
        next_id += 1
        new_size = sizes[i] + sizes[j]
        linkage.append((i, j, dmin, new_size))
        pbar.update(1)

        # Update active sets
        active.remove(i)
        active.remove(j)
        active.add(new_id)
        sizes[new_id] = new_size

        # Compute distances from new_id to each other active cluster k
        for k in list(active):
            if k == new_id:
                continue
            # Retrieve d(i,k) and d(j,k)
            a = min(i, k)
            b = max(i, k)
            dik = dist.get((a, b), 1.0)
            a = min(j, k)
            b = max(j, k)
            djk = dist.get((a, b), 1.0)
            d_new_k = (sizes[i] * dik + sizes[j] * djk) / (sizes[i] + sizes[j])
            a = min(new_id, k)
            b = max(new_id, k)
            dist[(a, b)] = d_new_k

        # Remove distances involving i or j
        to_delete = []
        for (a, b) in dist.keys():
            if a in (i, j) or b in (i, j):
                to_delete.append((a, b))
        for key in to_delete:
            dist.pop(key, None)

    try:
        pbar.close()
    except Exception:
        pass
    return linkage


def build_linkage(cat_sets: List[Set[str]]) -> List[Tuple[int, int, float, int]]:
    n = len(cat_sets)
    if n == 0:
        return []
    sizes_init = {i: 1 for i in range(n)}
    dist_init: Dict[Tuple[int, int], float] = {}
    total_pairs = n * (n - 1) // 2
    pbar = tqdm(total=total_pairs, desc="pairwise")
    for i in range(n):
        for j in range(i + 1, n):
            d = cosine_distance(cat_sets[i], cat_sets[j])
            dist_init[(i, j)] = d
            pbar.update(1)
    try:
        pbar.close()
    except Exception:
        pass
    return average_linkage_hclust(dist_init, sizes_init)


def threshold_union_clusters(cat_sets: List[Set[str]], labels: List[str], threshold: float) -> List[List[str]]:
    """Union-Find clustering: merge labels if cosine distance < threshold."""
    n = len(labels)
    parent = list(range(n))
    rank = [0] * n

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    # Pairwise merge under threshold
    total_pairs = n * (n - 1) // 2
    pbar = tqdm(total=total_pairs, desc="threshold")
    for i in range(n):
        for j in range(i + 1, n):
            d = cosine_distance(cat_sets[i], cat_sets[j])
            if d < threshold:
                union(i, j)
            pbar.update(1)
    try:
        pbar.close()
    except Exception:
        pass

    # Collect clusters
    clusters: Dict[int, List[str]] = {}
    for i in range(n):
        r = find(i)
        clusters.setdefault(r, []).append(labels[i])

    # Sort members and clusters for determinism
    flat_clusters = [sorted(members) for members in clusters.values()]
    flat_clusters.sort(key=lambda m: (-len(m), m))
    return flat_clusters


def linkage_to_tree(linkage: List[Tuple[int, int, float, int]], labels: List[str]) -> dict:
    """Convert linkage merges to a nested tree dict structure."""
    if not labels:
        return {}

    # Leaves
    nodes: Dict[int, dict] = {i: {"name": labels[i]} for i in range(len(labels))}
    current_id = len(labels)
    root_id = None
    for i, j, dist, new_size in linkage:
        node = {
            "name": f"cluster_{current_id}",
            "distance": dist,
            "size": new_size,
            "children": [nodes[i], nodes[j]],
        }
        nodes[current_id] = node
        root_id = current_id
        current_id += 1
    # If no merges (single label), root is the single leaf
    if root_id is None:
        root_id = 0
    return {
        "mode": None,  # filled by caller
        "num_leaves": len(labels),
        "root": nodes[root_id],
    }


def main():
    parser = argparse.ArgumentParser(description="Hierarchical clustering of reward categories to build a reward tree")
    parser.add_argument("--input", type=str, default=os.path.join("tools", "fgrs_pipeline", "output", "reward_categories_by_id.jsonl"), help="Path to JSONL input")
    parser.add_argument("--mode", type=str, choices=["all", "good", "bad"], default="all", help="Which categories to use")
    parser.add_argument("--output", type=str, default=os.path.join("tools", "fgrs_pipeline", "output", "reward_hierarchy.json"), help="Path to JSON output")
    parser.add_argument("--min_support", type=int, default=1, help="Minimum item count for a category to be included")
    parser.add_argument("--flat_threshold", type=float, default=None, help="Flat union clustering threshold; set to use flat clustering")
    parser.add_argument("--method", type=str, choices=["birch", "average"], default="birch", help="Hierarchical clustering method")
    parser.add_argument("--birch_threshold", type=float, default=0.5, help="BIRCH radius threshold")
    parser.add_argument("--birch_branching_factor", type=int, default=50, help="BIRCH branching factor")
    parser.add_argument("--vectorizer", type=str, choices=["bge", "set"], default="bge", help="Vectorization method for categories in birch mode")
    parser.add_argument("--bge_model", type=str, default="huggingface_cache/bge-en-icl", help="Embedding model name for bge vectorizer")
    parser.add_argument("--bge_batch_size", type=int, default=32, help="Batch size for embedding inference")
    parser.add_argument("--intra_merge_threshold", type=float, default=0.2, help="Merge close children within each cluster using cosine distance on co-occurrence sets")
    args = parser.parse_args()

    records = load_jsonl(args.input)
    if not records:
        raise SystemExit(f"No records found at {args.input}")

    cat2items = build_category_sets(records, args.mode)
    # Filter by support
    cat2items = {c: s for c, s in cat2items.items() if len(s) >= args.min_support}
    labels = sorted(cat2items.keys())
    cat_sets = [cat2items[c] for c in labels]

    if not labels:
        raise SystemExit("No categories after filtering; try lowering --min_support")

    # Ensure output dir exists
    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    if args.flat_threshold is not None:
        clusters = threshold_union_clusters(cat_sets, labels, args.flat_threshold)
        out = {
            "mode": args.mode,
            "threshold": args.flat_threshold,
            "num_categories": len(labels),
            "num_clusters": len(clusters),
            "clusters": [
                {"id": f"cluster_{i}", "members": members}
                for i, members in enumerate(clusters)
            ],
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"Wrote flat clustering (threshold {args.flat_threshold}) to: {args.output}")
        print(f"Mode: {args.mode}")
        print(f"Categories: {len(labels)}  Clusters: {len(clusters)}")
    else:
        if args.method == "average":
            linkage = build_linkage(cat_sets)
            tree = linkage_to_tree(linkage, labels)
            tree["mode"] = args.mode
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(tree, f, ensure_ascii=False, indent=2)
            print(f"Wrote reward hierarchy to: {args.output}")
            print(f"Mode: {args.mode}")
            print(f"Categories: {len(labels)}")
            if linkage:
                print(f"Merges: {len(linkage)}")
            else:
                print("Only one category present, no merges performed.")
        else:
            import numpy as np
            from sklearn.cluster import Birch
            if args.vectorizer == "bge":
                from transformers import AutoTokenizer, AutoModel
                import torch
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                tokenizer = AutoTokenizer.from_pretrained(args.bge_model)
                model = AutoModel.from_pretrained(args.bge_model)
                model = model.to(device)
                model.eval()
                def batch_embed(texts: List[str]) -> np.ndarray:
                    embs = []
                    for i in tqdm(range(0, len(texts), args.bge_batch_size), total=(len(texts) + args.bge_batch_size - 1) // args.bge_batch_size, desc="embed"):
                        batch = texts[i:i+args.bge_batch_size]
                        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                        with torch.no_grad():
                            outputs = model(**inputs)
                            hidden = outputs.last_hidden_state
                            mask = inputs["attention_mask"].unsqueeze(-1)
                            masked = hidden * mask
                            summed = masked.sum(dim=1)
                            counts = mask.sum(dim=1)
                            pooled = summed / torch.clamp(counts, min=1)
                            vec = torch.nn.functional.normalize(pooled, p=2, dim=1)
                        embs.append(vec.cpu().numpy())
                    return np.concatenate(embs, axis=0)
                X = batch_embed(labels)
            else:
                items = sorted({x for s in cat_sets for x in s})
                idx = {it: i for i, it in enumerate(items)}
                m = len(items)
                X = np.zeros((len(cat_sets), m), dtype=np.float32)
                for r, s in enumerate(cat_sets):
                    for it in s:
                        X[r, idx[it]] = 1.0
                norms = np.linalg.norm(X, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                X = X / norms

            birch = Birch(n_clusters=None, threshold=args.birch_threshold, branching_factor=args.birch_branching_factor)
            birch.fit(X)
            # Assign each category to its nearest BIRCH subcluster
            # Use predict(X) to get a label per input sample
            sub_labels = birch.predict(X).tolist()
            clusters_map: Dict[int, List[str]] = {}
            for i, c in enumerate(sub_labels):
                clusters_map.setdefault(int(c), []).append(labels[i])
            children = []
            label_index = {lb: idx for idx, lb in enumerate(labels)}
            support = {lb: len(cat2items[lb]) for lb in labels}
            for cid, members in sorted(clusters_map.items(), key=lambda kv: (-len(kv[1]), kv[1])):
                if args.intra_merge_threshold is not None:
                    idxs = [label_index[m] for m in members]
                    sub_sets = [cat_sets[i] for i in idxs]
                    groups = threshold_union_clusters(sub_sets, members, args.intra_merge_threshold)
                    reps = []
                    for g in groups:
                        rep = sorted(g, key=lambda x: (-support[x], x))[0]
                        reps.append(rep)
                    final_members = sorted(set(reps))
                else:
                    final_members = sorted(members)
                children.append({
                    "name": f"cluster_{cid}",
                    "size": len(final_members),
                    "children": [{"name": m} for m in final_members],
                })
            tree = {
                "mode": args.mode,
                "num_leaves": len(labels),
                "root": {
                    "name": "root",
                    "children": children,
                },
            }
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(tree, f, ensure_ascii=False, indent=2)
            print(f"Wrote BIRCH reward hierarchy to: {args.output}")
            print(f"Mode: {args.mode}")
            print(f"Categories: {len(labels)}  Clusters: {len(children)}")


if __name__ == "__main__":
    main()