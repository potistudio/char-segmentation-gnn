"""Structural probes over a dataset -- the measurements the plan is built on.

Three questions the training log cannot answer:

1. **Edge composition.** What fraction of edges is the loss actually spending
   itself on? Intra-contour edges are always positive and ``same_contour`` is
   an input feature, so they are free; the shipping decision is the contour
   pair, and there are two orders of magnitude fewer of those.
2. **Receptive field.** How far can information travel in ``L`` message-passing
   hops, in em? If it is shorter than one character cell the model cannot see
   the neighbouring glyph, and character pitch -- the signal that separates
   "ll" from one wide glyph, or "こ" from two "一" -- is not observable.
3. **Connectivity ceiling.** Are a character's own contours even connected in
   the graph? If not, no threshold and no model can group them, and the
   recall ceiling is below 100% before training starts.

Usage:
    python -m glyph_gnn.analyze --data dataset/ja-train --limit 300
"""

from __future__ import annotations

import argparse
from collections import deque

import numpy as np
from tqdm.auto import tqdm

from .dataset import load_dataset, load_packed
from .pack import SAME_CONTOUR, is_packed


def sample_graphs(data: str, limit: int, seed: int, progress: bool) -> list:
    """Reads up to ``limit`` graphs, sampled evenly across the store."""
    if is_packed(data):
        full, _ = load_packed(data, val_frac=0.0, split_by_font=False)
        ids = np.linspace(0, len(full) - 1, min(limit, len(full))).astype(np.int64)
        return [full[int(i)] for i in tqdm(ids, desc="reading", unit="graph",
                                           disable=None if progress else True)]
    # The msgpack loader decodes whole shards, so read only as many as needed.
    full, _ = load_dataset(data, val_frac=0.0, split_by_font=False, seed=seed,
                           limit_shards=max(1, limit // 1000), progress=progress)
    rng = np.random.default_rng(seed)
    if len(full) > limit:
        return [full[int(i)] for i in rng.choice(len(full), limit, replace=False)]
    return full


def edge_composition(graphs: list) -> dict:
    """Splits edges into intra-contour, inter-contour, and contour pairs."""
    intra = intra_pos = inter = inter_pos = 0
    pairs = pair_pos = 0
    per_pair: list[int] = []
    nodes = contours = 0

    for g in graphs:
        cid = g.contour_id.numpy()
        src, dst = g.edge_index.numpy()
        y = g.y.numpy()
        same = g.edge_attr.numpy()[:, SAME_CONTOUR] != 0
        nodes += cid.size
        contours += np.unique(cid).size
        intra += int(same.sum())
        intra_pos += int(y[same].sum())
        inter += int((~same).sum())
        inter_pos += int(y[~same].sum())

        ca, cb = cid[src][~same], cid[dst][~same]
        key = np.minimum(ca, cb).astype(np.int64) * (cid.max() + 1) + np.maximum(ca, cb)
        uniq, inverse, counts = np.unique(key, return_inverse=True, return_counts=True)
        pairs += uniq.size
        per_pair.extend(counts.tolist())
        label = np.zeros(uniq.size)
        np.maximum.at(label, inverse, y[~same])
        pair_pos += int(label.sum())

    total = intra + inter
    return {
        "graphs": len(graphs), "nodes": nodes, "contours": contours, "edges": total,
        "intra_frac": intra / max(total, 1), "intra_pos": intra_pos / max(intra, 1),
        "inter_frac": inter / max(total, 1), "inter_pos": inter_pos / max(inter, 1),
        "pairs": pairs, "pair_pos": pair_pos / max(pairs, 1),
        "per_pair": np.array(per_pair) if per_pair else np.zeros(1),
        "decision_frac": pairs / max(total, 1),
    }


def _adjacency(num_nodes: int, src: np.ndarray, dst: np.ndarray) -> list[list[int]]:
    adj: list[list[int]] = [[] for _ in range(num_nodes)]
    for a, b in zip(src.tolist(), dst.tolist(), strict=True):
        adj[a].append(b)
    return adj


def receptive_field(graphs: list, hops: int, probes: int, seed: int) -> dict:
    """Max x-distance reachable in ``hops`` message-passing steps, in em."""
    rng = np.random.default_rng(seed)
    reach: list[float] = []
    widths: list[float] = []

    for g in graphs:
        x = g.x.numpy()[:, 0]
        src, dst = g.edge_index.numpy()
        adj = _adjacency(x.size, src, dst)
        widths.append(float(x.max() - x.min()))
        starts = rng.choice(x.size, size=min(probes, x.size), replace=False)
        for start in starts:
            seen = {int(start): 0}
            queue = deque([int(start)])
            while queue:
                u = queue.popleft()
                if seen[u] == hops:
                    continue
                for v in adj[u]:
                    if v not in seen:
                        seen[v] = seen[u] + 1
                        queue.append(v)
            idx = np.fromiter(seen.keys(), dtype=np.int64, count=len(seen))
            reach.append(float(np.abs(x[idx] - x[start]).max()))

    reach_arr = np.array(reach) if reach else np.zeros(1)
    width = float(np.mean(widths)) if widths else 0.0
    return {"hops": hops, "mean": float(reach_arr.mean()),
            "median": float(np.median(reach_arr)), "p95": float(np.percentile(reach_arr, 95)),
            "line_width": width, "frac": float(reach_arr.mean()) / max(width, 1e-9)}


def connectivity(graphs: list) -> dict:
    """Fraction of characters whose own contours are disconnected in the graph."""
    total = multi = broken = 0

    for g in graphs:
        cid = g.contour_id.numpy()
        chid = g.char_id.numpy()
        src, dst = g.edge_index.numpy()
        ca, cb = cid[src], cid[dst]
        cross = ca != cb

        adj: dict[int, set[int]] = {}
        for a, b in zip(ca[cross].tolist(), cb[cross].tolist(), strict=True):
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)

        contour_char: dict[int, int] = dict(zip(cid.tolist(), chid.tolist(), strict=True))
        by_char: dict[int, list[int]] = {}
        for contour, char in contour_char.items():
            by_char.setdefault(char, []).append(contour)

        for members in by_char.values():
            total += 1
            if len(members) < 2:
                continue
            multi += 1
            wanted = set(members)
            stack = [members[0]]
            seen = {members[0]}
            while stack:
                u = stack.pop()
                for v in adj.get(u, ()):
                    if v in wanted and v not in seen:
                        seen.add(v)
                        stack.append(v)
            broken += len(seen) != len(wanted)

    return {"chars": total, "multi": multi, "broken": broken,
            "broken_frac": broken / max(multi, 1),
            "ceiling": 1.0 - broken / max(total, 1)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True, help="msgpack shard directory or packed store")
    ap.add_argument("--limit", type=int, default=300, help="graphs to probe")
    ap.add_argument("--hops", default="4,8", help="comma-separated message-passing depths")
    ap.add_argument("--probes", type=int, default=12, help="start nodes per graph")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-progress", action="store_true")
    args = ap.parse_args()

    graphs = sample_graphs(args.data, args.limit, args.seed, not args.no_progress)
    print(f"\n{args.data}: probing {len(graphs)} graphs\n")

    c = edge_composition(graphs)
    print("=== edge composition ===")
    print(f"nodes/graph {c['nodes'] / c['graphs']:.1f} | contours/graph"
          f" {c['contours'] / c['graphs']:.1f} | edges/graph {c['edges'] / c['graphs']:.1f}")
    print(f"intra-contour  {c['intra_frac'] * 100:5.1f}% of edges,"
          f" {c['intra_pos'] * 100:5.1f}% positive")
    print(f"inter-contour  {c['inter_frac'] * 100:5.1f}% of edges,"
          f" {c['inter_pos'] * 100:5.1f}% positive")
    print(f"contour pairs  {c['pairs']:,} ({c['pairs'] / c['graphs']:.1f}/graph),"
          f" {c['pair_pos'] * 100:5.1f}% positive")
    print(f"  point edges per pair: median {np.median(c['per_pair']):.0f}"
          f" mean {c['per_pair'].mean():.1f} p95 {np.percentile(c['per_pair'], 95):.0f}")
    print(f"  independent decisions: {c['decision_frac'] * 100:.2f}% of edges")
    # Max pooling turns a per-edge false positive rate into a per-pair one.
    median = max(float(np.median(c["per_pair"])), 1.0)
    print(f"  a 1% per-edge false positive rate becomes"
          f" {(1 - 0.99 ** median) * 100:.0f}% per pair at the median")

    print("\n=== receptive field ===")
    for hops in (int(h) for h in args.hops.split(",")):
        r = receptive_field(graphs, hops, args.probes, args.seed)
        print(f"{hops:2d} hops: x-reach mean {r['mean']:.2f} em | median {r['median']:.2f}"
              f" | p95 {r['p95']:.2f} | line {r['line_width']:.2f} em"
              f" -> sees {r['frac'] * 100:.0f}% of the line")

    k = connectivity(graphs)
    print("\n=== connectivity ceiling ===")
    print(f"characters {k['chars']:,} | multi-contour {k['multi']:,}"
          f" ({k['multi'] / max(k['chars'], 1) * 100:.0f}%)")
    print(f"unreachable within their own contours: {k['broken']:,}"
          f" ({k['broken_frac'] * 100:.1f}% of multi-contour characters)")
    print(f"recall ceiling: {k['ceiling'] * 100:.1f}% of characters are groupable at all")


if __name__ == "__main__":
    main()
