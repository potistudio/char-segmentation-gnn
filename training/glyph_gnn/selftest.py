"""Self-test for the grouping metrics, run against a real shard.

:mod:`glyph_gnn.metrics` is what selects ``best.pt``, so if it drifts from the
postprocessor that ships, every later measurement is wrong while still looking
plausible. This checks it two ways:

* against known-answer inputs (oracle probabilities, all-merge, all-split)
* against :func:`glyph_gnn.postprocess.group_characters`, which mirrors
  ``crates/glyph-infer/src/postprocess.rs``, on random probabilities at every
  threshold and several batch sizes

Usage:
    python -m glyph_gnn.selftest --data dataset/ja-train
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from .dataset import load_shard
from .metrics import DEFAULT_THRESHOLDS, GroupingEvaluator
from .postprocess import group_characters


def _evaluate(graphs: list, probs: list[torch.Tensor], batch_size: int) -> GroupingEvaluator:
    evaluator = GroupingEvaluator()
    for batch in DataLoader(graphs, batch_size=batch_size):
        offset = evaluator.num_graphs
        flat = torch.cat([probs[offset + i] for i in range(int(batch.num_graphs))])
        evaluator.add(batch, flat)
    return evaluator


def check_known_answers(graphs: list) -> None:
    """Oracle and degenerate models must land on their known scores."""
    labels = [g.y.float() for g in graphs]
    distinct_chars = sum(int(g.char_id.unique().numel()) for g in graphs)

    oracle = _evaluate(graphs, labels, 16).score(0.5)
    assert abs(oracle["pair_f1"] - 1.0) < 1e-9, "oracle must reach a perfect pair F1"
    assert oracle["char_merge_rate"] == 0.0, "oracle must never merge characters"
    assert oracle["chars"] == distinct_chars, (
        f"character ids collided across the pass: {oracle['chars']} != {distinct_chars}"
    )
    assert oracle["graphs"] == len(graphs)
    # Whatever the oracle still splits is the graph's own fault: those contours
    # have no path between them, so no threshold can ever join them.
    ceiling = oracle["char_split_rate"]

    merged = _evaluate(graphs, [torch.ones_like(y) for y in labels], 16).score(0.5)
    assert merged["char_merge_rate"] > 0.9, "merging everything must merge nearly every char"
    assert merged["char_split_rate"] <= ceiling + 1e-9, (
        "merging cannot split more than the oracle strands"
    )

    split = _evaluate(graphs, [torch.zeros_like(y) for y in labels], 16).score(0.5)
    assert split["char_merge_rate"] == 0.0, "splitting everything cannot merge"

    print(f"known answers ok | recall ceiling {1 - ceiling:.4f}"
          f" | {oracle['chars']:,} characters | {oracle['pairs']:,} contour pairs")


def check_agrees_with_postprocess(graphs: list, seed: int) -> None:
    """The batched evaluator must match the per-graph postprocessor exactly."""
    generator = torch.Generator().manual_seed(seed)
    probs = [torch.rand(g.y.numel(), generator=generator) for g in graphs]

    for threshold in DEFAULT_THRESHOLDS:
        expected = []
        for graph, prob in zip(graphs, probs, strict=True):
            contour_id = graph.contour_id.tolist()
            src, dst = graph.edge_index.tolist()
            groups = group_characters(graph.num_nodes, contour_id, src, dst,
                                      prob.tolist(), threshold)
            by_char: dict[int, set[int]] = {}
            for contour, char in zip(contour_id, graph.char_id.tolist(), strict=True):
                by_char.setdefault(char, set()).add(contour)
            expected.append({frozenset(g) for g in groups}
                            == {frozenset(v) for v in by_char.values()})
        want = np.array(expected)

        for batch_size in (1, 7, 32):
            got = _evaluate(graphs, probs, batch_size).score(threshold, per_graph=True)
            if not np.array_equal(got["correct"], want):
                wrong = np.nonzero(got["correct"] != want)[0]
                raise AssertionError(
                    f"threshold {threshold} batch {batch_size}: disagreed on"
                    f" {wrong.size} graph(s), first at index {wrong[0]}"
                )
        print(f"  threshold {threshold:.2f}: {want.sum():>4}/{want.size} graphs correct,"
              " batched == postprocess")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True, help="directory holding .msgpack shards")
    ap.add_argument("--graphs", type=int, default=120, help="graphs to test against")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    shards = sorted(Path(args.data).glob("*.msgpack"))
    if not shards:
        raise FileNotFoundError(f"no .msgpack shards under {args.data}"
                                " (the self-test needs labeled shards, not a packed store)")
    graphs = load_shard(shards[0])[:args.graphs]
    print(f"{len(graphs)} graphs from {shards[0].name}\n")

    check_known_answers(graphs)
    print("\nagreement with glyph-infer's postprocessor:")
    check_agrees_with_postprocess(graphs, args.seed)
    print("\nall checks passed")


if __name__ == "__main__":
    main()
