"""Self-test for the grouping metrics, run against a real shard.

:mod:`glyph_gnn.metrics` is what selects ``best.pt`` and :mod:`glyph_gnn.loss`
is what the model is pulled toward, so if either drifts from the postprocessor
that ships, every later measurement is wrong while still looking plausible.
This checks:

* the metrics against known-answer inputs (oracle probabilities, all-merge,
  all-split)
* the metrics against :func:`glyph_gnn.postprocess.group_characters`, which
  mirrors ``crates/glyph-infer/src/postprocess.rs``, on random probabilities at
  every threshold and several batch sizes
* the pair pooling against the hard max it stands in for
* that the pair loss reacts to a single spiked edge, which is the failure the
  point-edge loss cannot see
* that turning the pair term off reproduces the old point-only objective

Usage:
    python -m glyph_gnn.selftest --data dataset/ja-train
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from .dataset import load_shard
from .loss import focal_loss, glyph_loss, soft_max_pool
from .metrics import DEFAULT_THRESHOLDS, GroupingEvaluator, contour_pair_index
from .model import (
    C_AREA,
    C_H,
    C_MINX,
    C_MINY,
    C_W,
    GlyphEdgeGNN,
    contour_relations,
)
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


def check_single_graph_matches_batched(graphs: list) -> None:
    """The path ONNX traces must equal the path training takes.

    ``pool`` has two branches: ``batch=None`` reduces over every node, which is
    what the export traces because inference runs one line at a time, and the
    scattered branch, which is what training runs. A single graph has to give
    the same answer through both, or the model ships behaving differently from
    how it was trained -- silently, since nothing else would notice.
    """
    graph = graphs[0]
    model = GlyphEdgeGNN(node_dim=int(graph.x.shape[1]),
                         contour_dim=int(graph.contour_attr.shape[1]),
                         hidden=32, layers=2, dropout=0.0)
    model.eval()
    args = (graph.x, graph.edge_index, graph.edge_attr, graph.contour_id, graph.contour_attr)
    with torch.no_grad():
        exported = model(*args)
        trained = model(*args, torch.zeros(graph.num_nodes, dtype=torch.long))
    gap = float((exported - trained).abs().max())
    assert gap < 1e-5, f"single-graph and batched pooling disagree by {gap:.3e}"

    # And a two-graph batch must not leak the pool across the boundary.
    pair = Batch.from_data_list([graphs[0], graphs[1]])
    with torch.no_grad():
        together = model(pair.x, pair.edge_index, pair.edge_attr,
                         pair.contour_id, pair.contour_attr, pair.batch)
        alone = model(*args)
    leak = float((together[:graphs[0].y.numel()] - alone).abs().max())
    assert leak < 1e-5, f"batching changed a graph's logits by {leak:.3e} (pool leaked)"
    print(f"pooling isolation ok | export vs train {gap:.2e}, batch leak {leak:.2e}")


def check_contour_layout(graphs: list) -> None:
    """``contour_attr`` columns must line up with the geometry of the nodes.

    The layout is declared twice -- ``graph.rs`` writes it and ``model.py``
    reads it to derive the relational features -- so a reordering on either
    side has to fail here rather than quietly feed the attention bias numbers
    that mean something else.
    """
    # A pair with itself: fully overlapping, no offset, same winding.
    row = graphs[0].contour_attr[:1]
    self_rel = contour_relations(row, row)[0]
    assert torch.allclose(self_rel[:3], torch.zeros(3), atol=1e-6), "self offset must vanish"
    assert abs(float(self_rel[3]) - float(row[0, C_W])) < 1e-5, "self x-overlap is the width"
    assert abs(float(self_rel[4]) - float(row[0, C_H])) < 1e-5, "self y-overlap is the height"
    assert abs(float(self_rel[7])) < 1e-6 and abs(float(self_rel[8])) < 1e-6
    assert abs(float(self_rel[9])) < 1e-5, "self size ratio must be log 1"
    assert float(self_rel[10]) > 0, "a contour must agree with its own winding"

    checked = 0
    for graph in graphs[:25]:
        xs, ys = graph.x[:, 0], graph.x[:, 1]
        for contour in range(int(graph.contour_attr.shape[0])):
            owned = graph.contour_id == contour
            if not bool(owned.any()):
                continue
            box = graph.contour_attr[contour]
            # Resampled points land inside the true outline box, and no more
            # than one sample spacing (0.05 em) short of its edges.
            assert float(box[C_MINX]) <= float(xs[owned].min()) + 0.06
            assert float(box[C_MINY]) <= float(ys[owned].min()) + 0.06
            assert float(box[C_MINX] + box[C_W]) >= float(xs[owned].max()) - 0.06
            assert float(box[C_MINY] + box[C_H]) >= float(ys[owned].max()) - 0.06
            # A polygon cannot enclose more than its own bounding box.
            assert abs(float(box[C_AREA])) <= float(box[C_W] * box[C_H]) + 1e-6
            checked += 1

    print(f"contour layout ok | {checked:,} contour boxes agree with their nodes")


def check_pooling() -> None:
    """The pair pooling must approach the max that inference actually takes."""
    logits = torch.tensor([-3.0, -2.5, 4.0, -2.0, -3.5, 0.5])
    index = torch.zeros(6, dtype=torch.long)

    cold = float(soft_max_pool(logits, index, 1, temperature=0.01))
    assert abs(cold - float(logits.max())) < 1e-3, (
        f"temperature -> 0 must reproduce the hard max, got {cold}"
    )
    warm = float(soft_max_pool(logits, index, 1, temperature=1000.0))
    assert abs(warm - float(logits.mean())) < 1e-2, (
        f"a large temperature must tend to the mean, got {warm}"
    )
    # Between the two it stays inside the range, and above the mean: pooling
    # must never be dragged below average by the quiet edges.
    mid = float(soft_max_pool(logits, index, 1, temperature=1.0))
    assert float(logits.mean()) < mid < float(logits.max())

    # Two pairs at once, to catch a scatter that leaks across bins.
    split = soft_max_pool(logits, torch.tensor([0, 0, 0, 1, 1, 1]), 2, 0.01)
    assert abs(float(split[0]) - 4.0) < 1e-3 and abs(float(split[1]) - 0.5) < 1e-3

    print("pooling ok | hard max, mean, and per-bin isolation all reproduce")


def check_pair_loss_punishes_one_spike(graphs: list) -> None:
    """One edge crossing the threshold merges two characters; the loss must see it.

    This is the whole point of the pair term. A single false positive among a
    pair's two dozen point edges is a rounding error to the point-edge loss but
    destroys a character downstream, so the pair term has to react far more
    strongly than the point term does.
    """
    graph = next(g for g in graphs
                 if (g.y == 0).any() and (g.contour_id[g.edge_index[0]]
                                          != g.contour_id[g.edge_index[1]]).any())
    contour_id, edge_index, y = graph.contour_id, graph.edge_index, graph.y
    inter, _, keys, _ = contour_pair_index(contour_id, edge_index)

    # A moderately confident, entirely correct model.
    correct = torch.where(y > 0.5, torch.full_like(y, 2.5), torch.full_like(y, -2.5))
    settings = dict(pair_weight=0.5, intra_weight=0.05, temperature=1.0)
    base, base_terms = glyph_loss(correct, y, contour_id, edge_index, **settings)

    # Push one point edge of one negative pair over the line, as a false merge
    # does. The other stored direction stays correct -- the max still takes it.
    inter_edges = torch.nonzero(inter).squeeze(-1)
    negative = inter_edges[torch.nonzero(y[inter] == 0).squeeze(-1)[0]]
    spiked = correct.clone()
    spiked[negative] = 5.0
    hurt, hurt_terms = glyph_loss(spiked, y, contour_id, edge_index, **settings)

    point_growth = hurt_terms["point"] / base_terms["point"]
    pair_growth = hurt_terms["pair"] / base_terms["pair"]
    assert hurt > base, "a false merge must cost more than a clean prediction"
    assert pair_growth > point_growth, (
        "the pair term must react harder than the point term, got pair"
        f" x{pair_growth:.2f} vs point x{point_growth:.2f}"
    )
    print(f"pair loss ok | one spiked edge out of {int(inter.sum()):,} costs"
          f" point x{point_growth:.2f}, pair x{pair_growth:.2f}"
          f" ({keys.numel()} pairs)")


def check_point_only_matches_focal(graphs: list) -> None:
    """With the pair term off and intra edges unweighted, nothing has changed.

    Guards the refactor: an old point-only run must still be reproducible.
    """
    graph = graphs[0]
    torch.manual_seed(3)
    logits = torch.randn(graph.y.numel())
    combined, _ = glyph_loss(logits, graph.y, graph.contour_id, graph.edge_index,
                             alpha=0.25, gamma=2.0, pair_weight=0.0, intra_weight=1.0)
    expected = focal_loss(logits, graph.y, alpha=0.25, gamma=2.0)
    assert torch.allclose(combined, expected), "pair_weight=0 must be the old objective"
    print("compatibility ok | pair_weight=0, intra_weight=1 reproduces focal_loss")


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
    check_contour_layout(graphs)
    check_single_graph_matches_batched(graphs)
    check_pooling()
    check_point_only_matches_focal(graphs)
    check_pair_loss_punishes_one_spike(graphs)
    print("\nagreement with glyph-infer's postprocessor:")
    check_agrees_with_postprocess(graphs, args.seed)
    print("\nall checks passed")


if __name__ == "__main__":
    main()
