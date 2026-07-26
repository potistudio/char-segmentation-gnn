"""Metrics for the decision that actually ships.

The model is trained on point-pair edges but judged on contour groups:
``group_characters`` collapses every point edge between two contours to their
max probability, unions the pair when it clears the threshold, and returns the
components (``crates/glyph-infer/src/postprocess.rs``). Point-edge F1 measures
something far larger and far easier than that:

* 75-79% of edges sit inside a single contour and are always positive, and
  ``same_contour`` is fed to the model as an input feature, so they are free.
* Only 0.4-0.6% of edges are independent contour-pair decisions.
* Max pooling amplifies false positives. A contour pair spans 24 point edges
  at the Latin median, so a 1% per-edge false positive rate becomes
  ``1 - 0.99**24 = 21%`` at the pair level.

So this module scores contour pairs and whole groupings, and splits the
failures into the two modes that matter:

* **split** -- a character's contours land in more than one group ("こ", "た")
* **merge** -- a group holds contours from more than one character ("ll", "rI")

Both are driven by the same threshold in opposite directions, which is why
they have to be reported separately rather than folded into one F1.
"""

from __future__ import annotations

import numpy as np
import torch

#: Thresholds swept by default. ``group_characters`` needs a high threshold to
#: survive max pooling, so the grid is dense at the top.
DEFAULT_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9)


def contour_pair_index(contour_id: torch.Tensor, edge_index: torch.Tensor):
    """Maps each point edge onto the contour pair it votes on.

    Returns ``(inter, inverse, keys, stride)``: ``inter`` masks the edges that
    cross contours, ``inverse`` gives each of those the index of its undirected
    contour pair, and ``keys`` encodes the pair as ``lo * stride + hi``. Both
    stored directions of an edge collapse onto the same pair, matching what
    ``postprocess.rs`` aggregates over.

    Shared by the metrics and the pair-level loss so the two cannot drift.
    """
    src, dst = edge_index[0], edge_index[1]
    ca, cb = contour_id[src], contour_id[dst]
    inter = ca != cb
    stride = max(int(contour_id.max()) + 1 if contour_id.numel() else 1, 1)
    lo = torch.minimum(ca, cb)[inter]
    hi = torch.maximum(ca, cb)[inter]
    keys, inverse = torch.unique(lo * stride + hi, return_inverse=True)
    return inter, inverse, keys, stride


def scatter_amax(values: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
    """Per-bin maximum, the reduction ``group_characters`` applies to pairs."""
    return values.new_zeros(size).scatter_reduce_(
        0, index, values, reduce="amax", include_self=False
    )


def _components(num_contours: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Union-find over contour pairs, mirroring ``postprocess.rs``.

    Returns the root of each contour. Pairs never span graphs (no edge does),
    so one pass over the whole evaluation set stays graph-local.
    """
    parent = np.arange(num_contours, dtype=np.int64)

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    for u, v in zip(a.tolist(), b.tolist(), strict=True):
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[ru] = rv
    return np.fromiter((find(i) for i in range(num_contours)), dtype=np.int64,
                       count=num_contours)


def _dense(values: np.ndarray) -> tuple[np.ndarray, int]:
    """Relabels arbitrary ids to ``0..k-1`` and returns the count."""
    _, inverse = np.unique(values, return_inverse=True)
    inverse = np.asarray(inverse).ravel()
    return inverse, (int(inverse.max()) + 1 if inverse.size else 0)


class GroupingEvaluator:
    """Accumulates contour-pair decisions, then scores them at any threshold.

    Aggregation runs per batch on the GPU; scoring is deferred so a whole
    threshold sweep costs one pass over the model instead of one per
    threshold. The buffers hold one row per contour pair (38 per graph for
    Japanese, 16 for Latin), so a 7k-graph validation set is a few hundred
    thousand rows.
    """

    def __init__(self) -> None:
        self._a: list[np.ndarray] = []
        self._b: list[np.ndarray] = []
        self._prob: list[np.ndarray] = []
        self._label: list[np.ndarray] = []
        self._contour_char: list[np.ndarray] = []
        self._contour_graph: list[np.ndarray] = []
        self.texts: list[str] = []
        # GlyphData.__inc__ makes the ids disjoint inside one batch; these
        # carry that across batches so the whole pass shares one namespace.
        self._contour_base = 0
        self._char_base = 0
        self._graph_base = 0

    @torch.no_grad()
    def add(self, batch, probs: torch.Tensor) -> None:
        """Folds one batch of point-edge probabilities into contour pairs."""
        contour_id, char_id = batch.contour_id, batch.char_id
        num_contours = int(contour_id.max()) + 1 if contour_id.numel() else 0
        num_chars = int(char_id.max()) + 1 if char_id.numel() else 0

        # Every node of a contour carries the same character and graph, so an
        # arbitrary winner among the duplicates is the right answer.
        contour_char = contour_id.new_zeros(num_contours).scatter_(0, contour_id, char_id)
        contour_graph = contour_id.new_zeros(num_contours).scatter_(0, contour_id, batch.batch)

        # One row per undirected contour pair; both stored directions and every
        # point edge between the two contours collapse into it by max, exactly
        # as postprocess.rs does.
        inter, inverse, keys, stride = contour_pair_index(contour_id, batch.edge_index)
        pair_prob = scatter_amax(probs[inter].float(), inverse, keys.numel())
        pair_label = scatter_amax(batch.y[inter].float(), inverse, keys.numel())

        self._a.append((keys // stride).cpu().numpy() + self._contour_base)
        self._b.append((keys % stride).cpu().numpy() + self._contour_base)
        self._prob.append(pair_prob.cpu().numpy())
        self._label.append(pair_label.cpu().numpy())
        self._contour_char.append(contour_char.cpu().numpy() + self._char_base)
        self._contour_graph.append(contour_graph.cpu().numpy() + self._graph_base)
        self.texts.extend(getattr(batch, "text", []) or [])

        self._contour_base += num_contours
        self._char_base += num_chars
        self._graph_base += int(batch.num_graphs)

    @property
    def num_graphs(self) -> int:
        return self._graph_base

    def _stack(self) -> tuple[np.ndarray, ...]:
        empty = np.empty(0, dtype=np.int64)
        cat = (lambda parts, dtype: np.concatenate(parts) if parts else empty.astype(dtype))
        return (
            cat(self._a, np.int64), cat(self._b, np.int64),
            cat(self._prob, np.float32), cat(self._label, np.float32),
            cat(self._contour_char, np.int64), cat(self._contour_graph, np.int64),
        )

    def score(self, threshold: float, per_graph: bool = False) -> dict:
        """Scores contour pairs and groupings at one threshold."""
        a, b, prob, label, contour_char, contour_graph = self._stack()
        num_contours = contour_char.size
        num_graphs = self._graph_base

        predicted = prob >= threshold
        target = label >= 0.5
        tp = int((predicted & target).sum())
        fp = int((predicted & ~target).sum())
        fn = int((~predicted & target).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

        root = _components(num_contours, a[predicted], b[predicted])
        # Both partitions are graph-local: components never span graphs, and
        # character ids were made disjoint per graph by GlyphData.__inc__.
        group, num_groups = _dense(root)
        char, num_chars = _dense(contour_char)

        # A grouping is correct exactly when the group partition and the
        # character partition have the same number of cells as their common
        # refinement -- i.e. neither splits nor merges the other.
        joint, num_joint = _dense(group.astype(np.int64) * max(num_chars, 1) + char)

        def per_graph_counts(dense: np.ndarray, count: int) -> np.ndarray:
            """Number of cells each graph owns; cells never span graphs."""
            owner = np.zeros(count, dtype=np.int64)
            owner[dense] = contour_graph
            return np.bincount(owner, minlength=num_graphs)

        groups_per_graph = per_graph_counts(group, num_groups)
        chars_per_graph = per_graph_counts(char, num_chars)
        joint_per_graph = per_graph_counts(joint, num_joint)
        correct = (joint_per_graph == groups_per_graph) & (joint_per_graph == chars_per_graph)

        # Decompose the failures. Each (group, character) cell of the joint
        # partition tells us how a character was cut up and what a group holds.
        cell_group = np.zeros(num_joint, dtype=np.int64)
        cell_char = np.zeros(num_joint, dtype=np.int64)
        cell_group[joint] = group
        cell_char[joint] = char
        groups_per_char = np.bincount(cell_char, minlength=num_chars)
        chars_per_group = np.bincount(cell_group, minlength=num_groups)

        split = groups_per_char > 1
        merged = np.zeros(num_chars, dtype=bool)
        np.logical_or.at(merged, cell_char, chars_per_group[cell_group] > 1)

        out = {
            "threshold": threshold,
            "pair_precision": precision,
            "pair_recall": recall,
            "pair_f1": f1,
            "pairs": int(prob.size),
            "grouping_acc": float(correct.mean()) if num_graphs else 0.0,
            "char_split_rate": float(split.mean()) if num_chars else 0.0,
            "char_merge_rate": float(merged.mean()) if num_chars else 0.0,
            "chars": int(num_chars),
            "graphs": int(num_graphs),
        }
        if per_graph:
            out["correct"] = correct
        return out

    def sweep(self, thresholds=DEFAULT_THRESHOLDS) -> list[dict]:
        return [self.score(t) for t in thresholds]

    def best(self, thresholds=DEFAULT_THRESHOLDS) -> dict:
        """Returns the sweep entry with the highest grouping accuracy.

        Ties break toward the higher threshold: a false merge destroys two
        characters and cascades through union-find, while a false split leaves
        the rest of the line intact.
        """
        return max(self.sweep(thresholds),
                   key=lambda m: (m["grouping_acc"], m["threshold"]))


def format_sweep(sweep: list[dict]) -> str:
    """Renders a threshold sweep as a fixed-width table."""
    header = (f"{'thr':>5} {'pair P':>8} {'pair R':>8} {'pair F1':>8}"
              f" {'group acc':>10} {'split':>8} {'merge':>8}")
    lines = [header, "-" * len(header)]
    best = max(sweep, key=lambda m: (m["grouping_acc"], m["threshold"])) if sweep else None
    for m in sweep:
        mark = "  <- best" if m is best else ""
        lines.append(
            f"{m['threshold']:>5.2f} {m['pair_precision']:>8.4f} {m['pair_recall']:>8.4f}"
            f" {m['pair_f1']:>8.4f} {m['grouping_acc']:>10.4f}"
            f" {m['char_split_rate']:>8.4f} {m['char_merge_rate']:>8.4f}{mark}"
        )
    return "\n".join(lines)
