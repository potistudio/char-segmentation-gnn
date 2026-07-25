"""Union-find postprocessing mirroring ``glyph-infer/src/postprocess.rs``."""

from __future__ import annotations

from collections import defaultdict


def group_characters(
    num_nodes: int,
    node_contour_ids: list[int],
    edge_src: list[int],
    edge_dst: list[int],
    probs: list[float],
    threshold: float,
) -> list[list[int]]:
    """Groups contours into character segments from thresholded edge probabilities."""
    if not node_contour_ids:
        return []

    num_contours = max(node_contour_ids) + 1
    parent = list(range(num_contours))

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            nxt = parent[x]
            parent[x] = root
            x = nxt
        return root

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    pair_prob: dict[tuple[int, int], float] = {}
    for src, dst, prob in zip(edge_src, edge_dst, probs, strict=True):
        ca = node_contour_ids[src]
        cb = node_contour_ids[dst]
        if ca == cb:
            continue
        key = (min(ca, cb), max(ca, cb))
        prev = pair_prob.get(key, 0.0)
        if prob > prev:
            pair_prob[key] = prob

    for (a, b), prob in pair_prob.items():
        if prob >= threshold:
            union(a, b)

    groups_map: dict[int, list[int]] = defaultdict(list)
    for contour in range(num_contours):
        groups_map[find(contour)].append(contour)

    return [sorted(contours) for contours in groups_map.values()]


def sort_groups_by_x(groups: list[list[int]], node_x: list[float], node_contour_ids: list[int]) -> list[list[int]]:
    """Orders groups left-to-right using the leftmost node in each group."""

    def contour_min_x(cids: list[int]) -> float:
        min_x = float("inf")
        cid_set = set(cids)
        for node, contour in enumerate(node_contour_ids):
            if contour in cid_set:
                min_x = min(min_x, node_x[node])
        return min_x

    return sorted(groups, key=contour_min_x)
