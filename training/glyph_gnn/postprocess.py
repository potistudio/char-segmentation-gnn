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
    parent = list(range(num_nodes))

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

    for src, dst, prob in zip(edge_src, edge_dst, probs, strict=True):
        if prob >= threshold:
            union(src, dst)

    votes: dict[tuple[int, int], int] = defaultdict(int)
    for node in range(num_nodes):
        comp = find(node)
        contour = node_contour_ids[node]
        votes[(contour, comp)] += 1

    contour_comp: dict[int, tuple[int, int]] = {}
    for (contour, comp), count in votes.items():
        best_comp, best_count = contour_comp.get(contour, (comp, 0))
        if count > best_count:
            contour_comp[contour] = (comp, count)

    groups_map: dict[int, list[int]] = defaultdict(list)
    for contour, (comp, _) in contour_comp.items():
        groups_map[comp].append(contour)

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
