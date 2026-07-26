"""Grouping decoders and the training signal for the group scorer.

``group_characters`` merges any contour pair whose score clears one threshold
and never reconsiders. Two things follow. A false merge and a false split pull
that threshold in opposite directions, so no setting fixes both -- which is
exactly the bind "こ" splitting and "ll" merging put us in. And union-find is
transitive, so one bad link swallows a whole run of characters with no way back.

:func:`greedy_groups` keeps the pair scores for what they are good at -- the
order to try merges in -- and hands each accept/reject to a scorer that sees
both groups and their union. :func:`training_candidates` builds what that
scorer learns from, out of the labels already in the dataset.
"""

from __future__ import annotations

import numpy as np
import torch

from .model import group_accumulators

MERGEABLE = ("sum_state", "max_state", "count", "arc", "area")


def merge_accumulators(acc: dict[str, torch.Tensor], left: torch.Tensor,
                       right: torch.Tensor) -> dict[str, torch.Tensor]:
    """Accumulators of ``left | right``, from the two sides alone.

    Sums add, boxes take the wider extent. No contour is touched, which is what
    makes a greedy walk over thousands of candidate merges affordable.
    """
    merged = {key: acc[key][left] + acc[key][right] for key in MERGEABLE}
    merged["max_state"] = torch.maximum(acc["max_state"][left], acc["max_state"][right])
    merged["box_min"] = torch.minimum(acc["box_min"][left], acc["box_min"][right])
    merged["box_max"] = torch.maximum(acc["box_max"][left], acc["box_max"][right])
    return merged


def singleton_accumulators(contour_states: torch.Tensor,
                           contour_attr: torch.Tensor) -> dict[str, torch.Tensor]:
    """One accumulator per contour, the starting point of agglomeration."""
    ids = torch.arange(contour_attr.shape[0], device=contour_attr.device)
    return group_accumulators(contour_states, contour_attr, ids, ids,
                              int(contour_attr.shape[0]))


@torch.no_grad()
def greedy_groups(
    head,
    contour_states: torch.Tensor,
    contour_attr: torch.Tensor,
    contour_graph: torch.Tensor,
    line: torch.Tensor,
    pair_a: torch.Tensor,
    pair_b: torch.Tensor,
    pair_prob: torch.Tensor,
    threshold: float,
    floor: float = 0.1,
) -> np.ndarray:
    """Agglomerates contours, asking the scorer before every merge.

    Returns the group root of every contour, the same shape of answer
    ``group_characters`` gives.

    The pair probabilities set the order candidates are tried in; the scorer
    decides each one, looking at both groups as they stand and at what their
    union would be. Unlike a threshold on the pair score, that judgement changes
    as groups grow: a link that is fine between two bare strokes stops being
    fine once one side is already a whole character.

    Graphs are independent, so they advance in lockstep: each round every graph
    offers its next candidate pair and all offers are scored in one batch. That
    is arithmetically identical to walking each graph on its own, but costs one
    model call per round rather than one per candidate -- hundreds instead of
    tens of thousands over a validation pass.

    ``floor`` drops pairs the edge model is sure about, purely to keep the
    candidate list short; the accept decision belongs to the scorer.
    """
    num_contours = int(contour_attr.shape[0])
    parent = np.arange(num_contours, dtype=np.int64)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    acc = singleton_accumulators(contour_states, contour_attr)
    keep = pair_prob >= floor
    order = torch.argsort(pair_prob[keep], descending=True)
    a = pair_a[keep][order].cpu().numpy()
    b = pair_b[keep][order].cpu().numpy()
    if a.size == 0:
        return parent

    # Strongest link the edge model sees between any two groups. Max survives
    # merging, so folding a group in is a row/column maximum.
    bridge = np.zeros((num_contours, num_contours), dtype=np.float32)
    probs = pair_prob.cpu().numpy()
    all_a, all_b = pair_a.cpu().numpy(), pair_b.cpu().numpy()
    bridge[all_a, all_b] = probs
    bridge[all_b, all_a] = probs

    # Bucket candidates by graph so every graph can be advanced together.
    graph_of = contour_graph.cpu().numpy()
    per_graph: dict[int, list[int]] = {}
    for position, graph in enumerate(graph_of[a].tolist()):
        per_graph.setdefault(graph, []).append(position)
    cursor = dict.fromkeys(per_graph, 0)

    while True:
        # One proposal per graph that still has candidates.
        proposals: list[tuple[int, int, int]] = []
        for graph, positions in per_graph.items():
            index = cursor[graph]
            while index < len(positions):
                position = positions[index]
                ra, rb = find(int(a[position])), find(int(b[position]))
                index += 1
                if ra != rb:
                    proposals.append((graph, ra, rb))
                    break
            cursor[graph] = index
        if not proposals:
            return parent

        left = torch.tensor([p[1] for p in proposals], device=contour_attr.device)
        right = torch.tensor([p[2] for p in proposals], device=contour_attr.device)
        owners = torch.tensor([p[0] for p in proposals], device=contour_attr.device)
        union = merge_accumulators(acc, left, right)
        links = torch.tensor(bridge[[p[1] for p in proposals], [p[2] for p in proposals]],
                             device=contour_attr.device)
        scores = torch.sigmoid(head(
            {key: value[left] for key, value in acc.items()},
            {key: value[right] for key, value in acc.items()},
            union, line[owners], links,
        )).cpu().numpy()

        for (_, ra, rb), score in zip(proposals, scores.tolist(), strict=True):
            if score < threshold:
                continue
            # Fold the loser into the winner, accumulators included, so later
            # rounds see the merged group.
            for key in MERGEABLE:
                acc[key][rb] = acc[key][rb] + acc[key][ra]
            acc["max_state"][rb] = torch.maximum(acc["max_state"][rb], acc["max_state"][ra])
            acc["box_min"][rb] = torch.minimum(acc["box_min"][rb], acc["box_min"][ra])
            acc["box_max"][rb] = torch.maximum(acc["box_max"][rb], acc["box_max"][ra])
            np.maximum(bridge[rb], bridge[ra], out=bridge[rb])
            bridge[:, rb] = bridge[rb]
            parent[ra] = rb


def pair_bridge(pair_a: torch.Tensor, pair_b: torch.Tensor, pair_prob: torch.Tensor,
                num_contours: int) -> np.ndarray:
    """Dense contour-by-contour matrix of the edge model's link probabilities."""
    bridge = np.zeros((num_contours, num_contours), dtype=np.float32)
    probs = pair_prob.detach().cpu().numpy()
    a, b = pair_a.cpu().numpy(), pair_b.cpu().numpy()
    bridge[a, b] = probs
    bridge[b, a] = probs
    return bridge


def training_candidates(contour_char: torch.Tensor, contour_graph: torch.Tensor,
                        contour_attr: torch.Tensor, bridge: np.ndarray,
                        seed: int = 0):
    """Merge questions for the scorer, from labels the dataset already has.

    Each candidate is a pair of contour groups plus the answer to "do these
    belong to the same character". Positives cut one character in two, so they
    cover the partial states a decoder actually passes through. Negatives pair
    a character -- or part of one -- with its nearest neighbour on the line.
    Nothing needs annotating.

    Returns ``(member_contour, member_group, left, right, labels, owners)``:
    the first two list group membership, ``left``/``right`` index the pairs and
    ``owners`` gives each pair its graph. A batch holds a few hundred
    characters, so walking them costs well under a millisecond.
    """
    device = contour_attr.device
    char_of = contour_char.tolist()
    graph_of = contour_graph.tolist()
    centre_of = contour_attr[:, 0].tolist()

    members: dict[int, list[int]] = {}
    graph: dict[int, int] = {}
    centre: dict[int, float] = {}
    for contour, char in enumerate(char_of):
        members.setdefault(char, []).append(contour)
        graph[char] = graph_of[contour]
        centre[char] = centre.get(char, 0.0) + centre_of[contour]
    for char, own in members.items():
        centre[char] /= len(own)

    by_graph: dict[int, list[int]] = {}
    for char, owner in graph.items():
        by_graph.setdefault(owner, []).append(char)

    member_contour: list[int] = []
    member_group: list[int] = []
    left: list[int] = []
    right: list[int] = []
    labels: list[float] = []
    owners: list[int] = []
    groups = 0

    def add_group(contours: list[int]) -> int:
        nonlocal groups
        member_contour.extend(contours)
        member_group.extend([groups] * len(contours))
        groups += 1
        return groups - 1

    links: list[float] = []

    def ask(a: list[int], b: list[int], label: float, owner: int) -> None:
        left.append(add_group(a))
        right.append(add_group(b))
        labels.append(label)
        owners.append(owner)
        # Same quantity the decoder will hand the head: the strongest link the
        # edge model sees across the two groups.
        links.append(float(bridge[np.ix_(a, b)].max()))

    rng = np.random.default_rng(seed)
    for owner, chars in by_graph.items():
        for char in chars:
            own = members[char]
            # Random cuts, so the head sees partial groups of every size rather
            # than only whole characters against whole characters. One cut per
            # contour keeps positives from being outnumbered by the two
            # negatives every character contributes.
            for _ in range(min(len(own) - 1, 3)):
                shuffled = list(own)
                rng.shuffle(shuffled)
                cut = int(rng.integers(1, len(shuffled)))
                ask(shuffled[:cut], shuffled[cut:], 1.0, owner)
            others = [c for c in chars if c != char]
            if not others:
                continue
            # Nearest other character of the same line, by centre x.
            partner = min(others, key=lambda c: abs(centre[c] - centre[char]))
            ask(own, members[partner], 0.0, owner)
            if len(own) > 1:
                # Part of a character against the neighbour: the mistake a
                # decoder makes once it has half-built a character.
                ask(own[:-1], members[partner], 0.0, owner)

    def to_long(values: list[int]) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.long, device=device)

    dtype = contour_attr.dtype
    return (to_long(member_contour), to_long(member_group), to_long(left), to_long(right),
            torch.tensor(labels, dtype=dtype, device=device), to_long(owners),
            torch.tensor(links, dtype=dtype, device=device))
