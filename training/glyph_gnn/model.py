"""Edge-classifying GNN.

The attention layer is a GAT variant implemented with only gather /
scatter_add tensor ops so the whole model exports cleanly to ONNX
(PyTorch Geometric's built-in conv layers rely on scatter kernels that
are fragile under torch.onnx). Semantics follow GATv2 with edge features.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeGATLayer(nn.Module):
    """Multi-head graph attention with edge features, ONNX-exportable."""

    def __init__(self, dim: int, edge_dim: int, heads: int = 4, dropout: float = 0.0):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.head_dim = dim // heads

        self.lin_src = nn.Linear(dim, dim)
        self.lin_dst = nn.Linear(dim, dim)
        self.lin_edge = nn.Linear(edge_dim, dim)
        # GATv2-style attention: a^T LeakyReLU(W_s h_s + W_d h_d + W_e e)
        self.att = nn.Parameter(torch.empty(1, heads, self.head_dim))
        nn.init.xavier_uniform_(self.att)
        self.out = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        src, dst = edge_index[0], edge_index[1]
        h, hd = self.heads, self.head_dim

        h_src = self.lin_src(x).view(-1, h, hd)[src]     # [E, H, D]
        h_dst = self.lin_dst(x).view(-1, h, hd)[dst]
        h_edge = self.lin_edge(edge_attr).view(-1, h, hd)

        z = F.leaky_relu(h_src + h_dst + h_edge, 0.2)
        logits = (z * self.att).sum(dim=-1)              # [E, H]

        # Softmax over incoming edges of each destination node. Global max
        # subtraction keeps exp() stable and, unlike per-node scatter-max,
        # lowers to plain ONNX ops.
        logits = logits - logits.max()
        num = torch.exp(logits)                          # [E, H]
        denom = x.new_zeros(n, h).scatter_add_(0, dst.unsqueeze(-1).expand(-1, h), num)
        alpha = num / (denom[dst] + 1e-9)                # [E, H]
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        msg = h_src * alpha.unsqueeze(-1)                # [E, H, D]
        agg = x.new_zeros(n, h, hd).scatter_add_(
            0, dst.view(-1, 1, 1).expand(-1, h, hd), msg
        )
        return self.norm(x + self.out(agg.reshape(n, h * hd)))


# Columns of a contour_attr row, matching crates/glyph-core/src/graph.rs.
# glyph_gnn.selftest checks these against the geometry of a real shard, so a
# reordering on the Rust side fails loudly instead of feeding the model
# plausible nonsense.
C_CX, C_CY, C_W, C_H, C_MINX, C_MINY, C_ARC, C_AREA = range(8)
#: Width of :func:`contour_relations`.
CONTOUR_RELATIONS = 11


def contour_relations(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Geometry between two contours, from their feature rows.

    Broadcasts over leading dimensions, so the same function serves the
    all-pairs bias inside the contour attention and the single pair behind one
    edge in the classifier.

    These are derived rather than stored. As edge columns they would cost about
    13 GiB on ja-train's 700M edges -- ``dist`` alone was 2.62 GiB -- and every
    one of them is a function of two contour rows that are already in memory.

    What they carry is the distinction the point-level features cannot express.
    Two 'l' stems sit side by side: their boxes do not overlap in x and the gap
    is a fraction of an em. The two strokes of "二" overlap in x completely and
    are stacked in y. A counter winds against the contour that encloses it and
    sits strictly inside its box.
    """
    eps = 1e-6
    ax0, ay0 = a[..., C_MINX], a[..., C_MINY]
    ax1, ay1 = ax0 + a[..., C_W], ay0 + a[..., C_H]
    bx0, by0 = b[..., C_MINX], b[..., C_MINY]
    bx1, by1 = bx0 + b[..., C_W], by0 + b[..., C_H]

    # Signed: positive is an overlap, negative is a gap, both in em so they are
    # directly comparable to character pitch.
    over_x = torch.minimum(ax1, bx1) - torch.maximum(ax0, bx0)
    over_y = torch.minimum(ay1, by1) - torch.maximum(ay0, by0)
    span_x = torch.minimum(a[..., C_W], b[..., C_W]).clamp(min=eps)
    span_y = torch.minimum(a[..., C_H], b[..., C_H]).clamp(min=eps)

    dx = b[..., C_CX] - a[..., C_CX]
    dy = b[..., C_CY] - a[..., C_CY]

    # How deeply one box sits inside the other; positive means enclosed.
    a_in_b = torch.minimum(torch.minimum(ax0 - bx0, ay0 - by0),
                           torch.minimum(bx1 - ax1, by1 - ay1))
    b_in_a = torch.minimum(torch.minimum(bx0 - ax0, by0 - ay0),
                           torch.minimum(ax1 - bx1, ay1 - by1))

    area_a, area_b = a[..., C_AREA], b[..., C_AREA]
    return torch.stack([
        dx,
        dy,
        torch.sqrt(dx * dx + dy * dy),
        over_x,
        over_y,
        (over_x / span_x).clamp(-4.0, 1.0),
        (over_y / span_y).clamp(-4.0, 1.0),
        a_in_b,
        b_in_a,
        torch.log((a[..., C_ARC] + eps) / (b[..., C_ARC] + eps)),
        # Opposite winding is the signature of an outer contour and its hole.
        torch.sign(area_a) * torch.sign(area_b),
    ], dim=-1)


def scatter_mean(values: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
    """Per-bin mean over rows, as gather/scatter_add so it exports to ONNX."""
    spread = index.unsqueeze(-1).expand(-1, values.shape[-1])
    total = values.new_zeros(size, values.shape[-1]).scatter_add_(0, spread, values)
    counts = values.new_zeros(size).scatter_add_(
        0, index, torch.ones_like(index, dtype=values.dtype)
    )
    return total / counts.clamp(min=1).unsqueeze(-1)


class ContourLayer(nn.Module):
    """Pools points into contour supernodes, attends across them, scatters back.

    Point-level message passing covers about 0.5 em in four hops, less than one
    character cell, so an edge never sees the glyph next door. A contour is a
    whole stroke and a line holds only 10-18 of them, so attention at that
    level spans the entire line for a few percent of the cost: 18^2 pairs
    against 6,000 point edges.

    Attention is dense rather than an edge list. Inference runs one line at a
    time, so the traced graph is a plain [C, C] product and the export needs no
    adjacency input; ``mask`` only exists to keep a training batch from
    attending across graph boundaries.
    """

    def __init__(self, dim: int, contour_dim: int, heads: int = 4, dropout: float = 0.0,
                 relations: bool = True):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.head_dim = dim // heads
        # The supernode starts from its own geometry (centroid, bbox, arc
        # length, signed area) plus the mean of the points it owns.
        self.embed = nn.Linear(contour_dim, dim)
        # Relative geometry enters as a per-head attention bias rather than as
        # more channels: whether two strokes overlap in x is a property of the
        # pair, and the pair is exactly what a score indexes.
        self.rel_bias = nn.Linear(CONTOUR_RELATIONS, heads) if relations else None
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.norm_contour = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = dropout

    def forward(self, hx: torch.Tensor, contour_id: torch.Tensor,
                contour_attr: torch.Tensor, relations: torch.Tensor,
                mask: torch.Tensor | None) -> torch.Tensor:
        contours = contour_attr.shape[0]
        h, hd = self.heads, self.head_dim
        z = self.norm_contour(
            scatter_mean(hx, contour_id, contours) + self.embed(contour_attr)
        )

        def heads_of(linear: nn.Module) -> torch.Tensor:
            return linear(z).view(contours, h, hd).transpose(0, 1)  # [H, C, D]

        scores = heads_of(self.query) @ heads_of(self.key).transpose(-2, -1)
        scores = scores * (self.head_dim ** -0.5)
        if self.rel_bias is not None:
            scores = scores + self.rel_bias(relations).permute(2, 0, 1)
        if mask is not None:
            scores = scores + mask
        weights = F.dropout(scores.softmax(dim=-1), p=self.dropout, training=self.training)
        mixed = (weights @ heads_of(self.value)).transpose(0, 1).reshape(contours, h * hd)

        z = z + self.proj(mixed)
        return self.norm(hx + self.out(z)[contour_id])


#: Width of :func:`group_scalars`.
GROUP_SCALARS = 7


def group_accumulators(contour_states: torch.Tensor, contour_attr: torch.Tensor,
                       member_contour: torch.Tensor, member_group: torch.Tensor,
                       num_groups: int) -> dict[str, torch.Tensor]:
    """Reduces each candidate group's members into mergeable accumulators.

    Every entry is a sum, a min or a max, so the union of two groups follows
    from their accumulators alone. That is what lets the decoder walk a greedy
    agglomeration -- thousands of candidate merges -- without re-pooling any
    group from its contours.

    ``member_contour`` and ``member_group`` are parallel: they list which
    contour belongs to which candidate, so a contour may appear in several
    candidates and a candidate need not be part of any partition.
    """
    states = contour_states[member_contour]
    rows = contour_attr[member_contour]
    width = states.shape[-1]
    spread = member_group.unsqueeze(-1).expand(-1, width)

    box_min = torch.stack([rows[:, C_MINX], rows[:, C_MINY]], dim=-1)
    box_max = box_min + torch.stack([rows[:, C_W], rows[:, C_H]], dim=-1)
    pair = member_group.unsqueeze(-1).expand(-1, 2)

    def add(values: torch.Tensor, index: torch.Tensor, cols: int) -> torch.Tensor:
        return values.new_zeros(num_groups, cols).scatter_add_(0, index, values)

    return {
        "sum_state": add(states, spread, width),
        "max_state": states.new_zeros(num_groups, width).scatter_reduce_(
            0, spread, states, reduce="amax", include_self=False),
        "count": add(torch.ones_like(member_group, dtype=states.dtype).unsqueeze(-1),
                     member_group.unsqueeze(-1), 1),
        "box_min": box_min.new_full((num_groups, 2), float("inf")).scatter_reduce_(
            0, pair, box_min, reduce="amin", include_self=False),
        "box_max": box_max.new_full((num_groups, 2), float("-inf")).scatter_reduce_(
            0, pair, box_max, reduce="amax", include_self=False),
        "arc": add(rows[:, C_ARC].unsqueeze(-1), member_group.unsqueeze(-1), 1),
        "area": add(rows[:, C_AREA].abs().unsqueeze(-1), member_group.unsqueeze(-1), 1),
    }


def group_scalars(acc: dict[str, torch.Tensor]) -> torch.Tensor:
    """Shape statistics of a group, from its accumulators.

    A character occupies roughly one cell and holds a bounded amount of ink, so
    a group that is twice as wide as the line's pitch, or that carries a stroke
    length no single glyph would, is the shape of a wrong merge.
    """
    eps = 1e-6
    extent = (acc["box_max"] - acc["box_min"]).clamp(min=0.0)
    width, height = extent[:, 0:1], extent[:, 1:2]
    count, arc, area = acc["count"], acc["arc"], acc["area"]
    return torch.cat([
        width,
        height,
        count,
        arc,
        area,
        width / height.clamp(min=eps),
        arc / (width + height).clamp(min=eps),
    ], dim=-1)


def group_descriptor(acc: dict[str, torch.Tensor]) -> torch.Tensor:
    """One row per group: pooled states plus its shape statistics."""
    mean_state = acc["sum_state"] / acc["count"].clamp(min=1.0)
    return torch.cat([mean_state, acc["max_state"], group_scalars(acc)], dim=-1)


#: Width of :func:`group_descriptor`, given a hidden width.
def descriptor_width(hidden: int) -> int:
    return hidden * 2 + GROUP_SCALARS


class GroupHead(nn.Module):
    """Scores whether two groups of contours belong to the same character.

    Not "is this set exactly one character": that question has no answer for the
    partial groups an agglomerative decoder passes through. One stroke of "二"
    is indistinguishable from the whole of "一", so a scorer trained on complete
    characters marks every intermediate state as wrong and the decoder can never
    build anything up. Measured: it left 25% of characters split against
    union-find's 4.7%.

    Asking about a *pair* of groups is well posed at every stage, which is why
    learned agglomerative clustering is normally framed this way. Both sides and
    their union are described, so the head sees what the merge would produce as
    well as what it starts from.
    """

    def __init__(self, hidden: int, dropout: float = 0.0):
        super().__init__()
        # +1 for the edge model's own opinion of the link. Without it the head
        # has to re-derive from pooled states and geometry what a well-trained
        # pair classifier already says, and the strongest signal available goes
        # unused; the scorer's job is to correct that opinion with group context,
        # not to replace it.
        self.net = nn.Sequential(
            nn.Linear(descriptor_width(hidden) * 3 + hidden + 1, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, left: dict[str, torch.Tensor], right: dict[str, torch.Tensor],
                union: dict[str, torch.Tensor], line: torch.Tensor,
                bridge: torch.Tensor) -> torch.Tensor:
        """Raw merge logits.

        ``line`` is the graph summary of each candidate; ``bridge`` is the edge
        model's strongest link probability between the two groups.
        """
        feats = torch.cat([group_descriptor(left), group_descriptor(right),
                           group_descriptor(union), line,
                           bridge.unsqueeze(-1)], dim=-1)
        return self.net(feats).squeeze(-1)


class GlyphEdgeGNN(nn.Module):
    """Node encoder -> stacked edge-aware GAT layers -> edge classifier.

    The classifier also sees a graph-level summary. Four rounds of message
    passing reach only 0.43 em (Latin) to 0.50 em (Japanese) along x -- less
    than one character cell -- so an edge decides on its own neighbourhood
    without ever seeing the glyph next door. Character pitch is exactly the
    signal that tells "ll" from one wide glyph, and it is not observable at
    that range. Pooling the whole graph puts it back within reach for the cost
    of one reduction.
    """

    def __init__(
        self,
        node_dim: int = 13,
        edge_dim: int = 4,
        hidden: int = 128,
        layers: int = 4,
        heads: int = 4,
        dropout: float = 0.1,
        context: bool = True,
        contour_dim: int = 8,
        contours: bool = True,
        relations: bool = True,
        group: bool = False,
    ):
        super().__init__()
        self.hparams = dict(
            node_dim=node_dim, edge_dim=edge_dim, hidden=hidden,
            layers=layers, heads=heads, dropout=dropout, context=context,
            contour_dim=contour_dim, contours=contours, relations=relations,
            group=group,
        )
        # Initial layer aligning heterogeneous node feature dimensions.
        self.encoder = nn.Sequential(
            nn.Linear(node_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
        )
        self.convs = nn.ModuleList(
            EdgeGATLayer(hidden, edge_dim, heads, dropout) for _ in range(layers)
        )
        # One contour round after each point round, so global structure is
        # available while the point representations are still forming rather
        # than bolted on at the end.
        self.use_relations = contours and relations
        self.contour_layers = nn.ModuleList(
            ContourLayer(hidden, contour_dim, heads, dropout, self.use_relations)
            for _ in range(layers)
        ) if contours else None
        # Mean and max carry different things: the mean is the average look of
        # the line, the max reports whether a feature occurs anywhere in it.
        self.context = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
        ) if context else None
        # The classifier sees the geometry of its own contour pair too: after
        # the contour rounds the states know about it, but the raw numbers are
        # free to gather and the decision is made here.
        classifier_in = (hidden * (4 if context else 3) + edge_dim
                         + (CONTOUR_RELATIONS if self.use_relations else 0))
        # Scores a candidate grouping, which is what the decoder needs to
        # accept or refuse a merge; see glyph_gnn.decode.
        self.group_head = GroupHead(hidden, dropout) if group else None
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def pool(self, hx: torch.Tensor, batch: torch.Tensor | None) -> torch.Tensor:
        """Per-graph mean and max, shape ``[num_graphs, 2 * hidden]``.

        ``batch is None`` means a single graph, which is the only case ONNX
        ever traces: inference runs one line at a time, so the export keeps its
        three-input signature and the Rust side needs no changes.
        """
        if batch is None:
            return torch.cat([hx.mean(dim=0, keepdim=True),
                              hx.max(dim=0, keepdim=True).values], dim=-1)
        graphs = int(batch.max()) + 1
        index = batch.unsqueeze(-1).expand(-1, hx.shape[-1])
        total = hx.new_zeros(graphs, hx.shape[-1]).scatter_add_(0, index, hx)
        counts = hx.new_zeros(graphs).scatter_add_(
            0, batch, torch.ones_like(batch, dtype=hx.dtype)
        )
        peak = hx.new_zeros(graphs, hx.shape[-1]).scatter_reduce_(
            0, index, hx, reduce="amax", include_self=False
        )
        return torch.cat([total / counts.clamp(min=1).unsqueeze(-1), peak], dim=-1)

    @staticmethod
    def contour_mask(contour_id: torch.Tensor, contour_attr: torch.Tensor,
                     batch: torch.Tensor | None) -> torch.Tensor | None:
        """Additive attention mask blocking contours of different graphs.

        ``None`` for a single graph, which is both the ONNX path and the reason
        the export needs no adjacency: with one graph every contour may attend
        to every other, so the mask is the identity operation and drops out.
        """
        if batch is None:
            return None
        owner = contour_id.new_zeros(contour_attr.shape[0]).scatter_(0, contour_id, batch)
        same = owner.unsqueeze(-1) == owner.unsqueeze(0)
        return torch.zeros_like(same, dtype=contour_attr.dtype).masked_fill(
            ~same, float("-inf")
        )

    def trunk(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor,
              contour_id: torch.Tensor, contour_attr: torch.Tensor,
              batch: torch.Tensor | None):
        """Runs message passing once. Returns (node states, line summary).

        Split out of :meth:`forward` so the edge classifier and the group head
        share one pass instead of each paying for their own.
        """
        hx = self.encoder(x)
        relations = mask = None
        if self.contour_layers is not None:
            mask = self.contour_mask(contour_id, contour_attr, batch)
        if self.use_relations:
            # Geometry does not change between rounds, so build the all-pairs
            # table once and let every layer read its own bias off it.
            relations = contour_relations(contour_attr.unsqueeze(1),
                                          contour_attr.unsqueeze(0))
        for depth, conv in enumerate(self.convs):
            hx = conv(hx, edge_index, edge_attr)
            if self.contour_layers is not None:
                hx = self.contour_layers[depth](hx, contour_id, contour_attr,
                                                relations, mask)
        summary = self.context(self.pool(hx, batch)) if self.context is not None else None
        return hx, summary

    def edge_logits(self, hx: torch.Tensor, summary: torch.Tensor | None,
                    edge_index: torch.Tensor, edge_attr: torch.Tensor,
                    contour_id: torch.Tensor, contour_attr: torch.Tensor,
                    batch: torch.Tensor | None) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        hs, hd = hx[src], hx[dst]
        parts = [hs, hd, (hs - hd).abs(), edge_attr]
        if self.use_relations:
            parts.append(contour_relations(contour_attr[contour_id[src]],
                                           contour_attr[contour_id[dst]]))
        if summary is not None:
            parts.append(summary[batch[src]] if batch is not None
                         else summary.expand(src.shape[0], -1))
        return self.classifier(torch.cat(parts, dim=-1)).squeeze(-1)

    def contour_states(self, hx: torch.Tensor, contour_id: torch.Tensor,
                       contour_attr: torch.Tensor) -> torch.Tensor:
        """One state per contour, pooled from the points it owns."""
        return scatter_mean(hx, contour_id, contour_attr.shape[0])

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor,
                contour_id: torch.Tensor, contour_attr: torch.Tensor,
                batch: torch.Tensor | None = None) -> torch.Tensor:
        """Returns raw edge logits, shape [num_edges]."""
        hx, summary = self.trunk(x, edge_index, edge_attr, contour_id, contour_attr, batch)
        return self.edge_logits(hx, summary, edge_index, edge_attr,
                                contour_id, contour_attr, batch)


MODEL_HPARAM_KEYS = frozenset(
    {"node_dim", "edge_dim", "hidden", "layers", "heads", "dropout", "context",
     "contour_dim", "contours", "relations", "group"}
)
GRAPH_HPARAM_KEYS = frozenset({"knn", "radius", "contour_bridge", "stack_bridge"})


#: Components added after checkpoints already existed. A checkpoint from before
#: one of them simply has no key for it, and that absence has to read as "off"
#: rather than as the current default -- otherwise loading it fails on weights
#: the run never had.
ADDED_LATER = {"context": False, "contours": False, "relations": False,
               "group": False}


def model_hparams(hparams: dict) -> dict:
    """Return only keys accepted by :class:`GlyphEdgeGNN`."""
    kwargs = {k: hparams[k] for k in MODEL_HPARAM_KEYS if k in hparams}
    for key, absent in ADDED_LATER.items():
        kwargs.setdefault(key, absent)
    return kwargs


def graph_hparams(hparams: dict) -> dict:
    """Graph construction settings stored alongside model hparams."""
    return {k: hparams[k] for k in GRAPH_HPARAM_KEYS if k in hparams}
