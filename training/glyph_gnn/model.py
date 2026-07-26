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

    def __init__(self, dim: int, contour_dim: int, heads: int = 4, dropout: float = 0.0):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.head_dim = dim // heads
        # The supernode starts from its own geometry (centroid, bbox, arc
        # length, signed area) plus the mean of the points it owns.
        self.embed = nn.Linear(contour_dim, dim)
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.norm_contour = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = dropout

    def forward(self, hx: torch.Tensor, contour_id: torch.Tensor,
                contour_attr: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        contours = contour_attr.shape[0]
        h, hd = self.heads, self.head_dim
        z = self.norm_contour(
            scatter_mean(hx, contour_id, contours) + self.embed(contour_attr)
        )

        def heads_of(linear: nn.Module) -> torch.Tensor:
            return linear(z).view(contours, h, hd).transpose(0, 1)  # [H, C, D]

        scores = heads_of(self.query) @ heads_of(self.key).transpose(-2, -1)
        scores = scores * (self.head_dim ** -0.5)
        if mask is not None:
            scores = scores + mask
        weights = F.dropout(scores.softmax(dim=-1), p=self.dropout, training=self.training)
        mixed = (weights @ heads_of(self.value)).transpose(0, 1).reshape(contours, h * hd)

        z = z + self.proj(mixed)
        return self.norm(hx + self.out(z)[contour_id])


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
    ):
        super().__init__()
        self.hparams = dict(
            node_dim=node_dim, edge_dim=edge_dim, hidden=hidden,
            layers=layers, heads=heads, dropout=dropout, context=context,
            contour_dim=contour_dim, contours=contours,
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
        self.contour_layers = nn.ModuleList(
            ContourLayer(hidden, contour_dim, heads, dropout) for _ in range(layers)
        ) if contours else None
        # Mean and max carry different things: the mean is the average look of
        # the line, the max reports whether a feature occurs anywhere in it.
        self.context = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
        ) if context else None
        self.classifier = nn.Sequential(
            nn.Linear(hidden * (4 if context else 3) + edge_dim, hidden),
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

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor,
                contour_id: torch.Tensor, contour_attr: torch.Tensor,
                batch: torch.Tensor | None = None) -> torch.Tensor:
        """Returns raw edge logits, shape [num_edges]."""
        hx = self.encoder(x)
        mask = (self.contour_mask(contour_id, contour_attr, batch)
                if self.contour_layers is not None else None)
        for depth, conv in enumerate(self.convs):
            hx = conv(hx, edge_index, edge_attr)
            if self.contour_layers is not None:
                hx = self.contour_layers[depth](hx, contour_id, contour_attr, mask)
        src, dst = edge_index[0], edge_index[1]
        hs, hd = hx[src], hx[dst]
        parts = [hs, hd, (hs - hd).abs(), edge_attr]
        if self.context is not None:
            summary = self.context(self.pool(hx, batch))
            parts.append(summary[batch[src]] if batch is not None
                         else summary.expand(src.shape[0], -1))
        return self.classifier(torch.cat(parts, dim=-1)).squeeze(-1)


MODEL_HPARAM_KEYS = frozenset(
    {"node_dim", "edge_dim", "hidden", "layers", "heads", "dropout", "context",
     "contour_dim", "contours"}
)
GRAPH_HPARAM_KEYS = frozenset({"knn", "radius", "contour_bridge"})


#: Components added after checkpoints already existed. A checkpoint from before
#: one of them simply has no key for it, and that absence has to read as "off"
#: rather than as the current default -- otherwise loading it fails on weights
#: the run never had.
ADDED_LATER = {"context": False, "contours": False}


def model_hparams(hparams: dict) -> dict:
    """Return only keys accepted by :class:`GlyphEdgeGNN`."""
    kwargs = {k: hparams[k] for k in MODEL_HPARAM_KEYS if k in hparams}
    for key, absent in ADDED_LATER.items():
        kwargs.setdefault(key, absent)
    return kwargs


def graph_hparams(hparams: dict) -> dict:
    """Graph construction settings stored alongside model hparams."""
    return {k: hparams[k] for k in GRAPH_HPARAM_KEYS if k in hparams}
