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


class GlyphEdgeGNN(nn.Module):
    """Node encoder -> stacked edge-aware GAT layers -> edge classifier."""

    def __init__(
        self,
        node_dim: int = 13,
        edge_dim: int = 4,
        hidden: int = 128,
        layers: int = 4,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hparams = dict(
            node_dim=node_dim, edge_dim=edge_dim, hidden=hidden,
            layers=layers, heads=heads, dropout=dropout,
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
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 3 + edge_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """Returns raw edge logits, shape [num_edges]."""
        hx = self.encoder(x)
        for conv in self.convs:
            hx = conv(hx, edge_index, edge_attr)
        src, dst = edge_index[0], edge_index[1]
        hs, hd = hx[src], hx[dst]
        feats = torch.cat([hs, hd, (hs - hd).abs(), edge_attr], dim=-1)
        return self.classifier(feats).squeeze(-1)
