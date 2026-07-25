"""Throwaway benchmark: candidate speedups for the training step."""

from __future__ import annotations

import sys
import time

import torch
import torch.nn.functional as F

from glyph_gnn.dataset import PackedGlyphDataset, make_loader
from glyph_gnn.loss import focal_loss
from glyph_gnn.model import EdgeGATLayer, GlyphEdgeGNN

STORE = "../dataset/ja-train-packed-v2"
GRAPHS = 4000
WARMUP, STEPS = 5, 40

baseline_forward = EdgeGATLayer.forward


def forward_index_add(self, x, edge_index, edge_attr):
    """Same math, but scatters with 1-D indices instead of expanded ones."""
    n = x.shape[0]
    src, dst = edge_index[0], edge_index[1]
    h, hd = self.heads, self.head_dim

    h_src = self.lin_src(x).view(-1, h, hd)[src]
    h_dst = self.lin_dst(x).view(-1, h, hd)[dst]
    h_edge = self.lin_edge(edge_attr).view(-1, h, hd)

    z = F.leaky_relu(h_src + h_dst + h_edge, 0.2)
    logits = (z * self.att).sum(dim=-1)
    logits = logits - logits.max()
    num = torch.exp(logits)
    denom = x.new_zeros(n, h).index_add_(0, dst, num)
    alpha = num / (denom[dst] + 1e-9)
    alpha = F.dropout(alpha, p=self.dropout, training=self.training)

    msg = (h_src * alpha.unsqueeze(-1)).reshape(-1, h * hd)
    agg = x.new_zeros(n, h * hd).index_add_(0, dst, msg)
    return self.norm(x + self.out(agg))


def run(label, *, budget=200_000, tf32=False, amp=None, index_add=False, compile=False,
        workers=0, max_graphs=64):
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    EdgeGATLayer.forward = forward_index_add if index_add else baseline_forward
    global WARMUP
    WARMUP = 25 if compile else 5

    torch.manual_seed(0)
    device = torch.device("cuda")
    data = PackedGlyphDataset(STORE).subset(range(GRAPHS))
    loader = make_loader(data, budget, max_graphs, shuffle=True, seed=0, num_workers=workers)
    model = GlyphEdgeGNN().to(device)
    if compile:
        model = torch.compile(model, dynamic=True)
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4)
    model.train()

    torch.cuda.reset_peak_memory_stats(device)
    graphs = edges = steps = 0
    started = None
    batches = iter(loader)
    while steps < WARMUP + STEPS:
        try:
            batch = next(batches)
        except StopIteration:
            batches = iter(loader)
            continue
        batch = batch.to(device)
        optim.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
            out = model(batch.x, batch.edge_index, batch.edge_attr)
        loss = focal_loss(out.float(), batch.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        steps += 1
        if steps == WARMUP:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
            started = time.time()
        elif steps > WARMUP:
            graphs += batch.num_graphs
            edges += int(batch.y.numel())
    torch.cuda.synchronize()
    elapsed = time.time() - started
    peak = torch.cuda.max_memory_allocated(device) / 2**30
    print(f"{label:34s} {graphs / elapsed:7.0f} graphs/s |"
          f" {edges / elapsed / 1e6:6.2f} M edges/s | peak {peak:5.2f} GiB |"
          f" {graphs / (steps - WARMUP):5.1f} graphs/batch")
    return graphs / elapsed


configs = {
    "baseline": {},
    "tf32": {"tf32": True},
    "bf16 autocast": {"amp": torch.bfloat16},
    "index_add": {"index_add": True},
    "budget 100k": {"budget": 100_000},
    "budget 400k": {"budget": 400_000, "max_graphs": 256},
    "budget 600k": {"budget": 600_000, "max_graphs": 256},
    "budget 1000k": {"budget": 1_000_000, "max_graphs": 256},
    "workers 4": {"workers": 4},
    "compile": {"compile": True},
    "bf16 + compile": {"amp": torch.bfloat16, "compile": True},
    "bf16 + budget 600k": {"amp": torch.bfloat16, "budget": 600_000, "max_graphs": 256},
}
if __name__ == "__main__":
    for name in sys.argv[1:] or list(configs):
        run(name, **configs[name])
