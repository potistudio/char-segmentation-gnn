"""MessagePack shard loading into PyTorch Geometric graph data."""

from __future__ import annotations

import random
from pathlib import Path

import msgpack
import torch
from torch_geometric.data import Data


def load_shard(path: Path) -> list[Data]:
    """Decodes one Rust-generated MessagePack shard into PyG ``Data`` objects."""
    with open(path, "rb") as f:
        samples = msgpack.unpackb(f.read(), raw=False)

    out = []
    for s in samples:
        n = s["num_nodes"]
        node_dim = s["node_dim"]
        edge_dim = s["edge_dim"]
        x = torch.tensor(s["node_features"], dtype=torch.float32).view(n, node_dim)
        edge_index = torch.tensor(s["edge_index"], dtype=torch.long).view(2, -1)
        edge_attr = torch.tensor(s["edge_features"], dtype=torch.float32).view(-1, edge_dim)
        y = torch.tensor(s["edge_labels"], dtype=torch.float32)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        data.font = s.get("font", "")
        data.text = s.get("text", "")
        out.append(data)
    return out


def load_dataset(
    data_dir: str | Path,
    val_frac: float = 0.1,
    split_by_font: bool = True,
    seed: int = 0,
    limit_shards: int | None = None,
) -> tuple[list[Data], list[Data]]:
    """Loads all shards and returns (train, val).

    With ``split_by_font`` the validation set holds out entire fonts, which
    measures generalization to unseen fonts as required by Phase 4.
    """
    shards = sorted(Path(data_dir).glob("*.msgpack"))
    if not shards:
        raise FileNotFoundError(f"no .msgpack shards under {data_dir}")
    if limit_shards:
        shards = shards[:limit_shards]

    graphs: list[Data] = []
    for p in shards:
        graphs.extend(load_shard(p))

    rng = random.Random(seed)
    if split_by_font:
        fonts = sorted({g.font for g in graphs})
        rng.shuffle(fonts)
        n_val = max(1, int(len(fonts) * val_frac))
        val_fonts = set(fonts[:n_val])
        train = [g for g in graphs if g.font not in val_fonts]
        val = [g for g in graphs if g.font in val_fonts]
        # Degenerate case: single font in the corpus.
        if not train:
            train, val = val, []
    else:
        rng.shuffle(graphs)
        n_val = int(len(graphs) * val_frac)
        val, train = graphs[:n_val], graphs[n_val:]
    return train, val
