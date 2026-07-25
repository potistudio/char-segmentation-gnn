"""MessagePack shard loading into PyTorch Geometric graph data."""

from __future__ import annotations

import random
from collections.abc import Iterator, Sequence
from pathlib import Path

import msgpack
import torch
from torch.utils.data import Sampler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm


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
    progress: bool = True,
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
    bar = tqdm(shards, desc="loading shards", unit="shard", disable=None if progress else True)
    for p in bar:
        graphs.extend(load_shard(p))
        bar.set_postfix(graphs=len(graphs), refresh=False)
    bar.close()

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


class EdgeBudgetBatchSampler(Sampler[list[int]]):
    """Groups graphs into batches bounded by a total edge count.

    Every activation in the GAT layers is shaped ``[num_edges, hidden]``, so
    peak memory tracks the edge count of a batch, not the graph count. Glyph
    graphs range from ~1k to ~24k edges, so a fixed ``batch_size`` makes peak
    memory swing by more than an order of magnitude and the largest batches
    blow up the GPU. Capping edges per batch keeps it flat.
    """

    def __init__(
        self,
        edge_counts: Sequence[int],
        max_edges: int,
        max_graphs: int,
        shuffle: bool = False,
        seed: int = 0,
    ):
        self.edge_counts = list(edge_counts)
        self.max_edges = max_edges
        self.max_graphs = max_graphs
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.batches = self._pack(range(len(self.edge_counts)))

    def _pack(self, order) -> list[list[int]]:
        batches: list[list[int]] = []
        current: list[int] = []
        edges = 0
        for i in order:
            count = self.edge_counts[i]
            full = len(current) >= self.max_graphs or edges + count > self.max_edges
            if current and full:
                batches.append(current)
                current, edges = [], 0
            current.append(i)
            edges += count
        if current:
            batches.append(current)
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        if self.shuffle:
            rng = random.Random(self.seed + self.epoch)
            order = list(range(len(self.edge_counts)))
            rng.shuffle(order)
            self.batches = self._pack(order)
            rng.shuffle(self.batches)
            self.epoch += 1
        yield from self.batches

    def __len__(self) -> int:
        # Reshuffling repacks the batches, so this can drift by a batch or two
        # between epochs; it is only used for progress reporting.
        return len(self.batches)


def make_loader(
    graphs: Sequence[Data],
    max_edges: int,
    max_graphs: int,
    shuffle: bool = False,
    seed: int = 0,
) -> DataLoader:
    """Builds a loader whose batches stay within ``max_edges``."""
    sampler = EdgeBudgetBatchSampler(
        [int(g.y.numel()) for g in graphs],
        max_edges=max_edges,
        max_graphs=max_graphs,
        shuffle=shuffle,
        seed=seed,
    )
    return DataLoader(graphs, batch_sampler=sampler)
