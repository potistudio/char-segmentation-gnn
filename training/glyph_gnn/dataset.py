"""Shard loading into PyTorch Geometric graph data.

Two backends share the interface the training loop needs (``__len__``,
``__getitem__``, ``edge_counts``, ``stats``): the MessagePack loader, which
decodes everything into RAM, and :class:`PackedGlyphDataset`, which
memory-maps a store produced by ``glyph_gnn.pack``. Use the packed store
once the corpus outgrows RAM.
"""

from __future__ import annotations

import copy
import random
from collections.abc import Iterator, Sequence
from pathlib import Path

import msgpack
import numpy as np
import torch
from torch.utils.data import Sampler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm

from .pack import INDEX_NAME, STREAMS


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


def graph_stats(graphs) -> dict:
    """Summarizes a dataset without materializing it when it is packed."""
    if hasattr(graphs, "stats"):
        return graphs.stats()
    nodes = sum(int(g.num_nodes) for g in graphs)
    edges = sum(int(g.y.numel()) for g in graphs)
    positives = sum(int(g.y.sum()) for g in graphs)
    return {"graphs": len(graphs), "nodes": nodes, "edges": edges,
            "pos_ratio": positives / edges if edges else 0.0,
            "fonts": len({g.font for g in graphs})}


def _starts(sizes: np.ndarray) -> np.ndarray:
    offsets = np.zeros(len(sizes) + 1, dtype=np.int64)
    np.cumsum(sizes, dtype=np.int64, out=offsets[1:])
    return offsets


class PackedGlyphDataset(torch.utils.data.Dataset):
    """Memory-mapped view over a store written by ``glyph_gnn.pack``.

    Only the index (a few arrays of per-graph counts) lives in RAM; sample
    payloads are paged in on demand, so resident memory is bounded by the
    OS page cache rather than by the corpus size. Sample offsets follow from
    the cumulative node and edge counts, so no offset table is stored.
    """

    def __init__(self, root: str | Path, ids: Sequence[int] | None = None):
        self.root = Path(root)
        index = np.load(self.root / INDEX_NAME, allow_pickle=False)
        self.node_dim, self.edge_dim = (int(v) for v in index["dims"])
        self.num_nodes = index["num_nodes"].astype(np.int64)
        self.num_edges = index["num_edges"].astype(np.int64)
        self.positives = index["positives"].astype(np.int64)
        self.fonts = index["fonts"][index["font_id"]]
        self.texts = index["texts"]
        self._offsets = {
            "node_features": _starts(self.num_nodes * self.node_dim),
            "edge_index": _starts(self.num_edges * 2),
            "edge_features": _starts(self.num_edges * self.edge_dim),
            "edge_labels": _starts(self.num_edges),
        }
        self.ids = (np.arange(len(self.num_nodes), dtype=np.int64) if ids is None
                    else np.asarray(ids, dtype=np.int64))
        self._maps: dict[str, np.memmap] | None = None

    def subset(self, ids: Sequence[int]) -> PackedGlyphDataset:
        """Returns a view over ``ids`` sharing this instance's index arrays."""
        clone = copy.copy(self)
        clone.ids = np.asarray(ids, dtype=np.int64)
        clone._maps = None
        return clone

    def __getstate__(self) -> dict:
        # Memory maps do not survive pickling to DataLoader workers; each
        # process reopens them on first access.
        return {**self.__dict__, "_maps": None}

    def _mapped(self) -> dict[str, np.memmap]:
        if self._maps is None:
            self._maps = {
                key: np.memmap(self.root / name, dtype=dtype, mode="r")
                for key, (name, dtype) in STREAMS.items()
            }
        return self._maps

    def _read(self, key: str, gid: int, count: int) -> np.ndarray:
        start = self._offsets[key][gid]
        return self._mapped()[key][start:start + count]

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int) -> Data:
        gid = int(self.ids[i])
        n, e = int(self.num_nodes[gid]), int(self.num_edges[gid])
        data = Data(
            x=torch.from_numpy(
                self._read("node_features", gid, n * self.node_dim).copy()
            ).view(n, self.node_dim),
            # Stored narrow (u32/u8) to match the Rust types; PyG needs
            # int64 indices and the loss needs float targets.
            edge_index=torch.from_numpy(
                self._read("edge_index", gid, 2 * e).astype(np.int64)
            ).view(2, e),
            edge_attr=torch.from_numpy(
                self._read("edge_features", gid, e * self.edge_dim).copy()
            ).view(e, self.edge_dim),
            y=torch.from_numpy(self._read("edge_labels", gid, e).astype(np.float32)),
        )
        data.font = str(self.fonts[gid])
        data.text = str(self.texts[gid])
        return data

    @property
    def edge_counts(self) -> np.ndarray:
        return self.num_edges[self.ids]

    def stats(self) -> dict:
        edges = int(self.num_edges[self.ids].sum())
        return {
            "graphs": len(self.ids),
            "nodes": int(self.num_nodes[self.ids].sum()),
            "edges": edges,
            "pos_ratio": int(self.positives[self.ids].sum()) / edges if edges else 0.0,
            "fonts": len(np.unique(self.fonts[self.ids])),
        }


def load_packed(
    root: str | Path,
    val_frac: float = 0.1,
    split_by_font: bool = True,
    seed: int = 0,
) -> tuple[PackedGlyphDataset, PackedGlyphDataset]:
    """Opens a packed store and splits it the same way as :func:`load_dataset`."""
    full = PackedGlyphDataset(root)
    rng = random.Random(seed)
    if split_by_font:
        names = sorted(set(full.fonts.tolist()))
        rng.shuffle(names)
        val_fonts = set(names[:max(1, int(len(names) * val_frac))])
        mask = np.isin(full.fonts, list(val_fonts))
        train_ids, val_ids = np.nonzero(~mask)[0], np.nonzero(mask)[0]
        if not len(train_ids):
            train_ids, val_ids = val_ids, train_ids
    else:
        order = list(range(len(full)))
        rng.shuffle(order)
        n_val = int(len(order) * val_frac)
        val_ids, train_ids = order[:n_val], order[n_val:]
    return full.subset(train_ids), full.subset(val_ids)


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
        self.edge_counts = np.asarray(edge_counts, dtype=np.int64).tolist()
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
    graphs,
    max_edges: int,
    max_graphs: int,
    shuffle: bool = False,
    seed: int = 0,
    num_workers: int = 0,
) -> DataLoader:
    """Builds a loader whose batches stay within ``max_edges``."""
    counts = getattr(graphs, "edge_counts", None)
    if counts is None:
        counts = [int(g.y.numel()) for g in graphs]
    sampler = EdgeBudgetBatchSampler(
        counts,
        max_edges=max_edges,
        max_graphs=max_graphs,
        shuffle=shuffle,
        seed=seed,
    )
    return DataLoader(graphs, batch_sampler=sampler, num_workers=num_workers,
                      persistent_workers=num_workers > 0)
