"""Repacks MessagePack shards into a memory-mappable tensor store.

The shards encode every float as a MessagePack array element, so decoding
one costs ~8s and materializes the whole sample as Python lists. That
forces the entire dataset into RAM to train at a reasonable speed, and a
23 GB corpus expands to ~28 GiB of tensors.

This writes flat binary streams that ``PackedGlyphDataset`` memory-maps, so
RAM use stops scaling with the dataset and decoding disappears. Three of the
four edge features are redundant on disk and are dropped, which shrinks the
store by 40% (20.09 -> 12.23 GiB for ja-train) without losing anything:

* ``dist`` is recomputed from ``dx``/``dy`` with the same f32 operations the
  Rust generator uses, so it round-trips bit for bit.
* ``same_contour`` is 0/1 and rides in bit 1 of the label byte.
* Node ids fit in u16 (the widest glyph graph has ~3.2k nodes), which the
  packer verifies per sample.

``node_ids`` carries the per-node contour and character ids as an interleaved
u16 pair. They are what the shipping decision is actually made over --
``group_characters`` collapses point edges to contour pairs -- so without them
training can only score the point-edge proxy. At 4 bytes per node they cost
~187 MiB on ja-train (+1.5%).

Size matters beyond disk: at 12.23 GiB the whole store stays in the OS page
cache on a 32 GiB host, so training reads it from disk once instead of
seeking per batch.

Usage:
    python -m glyph_gnn.pack --data ../dataset/ja-train --out ../dataset/ja-train-packed
"""

from __future__ import annotations

import argparse
import time
from contextlib import ExitStack
from pathlib import Path

import msgpack
import numpy as np
from tqdm.auto import tqdm

INDEX_NAME = "index.npz"
FORMAT_VERSION = 3
# Node/edge feature layout produced by crates/glyph-core/src/graph.rs.
EDGE_DIM = 4
DX, DY, DIST, SAME_CONTOUR = range(EDGE_DIM)
LABEL_BIT, SAME_CONTOUR_BIT = 1, 2
# Columns of the interleaved node_ids stream.
CONTOUR_ID, CHAR_ID = 0, 1
STREAMS = {
    "x": ("x.f32", np.float32),
    "edge_index": ("edge_index.u16", np.uint16),
    "edge_delta": ("edge_delta.f32", np.float32),
    "flags": ("flags.u8", np.uint8),
    "node_ids": ("node_ids.u16", np.uint16),
}


def is_packed(path: str | Path) -> bool:
    return (Path(path) / INDEX_NAME).is_file()


def edge_distance(delta: np.ndarray) -> np.ndarray:
    """Recomputes the dropped ``dist`` column from the ``dx``/``dy`` pairs."""
    dx, dy = delta[:, DX], delta[:, DY]
    return np.sqrt(dx * dx + dy * dy)


def encode(sample: dict) -> tuple[dict[str, np.ndarray], float]:
    """Converts one decoded sample into its on-disk streams.

    Also returns how far the recomputed ``dist`` drifts from the stored one,
    which the caller reports so a lossy round trip cannot pass unnoticed.
    """
    n, node_dim, edge_dim = sample["num_nodes"], sample["node_dim"], sample["edge_dim"]
    if edge_dim != EDGE_DIM:
        raise ValueError(f"expected {EDGE_DIM} edge features, got {edge_dim}")
    if n > np.iinfo(np.uint16).max:
        raise ValueError(f"{n} nodes exceeds the u16 node id range")

    x = np.asarray(sample["node_features"], dtype=np.float32)
    edge_index = np.asarray(sample["edge_index"], dtype=np.uint32)
    features = np.asarray(sample["edge_features"], dtype=np.float32).reshape(-1, EDGE_DIM)
    labels = np.asarray(sample["edge_labels"], dtype=np.uint8)
    contour_ids = np.asarray(sample["node_contour_ids"], dtype=np.int64)
    char_ids = np.asarray(sample["node_char_ids"], dtype=np.int64)

    e = labels.size
    if x.size != n * node_dim:
        raise ValueError("node feature size mismatch")
    if edge_index.size != 2 * e:
        raise ValueError("edge index size mismatch")
    if features.shape[0] != e:
        raise ValueError("edge feature size mismatch")
    if edge_index.size and edge_index.max() >= n:
        raise ValueError("edge index out of range")
    if contour_ids.size != n or char_ids.size != n:
        raise ValueError("node id size mismatch")
    # UNKNOWN_CHAR (u32::MAX) only appears at inference time; a shard carrying
    # it is not trainable data, and it would silently truncate to u16.
    widest = max(int(contour_ids.max(initial=0)), int(char_ids.max(initial=0)))
    if widest > np.iinfo(np.uint16).max:
        raise ValueError(f"node id {widest} exceeds the u16 range (unlabeled shard?)")

    same = features[:, SAME_CONTOUR]
    if not np.isin(same, (0.0, 1.0)).all():
        raise ValueError("same_contour is not binary; it cannot be packed into a bit")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("edge labels are not binary")

    delta = np.ascontiguousarray(features[:, [DX, DY]])
    drift = float(np.abs(edge_distance(delta) - features[:, DIST]).max()) if e else 0.0
    streams = {
        "x": x,
        "edge_index": edge_index.astype(np.uint16),
        "edge_delta": delta.ravel(),
        "flags": (labels * LABEL_BIT
                  + same.astype(np.uint8) * SAME_CONTOUR_BIT).astype(np.uint8, copy=False),
        "node_ids": np.stack([contour_ids, char_ids], axis=1).astype(np.uint16).ravel(),
    }
    return streams, drift


def pack_dataset(src: str | Path, dst: str | Path, limit_shards: int | None = None,
                 progress: bool = True) -> dict:
    """Converts every shard under ``src`` into the packed store at ``dst``."""
    shards = sorted(Path(src).glob("*.msgpack"))
    if not shards:
        raise FileNotFoundError(f"no .msgpack shards under {src}")
    if limit_shards:
        shards = shards[:limit_shards]

    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)

    num_nodes: list[int] = []
    num_edges: list[int] = []
    positives: list[int] = []
    fonts: list[str] = []
    texts: list[str] = []
    node_dim: int | None = None
    drift = 0.0

    with ExitStack() as stack:
        files = {
            key: stack.enter_context(open(dst / name, "wb"))
            for key, (name, _) in STREAMS.items()
        }
        bar = stack.enter_context(
            tqdm(shards, desc="packing", unit="shard", disable=None if progress else True)
        )
        for shard in bar:
            with open(shard, "rb") as f:
                samples = msgpack.unpackb(f.read(), raw=False)
            for s in samples:
                try:
                    streams, sample_drift = encode(s)
                except ValueError as exc:
                    raise ValueError(f"{shard.name}: {exc}") from exc
                if node_dim is None:
                    node_dim = s["node_dim"]
                elif node_dim != s["node_dim"]:
                    raise ValueError(f"{shard.name}: inconsistent node dim {node_dim}")

                for key, array in streams.items():
                    files[key].write(array.tobytes())
                drift = max(drift, sample_drift)
                num_nodes.append(s["num_nodes"])
                num_edges.append(streams["flags"].size)
                positives.append(int((streams["flags"] & LABEL_BIT).sum()))
                fonts.append(s.get("font", ""))
                texts.append(s.get("text", ""))
            bar.set_postfix(graphs=len(num_nodes), refresh=False)

    names, font_id = np.unique(np.array(fonts), return_inverse=True)
    np.savez(
        dst / INDEX_NAME,
        version=np.array(FORMAT_VERSION, dtype=np.uint32),
        dims=np.array((node_dim, EDGE_DIM), dtype=np.uint32),
        num_nodes=np.array(num_nodes, dtype=np.uint32),
        num_edges=np.array(num_edges, dtype=np.uint32),
        positives=np.array(positives, dtype=np.uint32),
        fonts=names,
        font_id=font_id.astype(np.uint32),
        texts=np.array(texts),
    )
    return {
        "graphs": len(num_nodes),
        "nodes": int(np.sum(num_nodes, dtype=np.int64)),
        "edges": int(np.sum(num_edges, dtype=np.int64)),
        "fonts": len(names),
        "bytes": sum((dst / name).stat().st_size for name, _ in STREAMS.values()),
        "dist_drift": drift,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True, help="directory of .msgpack shards")
    ap.add_argument("--out", required=True, help="destination directory for the packed store")
    ap.add_argument("--limit-shards", type=int, default=None)
    ap.add_argument("--no-progress", action="store_true")
    args = ap.parse_args()

    started = time.time()
    info = pack_dataset(args.data, args.out, args.limit_shards, not args.no_progress)
    print(f"packed {info['graphs']:,} graphs | {info['nodes']:,} nodes"
          f" | {info['edges']:,} edges | {info['fonts']} fonts")
    print(f"{info['bytes'] / 2**30:.2f} GiB -> {Path(args.out)} in {time.time() - started:.0f}s")
    print(f"max recomputed-vs-stored dist error: {info['dist_drift']:g}")


if __name__ == "__main__":
    main()
