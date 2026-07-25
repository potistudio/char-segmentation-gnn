"""Repacks MessagePack shards into a memory-mappable tensor store.

The shards encode every float as a MessagePack array element, so decoding
one costs ~8s and materializes the whole sample as Python lists. That
forces the entire dataset into RAM to train at a reasonable speed, and a
23 GB corpus expands to ~28 GiB of tensors.

This writes the same data as flat binary files whose element types match
the Rust ``GraphSample`` fields exactly, so the conversion is lossless.
``PackedGlyphDataset`` then memory-maps them: RAM use stops scaling with
the dataset, decoding disappears, and per-sample random access keeps
global shuffling and the edge-budget sampler working.

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
# Element types mirror crates/glyph-core/src/graph.rs::GraphSample.
STREAMS = {
    "node_features": ("x.f32", np.float32),
    "edge_index": ("edge_index.u32", np.uint32),
    "edge_features": ("edge_attr.f32", np.float32),
    "edge_labels": ("y.u8", np.uint8),
}


def is_packed(path: str | Path) -> bool:
    return (Path(path) / INDEX_NAME).is_file()


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
    dims: tuple[int, int] | None = None

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
                n, node_dim, edge_dim = s["num_nodes"], s["node_dim"], s["edge_dim"]
                arrays = {
                    key: np.asarray(s[key], dtype=dtype)
                    for key, (_, dtype) in STREAMS.items()
                }
                e = arrays["edge_labels"].size
                if arrays["node_features"].size != n * node_dim:
                    raise ValueError(f"{shard.name}: node feature size mismatch")
                if arrays["edge_index"].size != 2 * e:
                    raise ValueError(f"{shard.name}: edge index size mismatch")
                if arrays["edge_features"].size != e * edge_dim:
                    raise ValueError(f"{shard.name}: edge feature size mismatch")
                if dims is None:
                    dims = (node_dim, edge_dim)
                elif dims != (node_dim, edge_dim):
                    raise ValueError(f"{shard.name}: inconsistent feature dims {dims}")

                for key, array in arrays.items():
                    files[key].write(array.tobytes())
                num_nodes.append(n)
                num_edges.append(e)
                positives.append(int(arrays["edge_labels"].sum()))
                fonts.append(s.get("font", ""))
                texts.append(s.get("text", ""))
            bar.set_postfix(graphs=len(num_nodes), refresh=False)

    names, font_id = np.unique(np.array(fonts), return_inverse=True)
    np.savez(
        dst / INDEX_NAME,
        dims=np.array(dims, dtype=np.uint32),
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


if __name__ == "__main__":
    main()
