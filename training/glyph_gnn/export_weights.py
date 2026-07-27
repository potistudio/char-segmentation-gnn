"""Dumps a checkpoint's tensors for the native Rust runtime (Tier C).

ONNX buys portability the CPU path does not use: onnxruntime spends its time
walking a generic graph, materialising an ``[E, hidden]`` tensor per operator
and scattering into node rows. The Rust side already builds the graph, so it can
hold the edge list as CSR and turn every aggregation into a contiguous read --
no random writes, no atomics, and a loop that splits over nodes cleanly. What it
needs from training is only the weights.

Format, little-endian throughout::

    b"GNNW"        magic
    u32            format version
    u32            header length in bytes
    <header>       UTF-8 JSON: hparams, and one entry per tensor
    <data>         f32, row-major, in the header's order

Usage:
    python -m glyph_gnn.export_weights --checkpoint checkpoints/run/best.pt --out ../model.gnnw
"""

from __future__ import annotations

import argparse
import json
import struct

import torch

MAGIC = b"GNNW"
FORMAT_VERSION = 1


def dump(state_dict: dict[str, torch.Tensor], hparams: dict) -> bytes:
    entries, blobs, offset = [], [], 0
    for name, tensor in state_dict.items():
        flat = tensor.detach().to(torch.float32).contiguous().reshape(-1)
        entries.append({"name": name, "shape": list(tensor.shape), "offset": offset})
        blobs.append(flat.numpy().tobytes())
        offset += flat.numel() * 4
    header = json.dumps({"hparams": hparams, "tensors": entries},
                        separators=(",", ":")).encode()
    return b"".join([MAGIC, struct.pack("<II", FORMAT_VERSION, len(header)), header, *blobs])


def main() -> None:
    ap = argparse.ArgumentParser(description="Dump model weights for the Rust runtime")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="model.gnnw")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    # json cannot hold a tensor, and nothing downstream reads one out of here.
    hparams = {k: v for k, v in ckpt["hparams"].items()
               if isinstance(v, (int, float, str, bool, type(None)))}
    payload = dump(ckpt["state_dict"], hparams)
    with open(args.out, "wb") as f:
        f.write(payload)
    print(f"exported -> {args.out} ({len(payload) / 2**20:.2f} MiB, "
          f"{len(ckpt['state_dict'])} tensors)")
    print(f"hparams: {hparams}")


if __name__ == "__main__":
    main()
