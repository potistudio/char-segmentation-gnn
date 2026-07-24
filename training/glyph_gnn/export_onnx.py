"""ONNX export (Phase 2, step 4).

Exports the trained model with dynamic node/edge axes and verifies the
result against PyTorch using onnxruntime.

Usage:
    python -m glyph_gnn.export_onnx --checkpoint checkpoints/run/best.pt --out ../model.onnx
"""

from __future__ import annotations

import argparse

import torch

from .model import GlyphEdgeGNN


class InferenceWrapper(torch.nn.Module):
    """Adds the sigmoid so the Rust side receives probabilities directly."""

    def __init__(self, model: GlyphEdgeGNN):
        super().__init__()
        self.model = model

    def forward(self, node_features, edge_index, edge_features):
        return torch.sigmoid(self.model(node_features, edge_index, edge_features))


def make_dummy_inputs(node_dim: int, edge_dim: int, n: int = 50, e: int = 200):
    x = torch.randn(n, node_dim)
    edge_index = torch.randint(0, n, (2, e), dtype=torch.long)
    edge_attr = torch.randn(e, edge_dim)
    return x, edge_index, edge_attr


def main() -> None:
    ap = argparse.ArgumentParser(description="Export trained GNN to ONNX")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="model.onnx")
    ap.add_argument("--opset", type=int, default=18)
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = GlyphEdgeGNN(**ckpt["hparams"])
    model.load_state_dict(ckpt["state_dict"])
    wrapper = InferenceWrapper(model)
    wrapper.eval()

    node_dim = ckpt["hparams"]["node_dim"]
    edge_dim = ckpt["hparams"]["edge_dim"]
    dummy = make_dummy_inputs(node_dim, edge_dim)

    torch.onnx.export(
        wrapper,
        dummy,
        args.out,
        input_names=["node_features", "edge_index", "edge_features"],
        output_names=["edge_probs"],
        dynamic_axes={
            "node_features": {0: "num_nodes"},
            "edge_index": {1: "num_edges"},
            "edge_features": {0: "num_edges"},
            "edge_probs": {0: "num_edges"},
        },
        opset_version=args.opset,
        dynamo=False,
    )
    print(f"exported -> {args.out}")

    # Parity check on a different graph size to exercise the dynamic axes.
    try:
        import numpy as np
        import onnxruntime as rt
    except ImportError:
        print("onnxruntime not installed; skipping parity check")
        return

    # export() restores the module's original training mode afterwards, so
    # force eval again before comparing (dropout must stay disabled).
    wrapper.eval()
    x, ei, ea = make_dummy_inputs(node_dim, edge_dim, n=37, e=143)
    with torch.no_grad():
        expected = wrapper(x, ei, ea).numpy()
    sess = rt.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    got = sess.run(
        ["edge_probs"],
        {
            "node_features": x.numpy(),
            "edge_index": ei.numpy(),
            "edge_features": ea.numpy(),
        },
    )[0]
    err = float(np.abs(expected - got).max())
    print(f"onnxruntime parity: max abs diff = {err:.3e}")
    assert err < 1e-4, "ONNX output diverges from PyTorch"


if __name__ == "__main__":
    main()
