"""ONNX export (Phase 2, step 4).

Exports the trained model with dynamic node/edge axes and verifies the
result against PyTorch using onnxruntime.

Usage:
    python -m glyph_gnn.export_onnx --checkpoint checkpoints/run/best.pt --out ../model.onnx
"""

from __future__ import annotations

import argparse

import torch

from .model import GlyphEdgeGNN, graph_hparams, model_hparams


class InferenceWrapper(torch.nn.Module):
    """Adds the sigmoid so the Rust side receives probabilities directly."""

    def __init__(self, model: GlyphEdgeGNN):
        super().__init__()
        self.model = model

    def forward(self, node_features, edge_index, edge_features,
                node_contour_ids, contour_features):
        # batch stays None: inference runs one line at a time, so every
        # pooling reduces over the whole input and needs no graph vector.
        return torch.sigmoid(self.model(node_features, edge_index, edge_features,
                                        node_contour_ids, contour_features))


def make_dummy_inputs(node_dim: int, edge_dim: int, contour_dim: int,
                      n: int = 50, e: int = 200, c: int = 7):
    x = torch.randn(n, node_dim)
    edge_index = torch.randint(0, n, (2, e), dtype=torch.long)
    edge_attr = torch.randn(e, edge_dim)
    # Every contour must own at least one node, or its supernode row is a
    # mean over nothing; the real builder guarantees that.
    contour_id = torch.cat([torch.arange(c), torch.randint(0, c, (n - c,))]).long()
    # Contour features must stay in their real domain, not just their shape:
    # widths, heights and arc lengths are non-negative, and the model takes a
    # log of the arc length ratio, so plain randn would trace a NaN.
    contour_attr = torch.rand(c, contour_dim)
    return x, edge_index, edge_attr, contour_id, contour_attr


def main() -> None:
    ap = argparse.ArgumentParser(description="Export trained GNN to ONNX")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="model.onnx")
    ap.add_argument("--opset", type=int, default=18)
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    hparams = ckpt["hparams"]
    model = GlyphEdgeGNN(**model_hparams(hparams))
    model.load_state_dict(ckpt["state_dict"])
    wrapper = InferenceWrapper(model)
    wrapper.eval()

    graph_cfg = graph_hparams(hparams)
    if graph_cfg:
        print(f"graph config in checkpoint: {graph_cfg}")
        print("pass matching --knn / --radius / --contour-bridge to glyph-infer")

    node_dim = hparams["node_dim"]
    edge_dim = hparams["edge_dim"]
    contour_dim = hparams.get("contour_dim", 8)
    dummy = make_dummy_inputs(node_dim, edge_dim, contour_dim)

    torch.onnx.export(
        wrapper,
        dummy,
        args.out,
        input_names=["node_features", "edge_index", "edge_features",
                     "node_contour_ids", "contour_features"],
        output_names=["edge_probs"],
        dynamic_axes={
            "node_features": {0: "num_nodes"},
            "edge_index": {1: "num_edges"},
            "edge_features": {0: "num_edges"},
            "node_contour_ids": {0: "num_nodes"},
            "contour_features": {0: "num_contours"},
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
    # Different node, edge and contour counts, to exercise all three axes.
    x, ei, ea, ci, ca = make_dummy_inputs(node_dim, edge_dim, contour_dim,
                                          n=37, e=143, c=11)
    with torch.no_grad():
        expected = wrapper(x, ei, ea, ci, ca).numpy()
    sess = rt.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    got = sess.run(
        ["edge_probs"],
        {
            "node_features": x.numpy(),
            "edge_index": ei.numpy(),
            "edge_features": ea.numpy(),
            "node_contour_ids": ci.numpy(),
            "contour_features": ca.numpy(),
        },
    )[0]
    err = float(np.abs(expected - got).max())
    print(f"onnxruntime parity: max abs diff = {err:.3e}")
    assert err < 1e-4, "ONNX output diverges from PyTorch"


if __name__ == "__main__":
    main()
