"""ONNX export (Phase 2, step 4).

Exports the trained model with dynamic node/edge axes and verifies the
result against PyTorch using onnxruntime.

Two things here exist for CPU inference speed and are described where they
happen: the scatter sites trace to a row-wise ``ScatterND``, and the classifier
runs only on the edges postprocessing actually reads.

Usage:
    python -m glyph_gnn.export_onnx --checkpoint checkpoints/run/best.pt --out ../model.onnx
"""

from __future__ import annotations

import argparse

import torch

from .model import GlyphEdgeGNN, graph_hparams, model_hparams, row_wise_scatter
from .postprocess import group_characters

#: Thresholds the parity check sweeps. Wider than the useful range on purpose:
#: a disagreement anywhere in it means the export moved a decision.
PARITY_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


class InferenceWrapper(torch.nn.Module):
    """Adds the sigmoid so the Rust side receives probabilities directly.

    ``scored`` selects which edges reach the classifier. Message passing still
    runs over every edge, so the receptive field is untouched, but
    ``postprocess.rs`` folds edges into contour pairs and skips the ones whose
    endpoints share a contour -- 72% of them on Japanese text. Those rows were
    being computed and thrown away, and the classifier is 75,584 MAC per edge
    against the 2,048 the attention layers spend, so they were most of the
    per-edge cost. Passing every index reproduces the old behaviour exactly.
    """

    def __init__(self, model: GlyphEdgeGNN):
        super().__init__()
        self.model = model

    def forward(self, node_features, edge_index, edge_features,
                node_contour_ids, contour_features, scored):
        # batch stays None: inference runs one line at a time, so every
        # pooling reduces over the whole input and needs no graph vector.
        hx, summary = self.model.trunk(node_features, edge_index, edge_features,
                                       node_contour_ids, contour_features, None)
        return torch.sigmoid(self.model.edge_logits(
            hx, summary, edge_index[:, scored], edge_features[scored],
            node_contour_ids, contour_features, None))


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
    scored = torch.nonzero(
        contour_id[edge_index[0]] != contour_id[edge_index[1]], as_tuple=True
    )[0]
    return x, edge_index, edge_attr, contour_id, contour_attr, scored


def force_scatter_add(path: str) -> int:
    """Restores ``reduction="add"`` on the exported ``ScatterND`` nodes.

    torch.onnx drops ``accumulate=True`` from ``index_put_``: the node comes out
    with no ``reduction`` attribute, which ONNX defaults to "none" -- the last
    write to a row wins instead of the rows summing. Message passing feeds many
    edges to the same node, so the result is not approximate but wrong, and
    nothing upstream complains: the export succeeds and onnx.checker passes.
    Measured on a real line, output diverged by 0.9999.
    """
    import onnx
    from onnx import helper

    model = onnx.load(path)
    patched = 0
    for node in model.graph.node:
        if node.op_type != "ScatterND":
            continue
        reduction = next((a for a in node.attribute if a.name == "reduction"), None)
        if reduction is None:
            node.attribute.append(helper.make_attribute("reduction", "add"))
            patched += 1
        elif reduction.s != b"add":
            raise AssertionError(f"{node.name} has reduction={reduction.s!r}, expected add")
    onnx.save(model, path)
    return patched


def grouping(contour_id, edge_index, scored, probs, threshold):
    """Character groups the Rust postprocessor would produce from these probs."""
    src = edge_index[0][scored].tolist()
    dst = edge_index[1][scored].tolist()
    groups = group_characters(len(contour_id), contour_id.tolist(), src, dst,
                              [float(p) for p in probs], threshold)
    return sorted(tuple(g) for g in groups)


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

    with row_wise_scatter():
        torch.onnx.export(
            wrapper,
            dummy,
            args.out,
            input_names=["node_features", "edge_index", "edge_features",
                         "node_contour_ids", "contour_features", "scored"],
            output_names=["edge_probs"],
            dynamic_axes={
                "node_features": {0: "num_nodes"},
                "edge_index": {1: "num_edges"},
                "edge_features": {0: "num_edges"},
                "node_contour_ids": {0: "num_nodes"},
                "contour_features": {0: "num_contours"},
                "scored": {0: "num_scored"},
                "edge_probs": {0: "num_scored"},
            },
            opset_version=args.opset,
            dynamo=False,
        )
    print(f"exported -> {args.out} ({force_scatter_add(args.out)} ScatterND nodes set to add)")

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
    x, ei, ea, ci, ca, sc = make_dummy_inputs(node_dim, edge_dim, contour_dim,
                                              n=37, e=143, c=11)
    with torch.no_grad():
        expected = wrapper(x, ei, ea, ci, ca, sc).numpy()
    sess = rt.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    got = sess.run(
        ["edge_probs"],
        {
            "node_features": x.numpy(),
            "edge_index": ei.numpy(),
            "edge_features": ea.numpy(),
            "node_contour_ids": ci.numpy(),
            "contour_features": ca.numpy(),
            "scored": sc.numpy(),
        },
    )[0]
    err = float(np.abs(expected - got).max())
    # ScatterND sums a node's incoming edges in whatever order the runtime
    # partitions them, so the two paths agree to a few 1e-3 rather than
    # exactly. What has to hold is that no decision moved, which is why the
    # check below is on the grouping and not on the float.
    print(f"onnxruntime parity: max abs diff = {err:.3e}")
    assert err < 5e-2, "ONNX output diverges from PyTorch"
    for threshold in PARITY_THRESHOLDS:
        mine = grouping(ci, ei, sc, expected, threshold)
        theirs = grouping(ci, ei, sc, got, threshold)
        assert mine == theirs, f"grouping differs at threshold {threshold}"
    print(f"grouping identical across thresholds {PARITY_THRESHOLDS}")


if __name__ == "__main__":
    main()
