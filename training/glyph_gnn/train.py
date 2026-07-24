"""Training loop (Phase 2): focal loss, F1-based validation, best-checkpoint save.

Usage:
    python -m glyph_gnn.train --data ../dataset/train --out checkpoints/run
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from .dataset import load_dataset
from .loss import focal_loss
from .model import GlyphEdgeGNN


@torch.no_grad()
def evaluate(model, loader, device, threshold: float = 0.5) -> dict:
    model.eval()
    tp = fp = fn = tn = 0
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.edge_attr)
        pred = (torch.sigmoid(logits) >= threshold).float()
        y = batch.y
        tp += int(((pred == 1) & (y == 1)).sum())
        fp += int(((pred == 1) & (y == 0)).sum())
        fn += int(((pred == 0) & (y == 1)).sum())
        tn += int(((pred == 0) & (y == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    # Negative-class F1 matters most: negatives are the character boundaries.
    prec_n = tn / (tn + fn) if tn + fn else 0.0
    rec_n = tn / (tn + fp) if tn + fp else 0.0
    f1_n = 2 * prec_n * rec_n / (prec_n + rec_n) if prec_n + rec_n else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "f1_neg": f1_n}


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the glyph edge-classification GNN")
    ap.add_argument("--data", required=True, help="directory of .msgpack shards")
    ap.add_argument("--out", default="checkpoints/run")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--focal-alpha", type=float, default=0.25)
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--random-split", action="store_true", help="split by sample instead of by font")
    ap.add_argument("--limit-shards", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    train_set, val_set = load_dataset(
        args.data,
        val_frac=args.val_frac,
        split_by_font=not args.random_split,
        seed=args.seed,
        limit_shards=args.limit_shards,
    )
    print(f"dataset: {len(train_set)} train / {len(val_set)} val graphs")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)

    model = GlyphEdgeGNN(
        hidden=args.hidden, layers=args.layers, heads=args.heads, dropout=args.dropout
    ).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optim.zero_grad()
            logits = model(batch.x, batch.edge_index, batch.edge_attr)
            loss = focal_loss(logits, batch.y, alpha=args.focal_alpha, gamma=args.focal_gamma)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            total_loss += loss.detach().item() * batch.num_graphs
        sched.step()

        metrics = evaluate(model, val_loader, device) if val_set else {}
        avg_loss = total_loss / max(len(train_set), 1)
        line = f"epoch {epoch:3d} | loss {avg_loss:.4f} | {time.time() - t0:.1f}s"
        if metrics:
            line += (
                f" | val P {metrics['precision']:.4f} R {metrics['recall']:.4f}"
                f" F1 {metrics['f1']:.4f} F1(neg) {metrics['f1_neg']:.4f}"
            )
        print(line)

        score = metrics.get("f1_neg", -avg_loss)
        if score > best_f1:
            best_f1 = score
            torch.save(
                {"state_dict": model.state_dict(), "hparams": model.hparams, "epoch": epoch,
                 "metrics": metrics},
                out_dir / "best.pt",
            )
    torch.save(
        {"state_dict": model.state_dict(), "hparams": model.hparams, "epoch": args.epochs},
        out_dir / "last.pt",
    )
    print(f"best F1(neg) = {best_f1:.4f} -> {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
