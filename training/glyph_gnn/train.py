"""Training loop (Phase 2): focal loss, F1-based validation, best-checkpoint save.

Usage:
    python -m glyph_gnn.train --data ../dataset/train --out checkpoints/run
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from tqdm.auto import tqdm

from .dataset import load_dataset, make_loader
from .loss import focal_loss
from .model import GlyphEdgeGNN


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m {secs:02d}s" if hours else f"{minutes}m {secs:02d}s"


def dataset_stats(graphs) -> dict:
    nodes = sum(int(g.num_nodes) for g in graphs)
    edges = sum(int(g.y.numel()) for g in graphs)
    positives = sum(int(g.y.sum()) for g in graphs)
    fonts = len({g.font for g in graphs})
    return {"graphs": len(graphs), "nodes": nodes, "edges": edges,
            "pos_ratio": positives / edges if edges else 0.0, "fonts": fonts}


def print_run_summary(args, model, train_stats, val_stats, steps_per_epoch) -> None:
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    device_label = args.device
    if torch.device(args.device).type == "cuda":
        device_label += f" ({torch.cuda.get_device_name(args.device)})"

    def describe(stats: dict, suffix: str = "") -> str:
        return (
            f"{stats['graphs']:,} graphs | {stats['nodes']:,} nodes | {stats['edges']:,} edges"
            f" | {stats['pos_ratio'] * 100:.1f}% positive | {stats['fonts']} fonts{suffix}"
        )

    if val_stats["graphs"]:
        split = "random" if args.random_split else "held-out fonts"
        val_row = describe(val_stats, f" ({split})")
    else:
        val_row = "none (training metrics only)"

    model_row = (
        f"hidden {args.hidden} | layers {args.layers} | heads {args.heads}"
        f" | dropout {args.dropout} | {params:,} params"
    )
    optim_row = (
        f"AdamW lr {args.lr:.2e} | weight decay {args.weight_decay:.2e}"
        f" | cosine annealing T_max {args.epochs}"
    )
    schedule_row = (
        f"{args.epochs} epochs | batch <= {args.batch_size} graphs"
        f" / {args.max_edges_per_batch:,} edges | {steps_per_epoch:,} steps/epoch"
    )
    rows = [
        ("device", device_label),
        ("data", str(args.data)),
        ("train", describe(train_stats)),
        ("val", val_row),
        ("model", model_row),
        ("optimizer", optim_row),
        ("loss", f"focal alpha {args.focal_alpha} gamma {args.focal_gamma}"),
        ("schedule", schedule_row),
        ("output", str(Path(args.out))),
    ]

    width = max(len(f"{k:<10} {v}") for k, v in rows) + 2
    print("=" * width)
    print("Glyph edge GNN training".center(width))
    print("=" * width)
    for key, value in rows:
        print(f"{key:<10} {value}")
    print("=" * width)


def release(device) -> None:
    """Drops the allocator's cached blocks after an out-of-memory batch."""
    if device.type == "cuda":
        torch.cuda.empty_cache()


@torch.no_grad()
def evaluate(model, loader, device, threshold: float = 0.5, alpha: float = 0.25,
             gamma: float = 2.0, progress: bool = True) -> dict:
    model.eval()
    tp = fp = fn = tn = 0
    total_loss = 0.0
    graphs = 0
    bar = tqdm(loader, desc="  validate", unit="batch", leave=False,
               disable=None if progress else True)
    for batch in bar:
        batch = batch.to(device)
        try:
            logits = model(batch.x, batch.edge_index, batch.edge_attr)
            loss = focal_loss(logits, batch.y, alpha=alpha, gamma=gamma)
        except torch.OutOfMemoryError:
            batch = logits = loss = None
            release(device)
            continue
        total_loss += loss.item() * batch.num_graphs
        graphs += batch.num_graphs
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
    return {"precision": precision, "recall": recall, "f1": f1, "f1_neg": f1_n,
            "accuracy": (tp + tn) / max(tp + tn + fp + fn, 1),
            "loss": total_loss / max(graphs, 1)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the glyph edge-classification GNN")
    ap.add_argument("--data", required=True, help="directory of .msgpack shards")
    ap.add_argument("--out", default="checkpoints/run")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64, help="max graphs per batch")
    ap.add_argument("--max-edges-per-batch", type=int, default=200_000,
                    help="max edges per batch; lower this if the GPU runs out of memory")
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
    ap.add_argument("--no-progress", action="store_true", help="disable progress bars")
    args = ap.parse_args()

    progress = not args.no_progress
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    train_set, val_set = load_dataset(
        args.data,
        val_frac=args.val_frac,
        split_by_font=not args.random_split,
        seed=args.seed,
        limit_shards=args.limit_shards,
        progress=progress,
    )

    train_loader = make_loader(train_set, args.max_edges_per_batch, args.batch_size,
                               shuffle=True, seed=args.seed)
    val_loader = make_loader(val_set, args.max_edges_per_batch, args.batch_size)

    model = GlyphEdgeGNN(
        hidden=args.hidden, layers=args.layers, heads=args.heads, dropout=args.dropout
    ).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    print_run_summary(args, model, dataset_stats(train_set), dataset_stats(val_set),
                      len(train_loader))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    best_epoch = 0
    started = time.time()

    epochs = tqdm(range(1, args.epochs + 1), desc="epochs", unit="epoch",
                  disable=None if progress else True)
    for epoch in epochs:
        model.train()
        t0 = time.time()
        total_loss = 0.0
        seen = 0
        skipped = 0
        lr = optim.param_groups[0]["lr"]
        batches = tqdm(train_loader, desc=f"  epoch {epoch}/{args.epochs}", unit="batch",
                       leave=False, disable=None if progress else True)
        for batch in batches:
            batch = batch.to(device)
            optim.zero_grad(set_to_none=True)
            try:
                logits = model(batch.x, batch.edge_index, batch.edge_attr)
                loss = focal_loss(logits, batch.y, alpha=args.focal_alpha,
                                  gamma=args.focal_gamma)
                loss.backward()
            except torch.OutOfMemoryError:
                # One oversized batch should not kill a multi-hour run.
                optim.zero_grad(set_to_none=True)
                batch = logits = loss = None
                release(device)
                skipped += 1
                continue
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            total_loss += loss.detach().item() * batch.num_graphs
            seen += batch.num_graphs
            batches.set_postfix(loss=f"{total_loss / seen:.4f}", lr=f"{lr:.2e}",
                                grad=f"{float(grad_norm):.2f}", refresh=False)
        batches.close()
        if skipped:
            tqdm.write(f"epoch {epoch:3d}: skipped {skipped} batch(es) after CUDA OOM;"
                       " lower --max-edges-per-batch")
        sched.step()

        metrics = evaluate(model, val_loader, device, alpha=args.focal_alpha,
                           gamma=args.focal_gamma, progress=progress) if val_set else {}
        avg_loss = total_loss / max(seen, 1)
        elapsed = time.time() - t0

        score = metrics.get("f1_neg", -avg_loss)
        is_best = score > best_f1
        if is_best:
            best_f1, best_epoch = score, epoch
            torch.save(
                {"state_dict": model.state_dict(), "hparams": model.hparams, "epoch": epoch,
                 "metrics": metrics},
                out_dir / "best.pt",
            )

        line = (f"epoch {epoch:3d}/{args.epochs} | loss {avg_loss:.4f} | lr {lr:.2e}"
                f" | {format_duration(elapsed)} | {seen / max(elapsed, 1e-9):.0f} graphs/s")
        if device.type == "cuda":
            line += f" | peak {torch.cuda.max_memory_allocated(device) / 2**30:.1f}GiB"
            torch.cuda.reset_peak_memory_stats(device)
        if metrics:
            line += (
                f" | val loss {metrics['loss']:.4f} P {metrics['precision']:.4f}"
                f" R {metrics['recall']:.4f} F1 {metrics['f1']:.4f}"
                f" F1(neg) {metrics['f1_neg']:.4f}"
            )
        if is_best:
            line += "  <- best"
        tqdm.write(line)
        epochs.set_postfix(best=f"{best_f1:.4f}", refresh=False)
    epochs.close()

    torch.save(
        {"state_dict": model.state_dict(), "hparams": model.hparams, "epoch": args.epochs},
        out_dir / "last.pt",
    )
    print(f"finished {args.epochs} epochs in {format_duration(time.time() - started)}")
    print(f"best F1(neg) = {best_f1:.4f} @ epoch {best_epoch} -> {out_dir / 'best.pt'}")
    print(f"final weights -> {out_dir / 'last.pt'}")


if __name__ == "__main__":
    main()
