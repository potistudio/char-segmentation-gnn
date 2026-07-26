"""Training loop (Phase 2): focal loss, grouping-based validation, best-checkpoint save.

Validation scores the decision that ships -- contour groups -- rather than the
point-edge proxy the loss is computed on; see :mod:`glyph_gnn.metrics` for why
the two are far apart. The best checkpoint is chosen by exact grouping
accuracy at its best threshold.

Usage:
    python -m glyph_gnn.train --data ../dataset/train --out checkpoints/run
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from tqdm.auto import tqdm

from .dataset import graph_stats, load_dataset, load_packed, make_loader
from .loss import glyph_loss
from .metrics import DEFAULT_THRESHOLDS, GroupingEvaluator, format_sweep
from .model import GlyphEdgeGNN
from .pack import is_packed


def loss_settings(args) -> dict:
    """Bundles the loss hyperparameters shared by training and validation."""
    return dict(
        alpha=args.focal_alpha,
        gamma=args.focal_gamma,
        pair_weight=args.pair_loss_weight,
        pair_alpha=args.pair_focal_alpha,
        intra_weight=args.intra_edge_weight,
        temperature=args.pair_temperature,
    )


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m {secs:02d}s" if hours else f"{minutes}m {secs:02d}s"


def print_run_summary(args, model, train_stats, val_stats, steps_per_epoch,
                      thresholds) -> None:
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
    source = "packed (memory-mapped)" if is_packed(args.data) else "msgpack (in memory)"
    rows = [
        ("device", device_label),
        ("data", f"{args.data} | {source}"),
        ("train", describe(train_stats)),
        ("val", val_row),
        ("model", model_row),
        ("optimizer", optim_row),
        ("loss", f"focal gamma {args.focal_gamma} |"
                 f" point edges alpha {args.focal_alpha}"
                 f" (intra x{args.intra_edge_weight})"
                 f" {1 - args.pair_loss_weight:.2f} :"
                 f" {args.pair_loss_weight:.2f} contour pairs"
                 f" alpha {args.pair_focal_alpha} (T {args.pair_temperature})"),
        ("select", "exact grouping accuracy over contour groups, best of"
                   f" {len(thresholds)} thresholds"),
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
def evaluate(model, loader, device, settings: dict,
             thresholds=DEFAULT_THRESHOLDS, progress: bool = True) -> dict:
    """Runs one validation pass and scores it at every threshold.

    Contour-pair probabilities are accumulated once and swept afterwards, so
    the sweep costs no extra forward passes. The point-edge scores are kept
    alongside for continuity with older runs, but they are not what selects
    the checkpoint: 75-79% of edges are intra-contour and trivially positive.
    """
    model.eval()
    grouping = GroupingEvaluator()
    tp = fp = fn = tn = 0
    total_loss = 0.0
    graphs = 0
    bar = tqdm(loader, desc="  validate", unit="batch", leave=False,
               disable=None if progress else True)
    for batch in bar:
        batch = batch.to(device)
        try:
            logits = model(batch.x, batch.edge_index, batch.edge_attr)
            loss, _ = glyph_loss(logits, batch.y, batch.contour_id, batch.edge_index,
                                 **settings)
        except torch.OutOfMemoryError:
            batch = logits = loss = None
            release(device)
            continue
        total_loss += loss.item() * batch.num_graphs
        graphs += batch.num_graphs
        probs = torch.sigmoid(logits)
        grouping.add(batch, probs)
        pred = (probs >= 0.5).float()
        y = batch.y
        tp += int(((pred == 1) & (y == 1)).sum())
        fp += int(((pred == 1) & (y == 0)).sum())
        fn += int(((pred == 0) & (y == 1)).sum())
        tn += int(((pred == 0) & (y == 0)).sum())
    # Negative-class F1 on point edges: the old selection metric, now a diagnostic.
    prec_n = tn / (tn + fn) if tn + fn else 0.0
    rec_n = tn / (tn + fp) if tn + fp else 0.0
    f1_n = 2 * prec_n * rec_n / (prec_n + rec_n) if prec_n + rec_n else 0.0

    sweep = grouping.sweep(thresholds)
    best = max(sweep, key=lambda m: (m["grouping_acc"], m["threshold"]))
    return {**best, "sweep": sweep, "edge_f1_neg": f1_n,
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
    ap.add_argument("--focal-alpha", type=float, default=0.25,
                    help="positive-class weight for the point-edge term")
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    ap.add_argument("--pair-loss-weight", type=float, default=0.5,
                    help="share of the loss taken by the pooled contour-pair term;"
                         " 0 reproduces the point-only objective")
    ap.add_argument("--pair-focal-alpha", type=float, default=0.5,
                    help="positive-class weight for the pair term. Pairs run far"
                         " closer to balanced than point edges do, and the ratio"
                         " differs by script -- read it off glyph_gnn.analyze")
    ap.add_argument("--intra-edge-weight", type=float, default=0.05,
                    help="weight on intra-contour edges, which are always positive"
                         " and free to predict; 0 drops them from the loss")
    ap.add_argument("--pair-temperature", type=float, default=1.0,
                    help="pooling temperature; approaches inference's hard max as"
                         " it approaches 0")
    ap.add_argument("--thresholds", default=None,
                    help="comma-separated grouping thresholds to sweep each epoch"
                         f" (default {','.join(str(t) for t in DEFAULT_THRESHOLDS)})")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--random-split", action="store_true", help="split by sample instead of by font")
    ap.add_argument("--limit-shards", type=int, default=None,
                    help="msgpack input only; pack a subset instead for packed stores")
    ap.add_argument("--num-workers", type=int, default=0,
                    help="loader worker processes; only used for packed stores")
    ap.add_argument("--warm-cache", action="store_true",
                    help="read the packed store sequentially first; on a spinning disk"
                         " this is far cheaper than faulting it in at random")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--knn", type=int, default=8, help="graph kNN used when generating training data")
    ap.add_argument("--radius", type=float, default=0.25, help="graph radius (em) used when generating training data")
    ap.add_argument("--contour-bridge", type=float, default=0.35,
                    help="contour-pair bridge distance (em) used when generating training data")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-progress", action="store_true", help="disable progress bars")
    args = ap.parse_args()

    progress = not args.no_progress
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    thresholds = (tuple(float(t) for t in args.thresholds.split(","))
                  if args.thresholds else DEFAULT_THRESHOLDS)
    settings = loss_settings(args)

    packed = is_packed(args.data)
    if packed:
        train_set, val_set = load_packed(
            args.data,
            val_frac=args.val_frac,
            split_by_font=not args.random_split,
            seed=args.seed,
        )
    else:
        train_set, val_set = load_dataset(
            args.data,
            val_frac=args.val_frac,
            split_by_font=not args.random_split,
            seed=args.seed,
            limit_shards=args.limit_shards,
            progress=progress,
        )

    if args.warm_cache:
        if not packed:
            print("--warm-cache ignored: msgpack shards are already read into memory")
        else:
            t0 = time.time()
            read = train_set.warm_cache(progress)
            print(f"warmed {read / 2**30:.2f} GiB of page cache"
                  f" in {format_duration(time.time() - t0)}")

    workers = args.num_workers if packed else 0
    if args.num_workers and not packed:
        print("--num-workers ignored: in-memory graphs would be copied to every worker")
    train_loader = make_loader(train_set, args.max_edges_per_batch, args.batch_size,
                               shuffle=True, seed=args.seed, num_workers=workers)
    val_loader = make_loader(val_set, args.max_edges_per_batch, args.batch_size,
                             num_workers=workers)

    model = GlyphEdgeGNN(
        hidden=args.hidden, layers=args.layers, heads=args.heads, dropout=args.dropout
    ).to(device)
    model.hparams.update(
        knn=args.knn,
        radius=args.radius,
        contour_bridge=args.contour_bridge,
    )
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    print_run_summary(args, model, graph_stats(train_set), graph_stats(val_set),
                      len(train_loader), thresholds)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_score = -1.0
    best_epoch = 0
    metrics: dict = {}
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
        parts = {"point": 0.0, "pair": 0.0}
        for batch in batches:
            batch = batch.to(device)
            optim.zero_grad(set_to_none=True)
            try:
                logits = model(batch.x, batch.edge_index, batch.edge_attr)
                loss, terms = glyph_loss(logits, batch.y, batch.contour_id,
                                         batch.edge_index, **settings)
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
            for key, value in terms.items():
                parts[key] += value * batch.num_graphs
            seen += batch.num_graphs
            batches.set_postfix(loss=f"{total_loss / seen:.4f}", lr=f"{lr:.2e}",
                                grad=f"{float(grad_norm):.2f}", refresh=False)
        batches.close()
        if skipped:
            tqdm.write(f"epoch {epoch:3d}: skipped {skipped} batch(es) after CUDA OOM;"
                       " lower --max-edges-per-batch")
        sched.step()

        metrics = evaluate(model, val_loader, device, settings, thresholds,
                           progress) if val_set else {}
        avg_loss = total_loss / max(seen, 1)
        elapsed = time.time() - t0

        score = metrics.get("grouping_acc", -avg_loss)
        is_best = score > best_score
        if is_best:
            best_score, best_epoch = score, epoch
            torch.save(
                {"state_dict": model.state_dict(), "hparams": model.hparams, "epoch": epoch,
                 "metrics": metrics, "loss_settings": settings},
                out_dir / "best.pt",
            )

        line = (f"epoch {epoch:3d}/{args.epochs} | loss {avg_loss:.4f}"
                f" (pt {parts['point'] / max(seen, 1):.4f}"
                f" pr {parts['pair'] / max(seen, 1):.4f}) | lr {lr:.2e}"
                f" | {format_duration(elapsed)} | {seen / max(elapsed, 1e-9):.0f} graphs/s")
        if device.type == "cuda":
            line += f" | peak {torch.cuda.max_memory_allocated(device) / 2**30:.1f}GiB"
            torch.cuda.reset_peak_memory_stats(device)
        if metrics:
            line += (
                f" | val loss {metrics['loss']:.4f} | pair F1 {metrics['pair_f1']:.4f}"
                f" | group {metrics['grouping_acc']:.4f} @{metrics['threshold']:.2f}"
                f" | split {metrics['char_split_rate']:.4f}"
                f" merge {metrics['char_merge_rate']:.4f}"
            )
        if is_best:
            line += "  <- best"
        tqdm.write(line)
        epochs.set_postfix(best=f"{best_score:.4f}", refresh=False)
    epochs.close()

    if metrics:
        print("\nthreshold sweep (final epoch):")
        print(format_sweep(metrics["sweep"]))

    torch.save(
        {"state_dict": model.state_dict(), "hparams": model.hparams,
         "epoch": args.epochs, "loss_settings": settings},
        out_dir / "last.pt",
    )
    print(f"\nfinished {args.epochs} epochs in {format_duration(time.time() - started)}")
    print(f"best grouping accuracy = {best_score:.4f} @ epoch {best_epoch}"
          f" -> {out_dir / 'best.pt'}")
    print(f"final weights -> {out_dir / 'last.pt'}")


if __name__ == "__main__":
    main()
