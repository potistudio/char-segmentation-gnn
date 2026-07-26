"""Grouping evaluation with a per-pattern breakdown (Phase 4).

Scores a checkpoint the way inference is judged -- contour groups, not point
edges -- and breaks the result down by source text. On a corpus of random
strings every text is unique and the breakdown is noise, so pair this with a
fixed-text shard built by ``dataset-gen --text-file``: that is what turns
"the model is 94% right" into "the model merges ll 40% of the time".

Usage:
    python -m glyph_gnn.eval --checkpoint training/checkpoints/run/best.pt \\
        --data dataset/confusables
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from unicodedata import east_asian_width

import torch
from tqdm.auto import tqdm

from .dataset import load_dataset, load_packed, make_loader
from .metrics import DEFAULT_THRESHOLDS, GroupingEvaluator, format_sweep
from .model import GlyphEdgeGNN, model_hparams
from .pack import is_packed


def load_all(data: str, progress: bool = True):
    """Opens a store without holding anything out; this is not a training split."""
    if is_packed(data):
        full, _ = load_packed(data, val_frac=0.0, split_by_font=False)
        return full
    full, _ = load_dataset(data, val_frac=0.0, split_by_font=False, progress=progress)
    return full


@torch.no_grad()
def run(model, loader, device, progress: bool = True) -> GroupingEvaluator:
    model.eval()
    grouping = GroupingEvaluator()
    bar = tqdm(loader, desc="evaluating", unit="batch", disable=None if progress else True)
    for batch in bar:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.edge_attr,
                       batch.contour_id, batch.contour_attr, batch.batch)
        grouping.add(batch, torch.sigmoid(logits))
    bar.close()
    return grouping


def display_width(text: str) -> int:
    """Terminal columns a string occupies.

    Kana and kanji are double width, so padding by ``len`` leaves the table
    ragged in exactly the runs worth reading.
    """
    return sum(2 if east_asian_width(c) in "WF" else 1 for c in text)


def breakdown(grouping: GroupingEvaluator, threshold: float, top: int) -> str:
    """Per-text grouping accuracy, worst first."""
    scored = grouping.score(threshold, per_graph=True)
    correct = scored["correct"]
    if len(grouping.texts) != len(correct):
        return "(no per-graph texts in this store; skipping the breakdown)"

    tally: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for text, ok in zip(grouping.texts, correct, strict=True):
        tally[text][0] += 1
        tally[text][1] += int(ok)

    rows = sorted(tally.items(), key=lambda kv: (kv[1][1] / kv[1][0], -kv[1][0]))
    width = max((display_width(t) for t, _ in rows[:top]), default=4)
    width = max(width, 4)
    lines = [f"{'text':<{width}} {'n':>6} {'correct':>8} {'acc':>8}",
             "-" * (width + 25)]
    for text, (total, ok) in rows[:top]:
        pad = " " * (width - display_width(text))
        lines.append(f"{text}{pad} {total:>6} {ok:>8} {ok / total:>8.4f}")
    if len(rows) > top:
        lines.append(f"... {len(rows) - top} more patterns")
    return "\n".join(lines)


def use_utf8_stdout() -> None:
    """Print the dataset's own text without mangling or crashing.

    This is the one tool that echoes characters straight out of the corpus.
    A redirected stdout on Windows defaults to cp932 here, which garbles the
    breakdown when it lands in a file and raises UnicodeEncodeError outright on
    any character outside the code page -- exactly the kanji worth inspecting.
    """
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> None:
    use_utf8_stdout()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", required=True, help="msgpack shard directory or packed store")
    ap.add_argument("--threshold", type=float, default=None,
                    help="score at this threshold instead of the sweep's best")
    ap.add_argument("--thresholds", default=None, help="comma-separated sweep grid")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-edges-per-batch", type=int, default=200_000)
    ap.add_argument("--top", type=int, default=30, help="patterns to list, worst first")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-progress", action="store_true")
    args = ap.parse_args()

    progress = not args.no_progress
    device = torch.device(args.device)
    thresholds = (tuple(float(t) for t in args.thresholds.split(","))
                  if args.thresholds else DEFAULT_THRESHOLDS)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = GlyphEdgeGNN(**model_hparams(ckpt["hparams"])).to(device)
    model.load_state_dict(ckpt["state_dict"])

    dataset = load_all(args.data, progress)
    loader = make_loader(dataset, args.max_edges_per_batch, args.batch_size)
    grouping = run(model, loader, device, progress)

    sweep = grouping.sweep(thresholds)
    print(f"\n{args.checkpoint} on {args.data}")
    print(f"{grouping.num_graphs:,} graphs\n")
    print(format_sweep(sweep))

    chosen = args.threshold
    if chosen is None:
        chosen = max(sweep, key=lambda m: (m["grouping_acc"], m["threshold"]))["threshold"]
    m = grouping.score(chosen)
    print(f"\nat threshold {chosen:.2f}:")
    print(f"  grouping accuracy  {m['grouping_acc']:.4f}  ({m['graphs']:,} graphs)")
    print(f"  characters split   {m['char_split_rate']:.4f}  ({m['chars']:,} characters)")
    print(f"  characters merged  {m['char_merge_rate']:.4f}")
    print(f"  contour-pair F1    {m['pair_f1']:.4f}  ({m['pairs']:,} pairs)")
    print(f"\nper-pattern breakdown @ {chosen:.2f} (worst first):")
    print(breakdown(grouping, chosen, args.top))


if __name__ == "__main__":
    main()
