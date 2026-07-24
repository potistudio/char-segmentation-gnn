//! Synthetic dataset generator (Phase 1).
//!
//! Loads body/decorative font pools, lays out random text with aggressive
//! augmentation (negative tracking to force path intersections, baseline
//! jitter, aspect changes, point noise), converts the result into edge-
//! labeled graphs and writes MessagePack shards in parallel with rayon.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{bail, Context, Result};
use clap::Parser;
use rand::{RngExt, SeedableRng};
use rand_pcg::Pcg64Mcg;
use rayon::prelude::*;

use glyph_core::{build_graph, layout_text, GraphConfig, GraphSample, LayoutConfig, SampleConfig};

#[derive(Parser, Debug)]
#[command(name = "dataset-gen", about = "Generate synthetic glyph-graph datasets")]
struct Args {
    /// Directory containing body-text fonts (.ttf/.otf).
    #[arg(long)]
    body_fonts: PathBuf,

    /// Directory containing decorative fonts. Falls back to body fonts.
    #[arg(long)]
    deco_fonts: Option<PathBuf>,

    /// Probability of picking a decorative font (plan: 7:3 ratio).
    #[arg(long, default_value_t = 0.3)]
    deco_ratio: f64,

    /// Number of graph samples to generate.
    #[arg(long, default_value_t = 100_000)]
    count: usize,

    /// Output directory for MessagePack shards.
    #[arg(long)]
    out: PathBuf,

    /// Samples per shard file.
    #[arg(long, default_value_t = 1000)]
    shard_size: usize,

    /// RNG seed (per-sample streams are derived from this).
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Min / max characters per sample.
    #[arg(long, default_value_t = 2)]
    min_chars: usize,
    #[arg(long, default_value_t = 12)]
    max_chars: usize,

    /// Characters to draw text from.
    #[arg(
        long,
        default_value = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    )]
    charset: String,

    /// Graph construction: max neighbors per node.
    #[arg(long, default_value_t = 8)]
    knn: usize,

    /// Graph construction: connection radius in em units.
    #[arg(long, default_value_t = 0.25)]
    radius: f32,

    /// Contour resampling spacing in em units.
    #[arg(long, default_value_t = 0.05)]
    spacing: f32,

    /// Max sampled points per contour.
    #[arg(long, default_value_t = 64)]
    max_points: usize,
}

struct FontEntry {
    name: String,
    data: Vec<u8>,
}

fn load_fonts(dir: &Path) -> Result<Vec<FontEntry>> {
    let mut fonts = Vec::new();
    for entry in fs::read_dir(dir).with_context(|| format!("reading {}", dir.display()))? {
        let path = entry?.path();
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_ascii_lowercase());
        // .ttc collections are opened at face index 0.
        if !matches!(ext.as_deref(), Some("ttf" | "otf" | "ttc")) {
            continue;
        }
        let data = fs::read(&path)?;
        // Validate up front so workers never hit unparsable fonts.
        if rustybuzz_face_ok(&data) {
            fonts.push(FontEntry {
                name: path.file_name().unwrap().to_string_lossy().into_owned(),
                data,
            });
        } else {
            eprintln!("skipping unparsable font: {}", path.display());
        }
    }
    Ok(fonts)
}

fn rustybuzz_face_ok(data: &[u8]) -> bool {
    glyph_core::layout::layout_probe(data)
}

fn random_text(rng: &mut Pcg64Mcg, charset: &[char], min: usize, max: usize) -> String {
    let len = rng.random_range(min..=max);
    (0..len).map(|_| charset[rng.random_range(0..charset.len())]).collect()
}

fn generate_one(
    idx: usize,
    args: &Args,
    body: &[FontEntry],
    deco: &[FontEntry],
    charset: &[char],
    graph_cfg: &GraphConfig,
) -> Option<GraphSample> {
    // Independent, reproducible RNG stream per sample.
    let mut rng = Pcg64Mcg::seed_from_u64(args.seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ idx as u64);

    let use_deco = !deco.is_empty() && rng.random_range(0.0..1.0) < args.deco_ratio;
    let pool = if use_deco { deco } else { body };
    let font = &pool[rng.random_range(0..pool.len())];

    let text = random_text(&mut rng, charset, args.min_chars, args.max_chars);

    let layout_cfg = LayoutConfig {
        // Bias toward squeezing so intersecting paths dominate the dataset.
        tracking: rng.random_range(-0.18f32..0.02),
        baseline_jitter: rng.random_range(0.0f32..0.06),
        aspect_x: rng.random_range(0.8f32..1.25),
        point_noise: rng.random_range(0.0f32..0.004),
    };

    let (contours, upem) = layout_text(&font.data, &text, &layout_cfg, &mut rng).ok()?;
    let mut sample = build_graph(&contours, upem, graph_cfg);
    if sample.num_nodes < 8 || sample.num_edges() == 0 {
        return None;
    }
    sample.text = text;
    sample.font = font.name.clone();
    Some(sample)
}

fn main() -> Result<()> {
    let args = Args::parse();

    let body = load_fonts(&args.body_fonts)?;
    if body.is_empty() {
        bail!("no usable fonts in {}", args.body_fonts.display());
    }
    let deco = match &args.deco_fonts {
        Some(dir) => load_fonts(dir)?,
        None => Vec::new(),
    };
    println!(
        "fonts loaded: {} body, {} decorative (deco ratio {:.0}%)",
        body.len(),
        deco.len(),
        args.deco_ratio * 100.0
    );

    let charset: Vec<char> = args.charset.chars().collect();
    if charset.is_empty() {
        bail!("charset is empty");
    }

    let graph_cfg = GraphConfig {
        sample: SampleConfig {
            spacing: args.spacing,
            min_points: 6,
            max_points: args.max_points,
        },
        knn: args.knn,
        radius: args.radius,
    };

    fs::create_dir_all(&args.out)?;
    let num_shards = args.count.div_ceil(args.shard_size);
    let written = AtomicUsize::new(0);

    let start = std::time::Instant::now();
    (0..num_shards).into_par_iter().try_for_each(|shard| -> Result<()> {
        let lo = shard * args.shard_size;
        let hi = ((shard + 1) * args.shard_size).min(args.count);
        let samples: Vec<GraphSample> = (lo..hi)
            .filter_map(|i| generate_one(i, &args, &body, &deco, &charset, &graph_cfg))
            .collect();

        let payload = rmp_serde::to_vec_named(&samples)?;
        let path = args.out.join(format!("shard_{shard:05}.msgpack"));
        fs::write(&path, payload)?;

        let done = written.fetch_add(samples.len(), Ordering::Relaxed) + samples.len();
        if shard % 10 == 0 {
            println!(
                "shard {shard}/{num_shards} written ({done} samples, {:.1}s elapsed)",
                start.elapsed().as_secs_f32()
            );
        }
        Ok(())
    })?;

    println!(
        "done: {} samples in {:.1}s -> {}",
        written.load(Ordering::Relaxed),
        start.elapsed().as_secs_f32(),
        args.out.display()
    );
    Ok(())
}
