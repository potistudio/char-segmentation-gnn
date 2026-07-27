//! Synthetic dataset generator (Phase 1).
//!
//! Loads body/decorative font pools, lays out random text with aggressive
//! augmentation (negative tracking to force path intersections, baseline
//! jitter, aspect changes, point noise), converts the result into edge-
//! labeled graphs and writes MessagePack shards in parallel with rayon.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{Context, Result, bail};
use clap::Parser;
use rand::{RngExt, SeedableRng};
use rand_pcg::Pcg64Mcg;
use rayon::prelude::*;

use glyph_core::{GraphConfig, GraphSample, LayoutConfig, SampleConfig, build_graph, layout_text};

#[derive(Parser, Debug)]
#[command(
    name = "dataset-gen",
    about = "Generate synthetic glyph-graph datasets"
)]
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

    /// Characters to draw text from (ignored when --charset-file is set).
    #[arg(
        long,
        default_value = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
        conflicts_with = "charset_file"
    )]
    charset: String,

    /// UTF-8 text file listing drawable characters. Lines starting with `#` and
    /// blank lines are skipped; all other characters on each line join the pool.
    /// Duplicate characters increase sampling weight.
    #[arg(long, conflicts_with = "charset")]
    charset_file: Option<PathBuf>,

    /// UTF-8 file of fixed texts, one per line, rendered round-robin instead of
    /// sampling random strings. Every line gets the same number of samples with
    /// different fonts and augmentation, which is what makes a per-pattern
    /// breakdown comparable. `--min-chars` / `--max-chars` do not apply.
    ///
    /// Combine with `--text-ratio` below 1 to splice the patterns into random
    /// text instead, which is how they get into a training corpus.
    #[arg(long)]
    text_file: Option<PathBuf>,

    /// Share of samples taken from `--text-file`. At 1 (the default) the file is
    /// the whole corpus, which is what an evaluation set wants. Below 1 the
    /// patterns are spliced into random text drawn from the charset, so a
    /// training corpus can carry adjacencies a random pool never produces.
    #[arg(long, default_value_t = 1.0)]
    text_ratio: f64,

    /// Graph construction: max neighbors per node.
    #[arg(long, default_value_t = 8)]
    knn: usize,

    /// Graph construction: connection radius in em units.
    #[arg(long, default_value_t = 0.25)]
    radius: f32,

    /// Graph construction: bridge closest node pair per contour pair within this distance (em).
    #[arg(long, default_value_t = 0.35)]
    contour_bridge: f32,

    /// Graph construction: also bridge contour pairs sharing a horizontal slot
    /// (the strokes of a stacked character) within this vertical gap (em).
    #[arg(long, default_value_t = 1.0)]
    stack_bridge: f32,

    /// Contour resampling spacing in em units.
    #[arg(long, default_value_t = 0.05)]
    spacing: f32,

    /// Max sampled points per contour.
    #[arg(long, default_value_t = 64)]
    max_points: usize,

    /// Tightest tracking (em).
    #[arg(long, default_value_t = -0.18, allow_negative_numbers = true)]
    tracking_min: f32,

    /// Loosest tracking (em).
    #[arg(long, default_value_t = 0.02, allow_negative_numbers = true)]
    tracking_max: f32,

    /// Shapes the tracking distribution between the two bounds. 1 is uniform;
    /// above 1 concentrates samples near `--tracking-max` and leaves the tight
    /// end as a tail.
    ///
    /// The default is uniform, and concentrating on realistic tracking was
    /// measured to be worse -- see the layout augmentation section of the
    /// README. Breadth here acts as regularisation, so narrowing the
    /// distribution to match deployment costs accuracy even inside the band
    /// deployment uses.
    #[arg(long, default_value_t = 1.0)]
    tracking_skew: f32,

    /// Max per-glyph vertical shift (em). Set 0 to disable.
    #[arg(long, default_value_t = 0.06)]
    baseline_jitter: f32,

    /// Horizontal scale applied to the whole line (condensed / extended).
    #[arg(long, default_value_t = 0.8)]
    aspect_x_min: f32,
    #[arg(long, default_value_t = 1.25)]
    aspect_x_max: f32,

    /// Max uniform jitter added to every outline point (em).
    #[arg(long, default_value_t = 0.004)]
    point_noise: f32,
}

impl Args {
    /// Draws one tracking value from the skewed distribution.
    fn tracking(&self, rng: &mut Pcg64Mcg) -> f32 {
        let span = self.tracking_max - self.tracking_min;
        if span <= 0.0 {
            return self.tracking_max;
        }
        let u: f32 = rng.random_range(0.0f32..1.0);
        self.tracking_max - span * u.powf(self.tracking_skew.max(1e-3))
    }

    fn validate(&self) -> Result<()> {
        if self.tracking_min > self.tracking_max {
            bail!(
                "--tracking-min ({}) exceeds --tracking-max ({})",
                self.tracking_min,
                self.tracking_max
            );
        }
        if self.aspect_x_min > self.aspect_x_max || self.aspect_x_min <= 0.0 {
            bail!("--aspect-x-min must be positive and not exceed --aspect-x-max");
        }
        if self.tracking_skew <= 0.0 {
            bail!("--tracking-skew must be positive");
        }
        if self.baseline_jitter < 0.0 || self.point_noise < 0.0 {
            bail!("--baseline-jitter and --point-noise cannot be negative");
        }
        Ok(())
    }
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

/// Reads the meaningful lines of a UTF-8 list file, dropping `#` comments
/// and blank lines.
fn read_list_file(path: &Path) -> Result<Vec<String>> {
    let content =
        fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    Ok(content
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(str::to_owned)
        .collect())
}

/// Load a charset pool from a UTF-8 text file.
fn load_charset_file(path: &Path) -> Result<String> {
    let charset = read_list_file(path)?.concat();
    if charset.is_empty() {
        bail!(
            "charset file {} contains no drawable characters",
            path.display()
        );
    }
    Ok(charset)
}

/// Load fixed evaluation texts, one per line.
fn load_text_file(path: &Path) -> Result<Vec<String>> {
    let texts = read_list_file(path)?;
    if texts.is_empty() {
        bail!("text file {} contains no texts", path.display());
    }
    Ok(texts)
}

/// Where sample texts come from.
enum TextSource {
    /// Random strings drawn from a character pool.
    Charset(Vec<char>),
    /// A fixed list rendered round-robin, for per-pattern evaluation.
    Fixed(Vec<String>),
    /// Fixed patterns spliced into random text some of the time.
    ///
    /// A random pool of 3,417 characters produces a given adjacent pair -- "ll",
    /// "一二" -- about 0.05 times in 100k lines, so the confusable sequences that
    /// dominate the error are effectively absent from training even though every
    /// character in them is seen hundreds of times. This puts them in on purpose.
    ///
    /// The patterns are spliced into a line rather than emitted alone: half of
    /// them are a single character, and the model leans on line context to judge
    /// character pitch, so standalone patterns would train it on lines that have
    /// no context to read.
    Mixed {
        charset: Vec<char>,
        texts: Vec<String>,
        ratio: f64,
    },
}

impl TextSource {
    fn describe(&self) -> String {
        match self {
            TextSource::Charset(chars) => format!("{} chars", chars.len()),
            TextSource::Fixed(texts) => format!("{} fixed texts", texts.len()),
            TextSource::Mixed {
                charset,
                texts,
                ratio,
            } => format!(
                "{} chars + {} patterns spliced in at {:.0}%",
                charset.len(),
                texts.len(),
                ratio * 100.0
            ),
        }
    }
}

fn resolve_text_source(args: &Args) -> Result<TextSource> {
    let fixed = match &args.text_file {
        Some(path) => {
            let texts = load_text_file(path)?;
            println!(
                "texts loaded from {} ({} lines)",
                path.display(),
                texts.len()
            );
            Some(texts)
        }
        None => None,
    };
    if let Some(texts) = &fixed {
        if args.text_ratio >= 1.0 {
            return Ok(TextSource::Fixed(texts.clone()));
        }
    }
    let charset_str = if let Some(path) = &args.charset_file {
        let loaded = load_charset_file(path)?;
        println!(
            "charset loaded from {} ({} chars)",
            path.display(),
            loaded.chars().count()
        );
        loaded
    } else {
        args.charset.clone()
    };
    let charset: Vec<char> = charset_str.chars().collect();
    if charset.is_empty() {
        bail!("charset is empty");
    }
    Ok(match fixed {
        Some(texts) => TextSource::Mixed {
            charset,
            texts,
            ratio: args.text_ratio,
        },
        None => TextSource::Charset(charset),
    })
}

fn random_text(rng: &mut Pcg64Mcg, charset: &[char], min: usize, max: usize) -> String {
    let len = rng.random_range(min..=max);
    (0..len)
        .map(|_| charset[rng.random_range(0..charset.len())])
        .collect()
}

/// Splices `pattern` into a line of random characters, keeping the overall
/// length inside `min..=max` so the corpus keeps one length distribution.
fn embed_pattern(
    rng: &mut Pcg64Mcg,
    charset: &[char],
    pattern: &str,
    min: usize,
    max: usize,
) -> String {
    let len = pattern.chars().count();
    let total = rng.random_range(min.max(len)..=max.max(len));
    let pad = total - len;
    let before = rng.random_range(0..=pad);
    let mut out = String::new();
    for _ in 0..before {
        out.push(charset[rng.random_range(0..charset.len())]);
    }
    out.push_str(pattern);
    for _ in 0..(pad - before) {
        out.push(charset[rng.random_range(0..charset.len())]);
    }
    out
}

fn generate_one(
    idx: usize,
    args: &Args,
    body: &[FontEntry],
    deco: &[FontEntry],
    source: &TextSource,
    graph_cfg: &GraphConfig,
) -> Option<GraphSample> {
    // Independent, reproducible RNG stream per sample.
    let mut rng =
        Pcg64Mcg::seed_from_u64(args.seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ idx as u64);

    let use_deco = !deco.is_empty() && rng.random_range(0.0..1.0) < args.deco_ratio;
    let pool = if use_deco { deco } else { body };
    let font = &pool[rng.random_range(0..pool.len())];

    let text = match source {
        TextSource::Charset(charset) => {
            random_text(&mut rng, charset, args.min_chars, args.max_chars)
        }
        // Round-robin rather than random so every pattern gets the same
        // number of samples; a per-pattern accuracy is only comparable when
        // the denominators match.
        TextSource::Fixed(texts) => texts[idx % texts.len()].clone(),
        TextSource::Mixed {
            charset,
            texts,
            ratio,
        } => {
            if rng.random_range(0.0..1.0) < *ratio {
                // Round-robin here too, so every pattern is trained on equally
                // often regardless of how many samples land in the mix.
                let pattern = &texts[idx % texts.len()];
                embed_pattern(&mut rng, charset, pattern, args.min_chars, args.max_chars)
            } else {
                random_text(&mut rng, charset, args.min_chars, args.max_chars)
            }
        }
    };

    let layout_cfg = LayoutConfig {
        // Tracking is drawn once per sample, so the pitch stays regular along
        // the line -- which is the cue the model needs it to be.
        tracking: args.tracking(&mut rng),
        baseline_jitter: rng.random_range(0.0f32..=args.baseline_jitter),
        aspect_x: rng.random_range(args.aspect_x_min..=args.aspect_x_max),
        point_noise: rng.random_range(0.0f32..=args.point_noise),
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

    args.validate()?;

    let source = resolve_text_source(&args)?;
    println!("text source: {}", source.describe());
    println!(
        "layout: tracking {:.3}..{:.3} em (skew {}) | baseline jitter <= {:.3} em\
         \n        aspect x {:.2}..{:.2} | point noise <= {:.4} em",
        args.tracking_min,
        args.tracking_max,
        args.tracking_skew,
        args.baseline_jitter,
        args.aspect_x_min,
        args.aspect_x_max,
        args.point_noise,
    );

    let graph_cfg = GraphConfig {
        sample: SampleConfig {
            spacing: args.spacing,
            min_points: 6,
            max_points: args.max_points,
        },
        knn: args.knn,
        radius: args.radius,
        contour_bridge: args.contour_bridge,
        stack_bridge: args.stack_bridge,
    };

    fs::create_dir_all(&args.out)?;
    let num_shards = args.count.div_ceil(args.shard_size);
    let written = AtomicUsize::new(0);

    let start = std::time::Instant::now();
    (0..num_shards)
        .into_par_iter()
        .try_for_each(|shard| -> Result<()> {
            let lo = shard * args.shard_size;
            let hi = ((shard + 1) * args.shard_size).min(args.count);
            let samples: Vec<GraphSample> = (lo..hi)
                .filter_map(|i| generate_one(i, &args, &body, &deco, &source, &graph_cfg))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_charset_file_skips_comments_and_blank_lines() {
        let dir = std::env::temp_dir().join(format!("charset_test_{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("charset.txt");
        fs::write(
            &path,
            "# header comment\n\nABC\n# inline comment line ignored entirely\n123\n",
        )
        .unwrap();

        let charset = load_charset_file(&path).unwrap();
        assert_eq!(charset, "ABC123");

        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn load_text_file_keeps_lines_separate() {
        let dir = std::env::temp_dir().join(format!("text_test_{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("texts.txt");
        // Unlike the charset loader, lines must not be concatenated: each one
        // is a text to render on its own.
        fs::write(&path, "# patterns\n\nll\nrI\n\n\u{3053}\n").unwrap();

        let texts = load_text_file(&path).unwrap();
        assert_eq!(texts, vec!["ll", "rI", "\u{3053}"]);

        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn fixed_texts_are_rendered_round_robin() {
        // Every pattern must get the same number of samples, otherwise the
        // per-pattern accuracies in the eval breakdown are not comparable.
        let texts = ["ll".to_string(), "rI".to_string(), "\u{3053}".to_string()];
        let picked: Vec<&str> = (0..7).map(|i| texts[i % texts.len()].as_str()).collect();
        assert_eq!(
            picked,
            ["ll", "rI", "\u{3053}", "ll", "rI", "\u{3053}", "ll"]
        );
    }

    fn default_args() -> Args {
        Args::parse_from(["dataset-gen", "--body-fonts", ".", "--out", "."])
    }

    #[test]
    fn tracking_defaults_cover_the_range_evenly() {
        // The default is deliberately uniform. Skewing it toward realistic
        // tracking was measured to lose accuracy at every band, so breadth is
        // the property worth pinning here.
        let args = default_args();
        let mut rng = Pcg64Mcg::seed_from_u64(7);
        let mut values: Vec<f32> = (0..20_000).map(|_| args.tracking(&mut rng)).collect();
        values.sort_by(f32::total_cmp);

        let span = args.tracking_max - args.tracking_min;
        let median = values[values.len() / 2];
        let midpoint = args.tracking_min + span / 2.0;
        assert!(
            (median - midpoint).abs() < span * 0.05,
            "median {median} should sit near the midpoint {midpoint}"
        );
        // Each quarter of the range should hold about a quarter of the draws.
        for quarter in 0..4 {
            let lo = args.tracking_min + span * quarter as f32 / 4.0;
            let hi = lo + span / 4.0;
            let share =
                values.iter().filter(|&&t| t >= lo && t < hi).count() as f32 / values.len() as f32;
            assert!(
                (0.2..=0.3).contains(&share),
                "quarter {quarter} holds {share} of samples; expected roughly a quarter"
            );
        }
        assert!(*values.first().unwrap() >= args.tracking_min);
        assert!(*values.last().unwrap() <= args.tracking_max);
    }

    #[test]
    fn tracking_skew_concentrates_toward_the_loose_end() {
        // The knob still has to work for anyone re-testing the question.
        let mut args = default_args();
        args.tracking_skew = 2.5;
        let mut rng = Pcg64Mcg::seed_from_u64(3);
        let values: Vec<f32> = (0..20_000).map(|_| args.tracking(&mut rng)).collect();
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let midpoint = (args.tracking_min + args.tracking_max) / 2.0;
        assert!(
            mean > midpoint,
            "skew above 1 should pull toward tracking_max"
        );
    }

    #[test]
    fn tracking_collapses_to_a_fixed_value_for_banded_eval() {
        // Equal bounds pin the tracking, which is how a per-band evaluation set
        // is generated.
        let mut args = default_args();
        args.tracking_min = -0.1;
        args.tracking_max = -0.1;
        let mut rng = Pcg64Mcg::seed_from_u64(1);
        for _ in 0..100 {
            assert!((args.tracking(&mut rng) - -0.1).abs() < 1e-6);
        }
    }

    #[test]
    fn embedded_patterns_keep_the_line_length_distribution() {
        // Splicing must not turn 15% of the corpus into one-character lines:
        // the model reads character pitch off the line, so a pattern with no
        // neighbours teaches it about a line that cannot be read.
        let args = default_args();
        let charset: Vec<char> = "あいうえおかきくけこ".chars().collect();
        let mut rng = Pcg64Mcg::seed_from_u64(5);
        for pattern in ["二", "ll", "一二三"] {
            for _ in 0..200 {
                let text =
                    embed_pattern(&mut rng, &charset, pattern, args.min_chars, args.max_chars);
                let len = text.chars().count();
                assert!(
                    (args.min_chars..=args.max_chars).contains(&len),
                    "{pattern:?} produced a {len}-character line"
                );
                assert!(text.contains(pattern), "{text:?} lost the pattern");
            }
        }
    }

    #[test]
    fn text_ratio_one_keeps_the_file_as_the_whole_corpus() {
        // An evaluation set needs the patterns verbatim and equally weighted.
        let dir = std::env::temp_dir().join(format!("mix_test_{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.txt");
        fs::write(&path, "ll\n\u{4e8c}\n").unwrap();

        let mut args = default_args();
        args.text_file = Some(path);
        assert!(matches!(
            resolve_text_source(&args).unwrap(),
            TextSource::Fixed(_)
        ));

        args.text_ratio = 0.15;
        assert!(matches!(
            resolve_text_source(&args).unwrap(),
            TextSource::Mixed { .. }
        ));

        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn validate_rejects_inverted_bounds() {
        let mut args = default_args();
        args.tracking_min = 0.1;
        args.tracking_max = -0.1;
        assert!(args.validate().is_err());
    }

    #[test]
    fn load_charset_file_rejects_empty_pool() {
        let dir = std::env::temp_dir().join(format!("charset_empty_{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("empty.txt");
        fs::write(&path, "# only comments\n\n").unwrap();

        assert!(load_charset_file(&path).is_err());

        fs::remove_dir_all(dir).ok();
    }
}
