//! Inference engine (Phase 3) and evaluation harness (Phase 4).
//!
//! Modes:
//!   demo  — lay out text with a font, run the ONNX model, print character groups
//!   eval  — replay a MessagePack shard, report precision/recall/F1 and timings
//!
//! Preprocessing reuses glyph-core so it is identical to dataset generation.

mod postprocess;

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use ort::execution_providers::CUDAExecutionProvider;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Tensor;
use rand::SeedableRng;
use rand_pcg::Pcg64Mcg;

use glyph_core::{GraphConfig, GraphSample, LayoutConfig, build_graph, layout_text};
use postprocess::group_characters;

#[derive(Parser, Debug)]
#[command(name = "glyph-infer", about = "GNN-based glyph segment splitter")]
struct Args {
    /// Path to the exported ONNX model.
    #[arg(long)]
    model: PathBuf,

    /// Edge probability threshold for the same-character decision.
    #[arg(long, default_value_t = 0.5)]
    threshold: f32,

    /// Force CPU execution (skip the CUDA execution provider).
    #[arg(long)]
    cpu: bool,

    #[command(subcommand)]
    cmd: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Split a text string rendered with a font file.
    Demo {
        #[arg(long)]
        font: PathBuf,
        #[arg(long)]
        text: String,
        /// Extra tracking in em units (negative squeezes glyphs together).
        #[arg(long, default_value_t = 0.0, allow_negative_numbers = true)]
        tracking: f32,
    },
    /// Evaluate accuracy and latency on a generated MessagePack shard.
    Eval {
        #[arg(long)]
        shard: PathBuf,
        /// Max samples to evaluate (0 = all).
        #[arg(long, default_value_t = 0)]
        limit: usize,
    },
}

/// ort's error type is not `Send + Sync`, so it cannot flow through anyhow
/// directly; stringify it instead.
fn ort_err<R>(e: ort::Error<R>) -> anyhow::Error {
    anyhow::anyhow!("{e}")
}

struct Engine {
    session: Session,
    threshold: f32,
}

impl Engine {
    fn new(model: &PathBuf, use_cuda: bool, threshold: f32) -> Result<Self> {
        let base = || -> Result<_> {
            Session::builder()
                .map_err(ort_err)?
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .map_err(ort_err)?
                .with_intra_threads(4)
                .map_err(ort_err)
        };

        // Try CUDA strictly first so a registration failure is visible
        // instead of silently degrading to CPU.
        if use_cuda {
            let attempt = base()?
                .with_execution_providers([CUDAExecutionProvider::default()
                    .build()
                    .error_on_failure()])
                .map_err(ort_err)
                .and_then(|mut b| b.commit_from_file(model).map_err(ort_err));
            match attempt {
                Ok(session) => {
                    eprintln!("execution provider: CUDA");
                    return Ok(Self { session, threshold });
                }
                Err(e) => {
                    eprintln!("warning: CUDA EP unavailable ({e}); falling back to CPU");
                }
            }
        }

        let session = base()?
            .commit_from_file(model)
            .map_err(ort_err)
            .with_context(|| format!("loading {}", model.display()))?;
        eprintln!("execution provider: CPU");
        Ok(Self { session, threshold })
    }

    /// Runs the model on a prebuilt graph. Returns per-edge probabilities.
    fn run(&mut self, g: &GraphSample) -> Result<Vec<f32>> {
        let n = g.num_nodes as usize;
        let e = g.num_edges();
        let node_dim = g.node_dim as usize;
        let edge_dim = g.edge_dim as usize;

        // i64 edge index as required by the ONNX graph (torch.long).
        let edge_index: Vec<i64> = g.edge_index.iter().map(|&v| v as i64).collect();

        let nodes =
            Tensor::from_array(([n, node_dim], g.node_features.clone())).map_err(ort_err)?;
        let edges = Tensor::from_array(([2usize, e], edge_index)).map_err(ort_err)?;
        let eattr =
            Tensor::from_array(([e, edge_dim], g.edge_features.clone())).map_err(ort_err)?;

        let outputs = self
            .session
            .run(ort::inputs![
                "node_features" => nodes,
                "edge_index" => edges,
                "edge_features" => eattr,
            ])
            .map_err(ort_err)?;
        let (_, probs) = outputs["edge_probs"]
            .try_extract_tensor::<f32>()
            .map_err(ort_err)?;
        Ok(probs.to_vec())
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let mut engine = Engine::new(&args.model, !args.cpu, args.threshold)?;

    match &args.cmd {
        Command::Demo {
            font,
            text,
            tracking,
        } => demo(&mut engine, font, text, *tracking),
        Command::Eval { shard, limit } => eval(&mut engine, shard, *limit),
    }
}

fn demo(engine: &mut Engine, font: &PathBuf, text: &str, tracking: f32) -> Result<()> {
    let font_data = std::fs::read(font)?;
    let mut rng = Pcg64Mcg::seed_from_u64(0);
    let layout_cfg = LayoutConfig {
        tracking,
        ..LayoutConfig::default()
    };

    // --- preprocess ---
    let t0 = Instant::now();
    let (contours, upem) =
        layout_text(&font_data, text, &layout_cfg, &mut rng).context("layout failed")?;
    let graph = build_graph(&contours, upem, &GraphConfig::default());
    let t_pre = t0.elapsed();

    // --- inference ---
    let t1 = Instant::now();
    let probs = engine.run(&graph)?;
    let t_inf = t1.elapsed();

    // --- postprocess ---
    let t2 = Instant::now();
    let groups = group_characters(&graph, &probs, engine.threshold);
    let t_post = t2.elapsed();

    println!(
        "text: {:?} -> {} nodes, {} edges, {} contours",
        text,
        graph.num_nodes,
        graph.num_edges(),
        contours.len()
    );
    println!("detected {} character group(s):", groups.len());
    for (i, contour_ids) in groups.iter().enumerate() {
        println!("  char {i}: contours {contour_ids:?}");
    }
    println!(
        "timing: preprocess {:.2}ms | inference {:.2}ms | postprocess {:.2}ms | total {:.2}ms",
        t_pre.as_secs_f64() * 1e3,
        t_inf.as_secs_f64() * 1e3,
        t_post.as_secs_f64() * 1e3,
        (t_pre + t_inf + t_post).as_secs_f64() * 1e3,
    );
    Ok(())
}

fn eval(engine: &mut Engine, shard: &PathBuf, limit: usize) -> Result<()> {
    let raw = std::fs::read(shard)?;
    let mut samples: Vec<GraphSample> = rmp_serde::from_slice(&raw)?;
    if limit > 0 {
        samples.truncate(limit);
    }
    println!(
        "evaluating {} samples from {}",
        samples.len(),
        shard.display()
    );

    let (mut tp, mut fp, mut fnn, mut tn) = (0u64, 0u64, 0u64, 0u64);
    let (mut sum_inf, mut sum_post) = (0.0f64, 0.0f64);
    let mut exact_groups = 0usize;

    // Warm-up run so CUDA kernel compilation does not skew the timings.
    if let Some(first) = samples.first() {
        let _ = engine.run(first)?;
    }

    for g in &samples {
        let t1 = Instant::now();
        let probs = engine.run(g)?;
        sum_inf += t1.elapsed().as_secs_f64() * 1e3;

        let t2 = Instant::now();
        let groups = group_characters(g, &probs, engine.threshold);
        sum_post += t2.elapsed().as_secs_f64() * 1e3;

        for (p, &y) in probs.iter().zip(&g.edge_labels) {
            let pred = *p >= engine.threshold;
            match (pred, y == 1) {
                (true, true) => tp += 1,
                (true, false) => fp += 1,
                (false, true) => fnn += 1,
                (false, false) => tn += 1,
            }
        }

        let truth = postprocess::ground_truth_groups(g);
        if postprocess::groups_equal(&groups, &truth) {
            exact_groups += 1;
        }
    }

    let n = samples.len().max(1) as f64;
    let precision = tp as f64 / (tp + fp).max(1) as f64;
    let recall = tp as f64 / (tp + fnn).max(1) as f64;
    let f1 = 2.0 * precision * recall / (precision + recall).max(1e-12);
    let prec_n = tn as f64 / (tn + fnn).max(1) as f64;
    let rec_n = tn as f64 / (tn + fp).max(1) as f64;
    let f1_n = 2.0 * prec_n * rec_n / (prec_n + rec_n).max(1e-12);

    println!("edge metrics (positive = same character):");
    println!("  precision {precision:.4}  recall {recall:.4}  F1 {f1:.4}");
    println!("  negative-class F1 (boundaries): {f1_n:.4}");
    println!(
        "exact grouping accuracy: {:.2}% ({exact_groups}/{})",
        exact_groups as f64 / n * 100.0,
        samples.len()
    );
    println!(
        "latency per sample: inference {:.2}ms | postprocess {:.3}ms",
        sum_inf / n,
        sum_post / n
    );
    Ok(())
}
