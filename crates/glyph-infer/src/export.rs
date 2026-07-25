//! JSON export for the interactive GUI: normalized contours, graph tensors,
//! and per-edge probabilities from the ONNX model.

use std::path::Path;

use anyhow::Result;
use glyph_core::{
    ContourInstance, GraphConfig, GraphSample, LayoutConfig, build_graph, layout_text,
};
use rand::SeedableRng;
use rand_pcg::Pcg64Mcg;
use serde::Serialize;

use crate::Engine;

#[derive(Serialize)]
pub struct ExportPayload {
    pub text: String,
    pub contours: Vec<Vec<[f32; 2]>>,
    pub nodes: NodesOut,
    pub edges: EdgesOut,
    pub timing_ms: TimingOut,
}

#[derive(Serialize)]
pub struct NodesOut {
    pub x: Vec<f32>,
    pub y: Vec<f32>,
    pub contour_id: Vec<u32>,
}

#[derive(Serialize)]
pub struct EdgesOut {
    pub src: Vec<u32>,
    pub dst: Vec<u32>,
    pub prob: Vec<f32>,
}

#[derive(Serialize)]
pub struct TimingOut {
    pub preprocess: f64,
    pub inference: f64,
}

/// Normalizes contour polylines into the same space as [`build_graph`] nodes.
fn normalize_contours(contours: &[ContourInstance], upem: f32) -> Vec<Vec<[f32; 2]>> {
    let scale = 1.0 / upem;
    let mut min_x = f32::INFINITY;
    for inst in contours {
        let (bmin, _) = inst.contour.bbox();
        min_x = min_x.min(bmin[0]);
    }
    if !min_x.is_finite() {
        min_x = 0.0;
    }
    contours
        .iter()
        .map(|inst| {
            inst.contour
                .points
                .iter()
                .map(|p| [(p[0] - min_x) * scale, p[1] * scale])
                .collect()
        })
        .collect()
}

fn graph_tensors(g: &GraphSample) -> (NodesOut, EdgesOut) {
    let n = g.num_nodes as usize;
    let e = g.num_edges();
    let node_dim = g.node_dim as usize;
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for node in 0..n {
        let base = node * node_dim;
        x.push(g.node_features[base]);
        y.push(g.node_features[base + 1]);
    }
    let (src, dst) = g.edge_index.split_at(e);
    let nodes = NodesOut {
        x,
        y,
        contour_id: g.node_contour_ids.clone(),
    };
    let edges = EdgesOut {
        src: src.to_vec(),
        dst: dst.to_vec(),
        prob: Vec::new(),
    };
    (nodes, edges)
}

pub fn run_export(
    engine: &mut Engine,
    font: &Path,
    text: &str,
    tracking: f32,
    graph_cfg: &GraphConfig,
) -> Result<ExportPayload> {
    let font_data = std::fs::read(font)?;
    let mut rng = Pcg64Mcg::seed_from_u64(0);
    let layout_cfg = LayoutConfig {
        tracking,
        ..LayoutConfig::default()
    };

    let t0 = std::time::Instant::now();
    let (contours, upem) = layout_text(&font_data, text, &layout_cfg, &mut rng)?;
    let graph = build_graph(&contours, upem, graph_cfg);
    let t_pre = t0.elapsed();

    let t1 = std::time::Instant::now();
    let probs = engine.run(&graph)?;
    let t_inf = t1.elapsed();

    let (nodes, mut edges) = graph_tensors(&graph);
    edges.prob = probs;

    Ok(ExportPayload {
        text: text.to_string(),
        contours: normalize_contours(&contours, upem),
        nodes,
        edges,
        timing_ms: TimingOut {
            preprocess: t_pre.as_secs_f64() * 1e3,
            inference: t_inf.as_secs_f64() * 1e3,
        },
    })
}
