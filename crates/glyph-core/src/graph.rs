//! Graph construction: converts sampled contour points into a node/edge
//! graph with geometric features, using a KD-tree for neighbor search.

use std::collections::HashSet;
use std::num::NonZeroUsize;

use kiddo::{KdTree, SquaredEuclidean};
use serde::{Deserialize, Serialize};

use crate::path::Contour;
use crate::sample::{SampleConfig, resample_contour};

/// Feature vector width per node. Kept as a constant so Rust and Python agree.
///
/// A Fourier encoding of x (sin/cos at 0.5/1/2 em, and again at 1/2/4 em) was
/// tried here to hand the model a basis for character pitch, and measured
/// worse both times: grouping accuracy 0.7875 -> 0.7500 / 0.7583 on ja-6k,
/// reproduced on a second seed. Sample points sit 0.05 em apart, so even the
/// long set turns over fast enough to read as noise to the encoder that feeds
/// every attention layer. The graph-level pooling in the classifier supplies
/// the same global signal without touching the node features.
pub const NODE_DIM: usize = 13;
/// Feature vector width per edge.
pub const EDGE_DIM: usize = 4;
/// Feature vector width per contour supernode.
///
/// One row per contour rather than per sampled point. Four hops of point-level
/// message passing only reach ~0.5 em, so the model needs a level where a
/// whole stroke is a single token and the line is a handful of them.
pub const CONTOUR_DIM: usize = 8;

/// Char id used at inference time when the ground truth is unknown.
pub const UNKNOWN_CHAR: u32 = u32::MAX;

/// A contour placed in layout space, tagged with the character it belongs to.
#[derive(Debug, Clone)]
pub struct ContourInstance {
    /// Points in layout space, font units, y-up.
    pub contour: Contour,
    /// Ground-truth character index (cluster) or [`UNKNOWN_CHAR`].
    pub char_id: u32,
}

/// Tunable parameters for graph construction.
#[derive(Debug, Clone, Copy)]
pub struct GraphConfig {
    pub sample: SampleConfig,
    /// Max neighbors connected per node.
    pub knn: usize,
    /// Connection radius in normalized units (1.0 == upem).
    pub radius: f32,
    /// For each contour pair, connect the closest node pair when their
    /// distance is within this limit (em). kNN caps can block intra-glyph
    /// stroke links even when points are within [`radius`]; bridges ensure
    /// every nearby contour pair is scored by the model.
    pub contour_bridge: f32,
}

impl Default for GraphConfig {
    fn default() -> Self {
        Self {
            sample: SampleConfig::default(),
            knn: 8,
            radius: 0.25,
            contour_bridge: 0.35,
        }
    }
}

/// A serialized training/inference sample. Flat arrays keep the MessagePack
/// payload compact and trivially convertible to tensors on the Python side.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphSample {
    pub num_nodes: u32,
    pub node_dim: u32,
    /// Row-major `[num_nodes * node_dim]`.
    pub node_features: Vec<f32>,
    /// `[2 * num_edges]`: all sources first, then all destinations
    /// (matches PyTorch Geometric `edge_index` layout).
    pub edge_index: Vec<u32>,
    pub edge_dim: u32,
    /// Row-major `[num_edges * edge_dim]`.
    pub edge_features: Vec<f32>,
    pub num_contours: u32,
    pub contour_dim: u32,
    /// Row-major `[num_contours * contour_dim]`, indexed by contour id.
    pub contour_features: Vec<f32>,
    /// `[num_edges]`: 1 if both endpoints belong to the same character.
    pub edge_labels: Vec<u8>,
    /// `[num_nodes]`: ground-truth character id per node (debug/eval).
    pub node_char_ids: Vec<u32>,
    /// `[num_nodes]`: contour index per node, used to group results.
    pub node_contour_ids: Vec<u32>,
    /// Source text (empty at inference time).
    pub text: String,
    /// Font identifier (empty at inference time).
    pub font: String,
}

impl GraphSample {
    pub fn num_edges(&self) -> usize {
        self.edge_index.len() / 2
    }
}

struct NodeInfo {
    pos: [f32; 2],
    feat: [f32; NODE_DIM],
    char_id: u32,
    contour_id: u32,
}

/// Builds a graph from contours laid out in font units.
///
/// Coordinates are normalized by `1/upem` and translated so the leftmost
/// point sits at x=0 (translation invariance across text lengths). The
/// baseline (y=0 in font units) stays at y=0.
pub fn build_graph(contours: &[ContourInstance], upem: f32, cfg: &GraphConfig) -> GraphSample {
    let scale = 1.0 / upem;

    let mut nodes: Vec<NodeInfo> = Vec::new();
    let mut min_x = f32::INFINITY;

    // First pass: sample points and compute per-contour statistics.
    struct Pending {
        samples: Vec<crate::sample::SampledPoint>,
        centroid: [f32; 2],
        bbox_min: [f32; 2],
        bbox_wh: [f32; 2],
        arc_len: f32,
        signed_area: f32,
        char_id: u32,
    }
    let mut pending: Vec<Pending> = Vec::new();
    for inst in contours {
        let samples = resample_contour(&inst.contour, scale, &cfg.sample);
        if samples.is_empty() {
            continue;
        }
        let c = inst.contour.centroid();
        let (bmin, bmax) = inst.contour.bbox();
        min_x = min_x.min(bmin[0]);
        pending.push(Pending {
            samples,
            centroid: [c[0], c[1]],
            bbox_min: [bmin[0], bmin[1]],
            bbox_wh: [bmax[0] - bmin[0], bmax[1] - bmin[1]],
            arc_len: inst.contour.arc_length(),
            signed_area: inst.contour.signed_area(),
            char_id: inst.char_id,
        });
    }
    if !min_x.is_finite() {
        min_x = 0.0;
    }

    for (cid, p) in pending.iter().enumerate() {
        for s in &p.samples {
            let x = (s.pos[0] - min_x) * scale;
            let y = s.pos[1] * scale;
            let cx = (p.centroid[0] - min_x) * scale;
            let cy = p.centroid[1] * scale;
            let phase = s.t * std::f32::consts::TAU;
            let feat = [
                x,
                y,
                s.tangent[0],
                s.tangent[1],
                x - cx,
                y - cy,
                cx,
                cy,
                p.bbox_wh[0] * scale,
                p.bbox_wh[1] * scale,
                p.arc_len * scale,
                phase.cos(),
                phase.sin(),
            ];
            nodes.push(NodeInfo {
                pos: [x, y],
                feat,
                char_id: p.char_id,
                contour_id: cid as u32,
            });
        }
    }

    // Neighbor search over normalized positions.
    let mut tree: KdTree<f32, 2> = KdTree::new();
    for (i, n) in nodes.iter().enumerate() {
        tree.add(&n.pos, i as u64);
    }

    let mut pairs: HashSet<(u32, u32)> = HashSet::new();
    if nodes.len() > 1 {
        let k = NonZeroUsize::new(cfg.knn + 1).unwrap();
        let radius_sq = cfg.radius * cfg.radius;
        for (i, n) in nodes.iter().enumerate() {
            for hit in tree.nearest_n_within::<SquaredEuclidean>(&n.pos, radius_sq, k, false) {
                let j = hit.item as usize;
                if j == i {
                    continue;
                }
                let (a, b) = if i < j { (i, j) } else { (j, i) };
                pairs.insert((a as u32, b as u32));
            }
        }
    }

    // Bridge edges: one closest-node link per contour pair within contour_bridge.
    // kNN caps often omit these when dense local neighbours fill the budget.
    if cfg.contour_bridge > 0.0 {
        let contour_count = pending.len();
        let mut contour_nodes: Vec<Vec<usize>> = vec![Vec::new(); contour_count];
        for (i, n) in nodes.iter().enumerate() {
            contour_nodes[n.contour_id as usize].push(i);
        }
        let bridge_sq = cfg.contour_bridge * cfg.contour_bridge;
        for ca in 0..contour_count {
            for cb in (ca + 1)..contour_count {
                let mut best_dist_sq = f32::INFINITY;
                let mut best_pair: Option<(usize, usize)> = None;
                for &i in &contour_nodes[ca] {
                    for &j in &contour_nodes[cb] {
                        let dx = nodes[j].pos[0] - nodes[i].pos[0];
                        let dy = nodes[j].pos[1] - nodes[i].pos[1];
                        let d2 = dx * dx + dy * dy;
                        if d2 < best_dist_sq {
                            best_dist_sq = d2;
                            best_pair = Some((i, j));
                        }
                    }
                }
                if best_dist_sq <= bridge_sq {
                    if let Some((i, j)) = best_pair {
                        let (a, b) = if i < j { (i, j) } else { (j, i) };
                        pairs.insert((a as u32, b as u32));
                    }
                }
            }
        }
    }

    let num_edges = pairs.len() * 2;
    let mut src = Vec::with_capacity(num_edges);
    let mut dst = Vec::with_capacity(num_edges);
    let mut edge_features = Vec::with_capacity(num_edges * EDGE_DIM);
    let mut edge_labels = Vec::with_capacity(num_edges);

    let mut sorted_pairs: Vec<(u32, u32)> = pairs.into_iter().collect();
    sorted_pairs.sort_unstable();
    for (a, b) in sorted_pairs {
        for (s, d) in [(a, b), (b, a)] {
            let ns = &nodes[s as usize];
            let nd = &nodes[d as usize];
            let dx = nd.pos[0] - ns.pos[0];
            let dy = nd.pos[1] - ns.pos[1];
            let dist = (dx * dx + dy * dy).sqrt();
            let same_contour = (ns.contour_id == nd.contour_id) as u32 as f32;
            src.push(s);
            dst.push(d);
            edge_features.extend_from_slice(&[dx, dy, dist, same_contour]);
            let label = (ns.char_id != UNKNOWN_CHAR && ns.char_id == nd.char_id) as u8;
            edge_labels.push(label);
        }
    }

    let mut edge_index = src;
    edge_index.extend_from_slice(&dst);

    // One row per contour, in contour-id order. Areas are scaled by the square
    // of the linear scale so they stay in em^2.
    let mut contour_features = Vec::with_capacity(pending.len() * CONTOUR_DIM);
    for p in &pending {
        contour_features.extend_from_slice(&[
            (p.centroid[0] - min_x) * scale,
            p.centroid[1] * scale,
            p.bbox_wh[0] * scale,
            p.bbox_wh[1] * scale,
            (p.bbox_min[0] - min_x) * scale,
            p.bbox_min[1] * scale,
            p.arc_len * scale,
            p.signed_area * scale * scale,
        ]);
    }

    let mut node_features = Vec::with_capacity(nodes.len() * NODE_DIM);
    let mut node_char_ids = Vec::with_capacity(nodes.len());
    let mut node_contour_ids = Vec::with_capacity(nodes.len());
    for n in &nodes {
        node_features.extend_from_slice(&n.feat);
        node_char_ids.push(n.char_id);
        node_contour_ids.push(n.contour_id);
    }

    GraphSample {
        num_nodes: nodes.len() as u32,
        node_dim: NODE_DIM as u32,
        node_features,
        edge_index,
        edge_dim: EDGE_DIM as u32,
        edge_features,
        num_contours: pending.len() as u32,
        contour_dim: CONTOUR_DIM as u32,
        contour_features,
        edge_labels,
        node_char_ids,
        node_contour_ids,
        text: String::new(),
        font: String::new(),
    }
}
