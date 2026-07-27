//! The forward pass, written out against the graph instead of a generic runtime.
//!
//! Mirrors `training/glyph_gnn/model.py`. Every deviation from that file is a
//! bug, so the layer comments here point at what they correspond to rather than
//! re-explaining the architecture.

use std::path::Path;

use glyph_core::GraphSample;
use rayon::prelude::*;

use crate::ops::{Incoming, layer_norm, linear, relu, softmax_rows};
use crate::weights::{HParams, LoadError, Weights};

/// Columns of a `contour_attr` row, matching `crates/glyph-core/src/graph.rs`
/// and the constants at the top of `model.py`.
const C_CX: usize = 0;
const C_CY: usize = 1;
const C_W: usize = 2;
const C_H: usize = 3;
const C_MINX: usize = 4;
const C_MINY: usize = 5;
const C_ARC: usize = 6;
const C_AREA: usize = 7;

/// Width of [`relations_of`].
pub const CONTOUR_RELATIONS: usize = 11;

const EPS: f32 = 1e-6;
const LEAKY_SLOPE: f32 = 0.2;

/// Geometry between two contours, from their feature rows. Mirrors
/// `model.contour_relations`.
fn relations_of(a: &[f32], b: &[f32], out: &mut [f32]) {
    let (ax0, ay0) = (a[C_MINX], a[C_MINY]);
    let (ax1, ay1) = (ax0 + a[C_W], ay0 + a[C_H]);
    let (bx0, by0) = (b[C_MINX], b[C_MINY]);
    let (bx1, by1) = (bx0 + b[C_W], by0 + b[C_H]);

    let over_x = ax1.min(bx1) - ax0.max(bx0);
    let over_y = ay1.min(by1) - ay0.max(by0);
    let span_x = a[C_W].min(b[C_W]).max(EPS);
    let span_y = a[C_H].min(b[C_H]).max(EPS);

    let dx = b[C_CX] - a[C_CX];
    let dy = b[C_CY] - a[C_CY];

    let a_in_b = (ax0 - bx0).min(ay0 - by0).min((bx1 - ax1).min(by1 - ay1));
    let b_in_a = (bx0 - ax0).min(by0 - ay0).min((ax1 - bx1).min(ay1 - by1));

    out[0] = dx;
    out[1] = dy;
    out[2] = (dx * dx + dy * dy).sqrt();
    out[3] = over_x;
    out[4] = over_y;
    out[5] = (over_x / span_x).clamp(-4.0, 1.0);
    out[6] = (over_y / span_y).clamp(-4.0, 1.0);
    out[7] = a_in_b;
    out[8] = b_in_a;
    out[9] = ((a[C_ARC] + EPS) / (b[C_ARC] + EPS)).ln();
    // torch.sign is 0 at 0, which f32::signum is not.
    out[10] = sign(a[C_AREA]) * sign(b[C_AREA]);
}

fn sign(v: f32) -> f32 {
    if v > 0.0 {
        1.0
    } else if v < 0.0 {
        -1.0
    } else {
        0.0
    }
}

fn sigmoid(v: f32) -> f32 {
    1.0 / (1.0 + (-v).exp())
}

/// Wall time per stage, in milliseconds, summed over however many graphs were
/// run. Filled by [`Model::run_timed`].
#[derive(Debug, Default, Clone, Copy)]
pub struct Stages {
    /// Building the incoming-edge index and the contour relation table.
    pub setup: f64,
    pub encoder: f64,
    /// The point-level attention rounds.
    pub gat: f64,
    /// The contour supernode rounds.
    pub contour: f64,
    /// Graph-level pooling and its MLP.
    pub context: f64,
    pub classifier: f64,
}

impl Stages {
    pub fn total(&self) -> f64 {
        self.setup + self.encoder + self.gat + self.contour + self.context + self.classifier
    }

    /// Longest first, so a caller can print them in the order that matters.
    pub fn ranked(&self) -> Vec<(&'static str, f64)> {
        let mut rows = vec![
            ("setup", self.setup),
            ("encoder", self.encoder),
            ("gat", self.gat),
            ("contour", self.contour),
            ("context", self.context),
            ("classifier", self.classifier),
        ];
        rows.sort_by(|a, b| b.1.total_cmp(&a.1));
        rows
    }
}

/// Adds the time `body` takes to `slot`.
fn timed<T>(slot: &mut f64, body: impl FnOnce() -> T) -> T {
    let started = std::time::Instant::now();
    let out = body();
    *slot += started.elapsed().as_secs_f64() * 1e3;
    out
}

pub struct Model {
    w: Weights,
}

impl Model {
    pub fn load(path: &Path) -> Result<Self, LoadError> {
        Ok(Self {
            w: Weights::load(path)?,
        })
    }

    pub fn hparams(&self) -> HParams {
        self.w.hparams
    }

    /// Per-edge probabilities for the edges named in `scored`.
    ///
    /// Message passing always runs over the whole edge list; `scored` only
    /// selects which edges reach the classifier, exactly as the `scored` input
    /// does on the ONNX path.
    pub fn run(&self, g: &GraphSample, scored: &[u32]) -> Vec<f32> {
        self.run_timed(g, scored, &mut Stages::default())
    }

    /// [`run`](Self::run), adding to a per-stage time budget.
    ///
    /// Optimising this runtime without it is guesswork, and guesswork is what
    /// produced the wrong bottleneck twice already.
    pub fn run_timed(&self, g: &GraphSample, scored: &[u32], stages: &mut Stages) -> Vec<f32> {
        let hp = self.w.hparams;
        let n = g.num_nodes as usize;
        let e = g.num_edges();
        let c = g.num_contours as usize;
        let hidden = hp.hidden;
        let (src, dst) = g.edge_index.split_at(e);

        // Geometry does not change between rounds, so the all-pairs table is
        // built once and every layer reads its bias off it.
        let (incoming, relations) = timed(&mut stages.setup, || {
            let incoming = Incoming::build(dst, n);
            let relations = (hp.contours && hp.relations).then(|| {
                let mut table = vec![0.0f32; c * c * CONTOUR_RELATIONS];
                let dim = hp.contour_dim;
                for i in 0..c {
                    for j in 0..c {
                        let at = (i * c + j) * CONTOUR_RELATIONS;
                        relations_of(
                            &g.contour_features[i * dim..(i + 1) * dim],
                            &g.contour_features[j * dim..(j + 1) * dim],
                            &mut table[at..at + CONTOUR_RELATIONS],
                        );
                    }
                }
                table
            });
            (incoming, relations)
        });

        // --- node encoder: Linear -> ReLU -> Linear -> LayerNorm ---
        let mut hx = vec![0.0f32; n * hidden];
        timed(&mut stages.encoder, || {
            let mut first = vec![0.0f32; n * hidden];
            linear(
                &g.node_features,
                n,
                self.w.get("encoder.0.weight"),
                self.w.get("encoder.0.bias"),
                &mut first,
            );
            relu(&mut first);
            linear(
                &first,
                n,
                self.w.get("encoder.2.weight"),
                self.w.get("encoder.2.bias"),
                &mut hx,
            );
            layer_norm(
                &mut hx,
                hidden,
                self.w.get("encoder.3.weight"),
                self.w.get("encoder.3.bias"),
            );
        });

        for depth in 0..hp.layers {
            timed(&mut stages.gat, || {
                self.gat_layer(depth, &mut hx, src, dst, g, &incoming)
            });
            if hp.contours {
                timed(&mut stages.contour, || {
                    self.contour_layer(depth, &mut hx, g, relations.as_deref())
                });
            }
        }

        // --- graph summary: mean and max over every node ---
        let mut summary = None;
        timed(&mut stages.context, || {
            summary = hp.context.then(|| {
                let mut pooled = vec![f32::NEG_INFINITY; 2 * hidden];
                for (i, slot) in pooled[..hidden].iter_mut().enumerate() {
                    *slot = 0.0;
                    for row in 0..n {
                        *slot += hx[row * hidden + i];
                    }
                    *slot /= n.max(1) as f32;
                }
                for row in 0..n {
                    for i in 0..hidden {
                        let v = hx[row * hidden + i];
                        let slot = &mut pooled[hidden + i];
                        if v > *slot {
                            *slot = v;
                        }
                    }
                }
                let mut out = vec![0.0f32; hidden];
                linear(
                    &pooled,
                    1,
                    self.w.get("context.0.weight"),
                    self.w.get("context.0.bias"),
                    &mut out,
                );
                relu(&mut out);
                out
            })
        });

        timed(&mut stages.classifier, || {
            self.classify(g, &hx, summary.as_deref(), relations.as_deref(), scored)
        })
    }

    /// One round of edge-aware attention. Mirrors `EdgeGATLayer.forward`.
    fn gat_layer(
        &self,
        depth: usize,
        hx: &mut [f32],
        src: &[u32],
        dst: &[u32],
        g: &GraphSample,
        incoming: &Incoming,
    ) {
        let hp = self.w.hparams;
        let (n, e) = (g.num_nodes as usize, g.num_edges());
        let (hidden, heads) = (hp.hidden, hp.heads);
        let head_dim = hidden / heads;
        let p = |name: &str| format!("convs.{depth}.{name}");

        let mut h_src = vec![0.0f32; n * hidden];
        let mut h_dst = vec![0.0f32; n * hidden];
        linear(
            hx,
            n,
            self.w.get(&p("lin_src.weight")),
            self.w.get(&p("lin_src.bias")),
            &mut h_src,
        );
        linear(
            hx,
            n,
            self.w.get(&p("lin_dst.weight")),
            self.w.get(&p("lin_dst.bias")),
            &mut h_dst,
        );
        let mut h_edge = vec![0.0f32; e * hidden];
        linear(
            &g.edge_features,
            e,
            self.w.get(&p("lin_edge.weight")),
            self.w.get(&p("lin_edge.bias")),
            &mut h_edge,
        );
        let att = self.w.get(&p("att"));

        // Pass 1: attention logits per edge, and the global maximum the
        // reference subtracts before exponentiating.
        let mut logits = vec![0.0f32; e * heads];
        logits
            .par_chunks_mut(heads)
            .enumerate()
            .for_each(|(edge, row)| {
                let s = src[edge] as usize * hidden;
                let d = dst[edge] as usize * hidden;
                let x = edge * hidden;
                for (head, slot) in row.iter_mut().enumerate() {
                    let base = head * head_dim;
                    let mut total = 0.0;
                    for k in 0..head_dim {
                        let z = h_src[s + base + k] + h_dst[d + base + k] + h_edge[x + base + k];
                        let z = if z >= 0.0 { z } else { LEAKY_SLOPE * z };
                        total += z * att.data[base + k];
                    }
                    *slot = total;
                }
            });
        let peak = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        // Pass 2: each node reads its own incoming edges. Softmax denominator
        // and weighted sum in one walk, so nothing is scattered.
        let mut agg = vec![0.0f32; n * hidden];
        // Scratch is per thread, not per node: a node's degree is single
        // digits, so allocating here would cost more than the arithmetic.
        agg.par_chunks_mut(hidden).enumerate().for_each_init(
            || (vec![0.0f32; heads], Vec::<f32>::new()),
            |(denom, numerators), (node, row)| {
                let edges = incoming.of(node);
                if edges.is_empty() {
                    return;
                }
                denom.fill(0.0);
                numerators.clear();
                for &edge in edges {
                    for (head, total) in denom.iter_mut().enumerate() {
                        let num = (logits[edge as usize * heads + head] - peak).exp();
                        *total += num;
                        numerators.push(num);
                    }
                }
                for (i, &edge) in edges.iter().enumerate() {
                    let s = src[edge as usize] as usize * hidden;
                    for (head, total) in denom.iter().enumerate() {
                        let alpha = numerators[i * heads + head] / (total + 1e-9);
                        let base = head * head_dim;
                        for k in 0..head_dim {
                            row[base + k] += h_src[s + base + k] * alpha;
                        }
                    }
                }
            },
        );

        // Residual around the projected aggregate, then LayerNorm.
        let mut mixed = vec![0.0f32; n * hidden];
        linear(
            &agg,
            n,
            self.w.get(&p("out.weight")),
            self.w.get(&p("out.bias")),
            &mut mixed,
        );
        for (slot, add) in hx.iter_mut().zip(&mixed) {
            *slot += add;
        }
        layer_norm(
            hx,
            hidden,
            self.w.get(&p("norm.weight")),
            self.w.get(&p("norm.bias")),
        );
    }

    /// One round of attention between contour supernodes. Mirrors
    /// `ContourLayer.forward` with `mask=None`, which is the single-graph case.
    fn contour_layer(
        &self,
        depth: usize,
        hx: &mut [f32],
        g: &GraphSample,
        relations: Option<&[f32]>,
    ) {
        let hp = self.w.hparams;
        let (n, c) = (g.num_nodes as usize, g.num_contours as usize);
        let (hidden, heads) = (hp.hidden, hp.heads);
        let head_dim = hidden / heads;
        let p = |name: &str| format!("contour_layers.{depth}.{name}");

        // Pool the points each contour owns, then add the contour's own geometry.
        let mut z = vec![0.0f32; c * hidden];
        {
            let mut counts = vec![0.0f32; c];
            for node in 0..n {
                let contour = g.node_contour_ids[node] as usize;
                counts[contour] += 1.0;
                let (to, from) = (contour * hidden, node * hidden);
                for i in 0..hidden {
                    z[to + i] += hx[from + i];
                }
            }
            for (contour, row) in z.chunks_exact_mut(hidden).enumerate() {
                let inv = 1.0 / counts[contour].max(1.0);
                for v in row.iter_mut() {
                    *v *= inv;
                }
            }
            let mut embedded = vec![0.0f32; c * hidden];
            linear(
                &g.contour_features,
                c,
                self.w.get(&p("embed.weight")),
                self.w.get(&p("embed.bias")),
                &mut embedded,
            );
            for (slot, add) in z.iter_mut().zip(&embedded) {
                *slot += add;
            }
            layer_norm(
                &mut z,
                hidden,
                self.w.get(&p("norm_contour.weight")),
                self.w.get(&p("norm_contour.bias")),
            );
        }

        let mut q = vec![0.0f32; c * hidden];
        let mut k = vec![0.0f32; c * hidden];
        let mut v = vec![0.0f32; c * hidden];
        linear(
            &z,
            c,
            self.w.get(&p("query.weight")),
            self.w.get(&p("query.bias")),
            &mut q,
        );
        linear(
            &z,
            c,
            self.w.get(&p("key.weight")),
            self.w.get(&p("key.bias")),
            &mut k,
        );
        linear(
            &z,
            c,
            self.w.get(&p("value.weight")),
            self.w.get(&p("value.bias")),
            &mut v,
        );

        // Relative geometry enters as a per-head bias on the score, not as
        // extra channels: overlap is a property of the pair the score indexes.
        let bias = relations.map(|table| {
            let mut out = vec![0.0f32; c * c * heads];
            linear(
                table,
                c * c,
                self.w.get(&p("rel_bias.weight")),
                self.w.get(&p("rel_bias.bias")),
                &mut out,
            );
            out
        });

        let scale = (head_dim as f32).powf(-0.5);
        let mut mixed = vec![0.0f32; c * hidden];
        for head in 0..heads {
            let base = head * head_dim;
            let mut scores = vec![0.0f32; c * c];
            for i in 0..c {
                for j in 0..c {
                    let mut total = 0.0;
                    for d in 0..head_dim {
                        total += q[i * hidden + base + d] * k[j * hidden + base + d];
                    }
                    scores[i * c + j] = total * scale
                        + bias.as_ref().map_or(0.0, |b| b[(i * c + j) * heads + head]);
                }
            }
            softmax_rows(&mut scores, c);
            for i in 0..c {
                for j in 0..c {
                    let weight = scores[i * c + j];
                    for d in 0..head_dim {
                        mixed[i * hidden + base + d] += weight * v[j * hidden + base + d];
                    }
                }
            }
        }

        let mut projected = vec![0.0f32; c * hidden];
        linear(
            &mixed,
            c,
            self.w.get(&p("proj.weight")),
            self.w.get(&p("proj.bias")),
            &mut projected,
        );
        for (slot, add) in z.iter_mut().zip(&projected) {
            *slot += add;
        }
        let mut back = vec![0.0f32; c * hidden];
        linear(
            &z,
            c,
            self.w.get(&p("out.weight")),
            self.w.get(&p("out.bias")),
            &mut back,
        );

        for node in 0..n {
            let from = g.node_contour_ids[node] as usize * hidden;
            let to = node * hidden;
            for i in 0..hidden {
                hx[to + i] += back[from + i];
            }
        }
        layer_norm(
            hx,
            hidden,
            self.w.get(&p("norm.weight")),
            self.w.get(&p("norm.bias")),
        );
    }

    /// Edge classifier. Mirrors `GlyphEdgeGNN.edge_logits` plus the sigmoid the
    /// ONNX wrapper adds.
    fn classify(
        &self,
        g: &GraphSample,
        hx: &[f32],
        summary: Option<&[f32]>,
        relations: Option<&[f32]>,
        scored: &[u32],
    ) -> Vec<f32> {
        let hp = self.w.hparams;
        let (src, dst) = g.edge_index.split_at(g.num_edges());
        let (hidden, c) = (hp.hidden, g.num_contours as usize);
        let edge_dim = hp.edge_dim;
        let width = hidden * if summary.is_some() { 4 } else { 3 }
            + edge_dim
            + if relations.is_some() {
                CONTOUR_RELATIONS
            } else {
                0
            };

        let mut feats = vec![0.0f32; scored.len() * width];
        feats
            .par_chunks_mut(width)
            .enumerate()
            .for_each(|(i, row)| {
                let edge = scored[i] as usize;
                let (s, d) = (src[edge] as usize * hidden, dst[edge] as usize * hidden);
                for j in 0..hidden {
                    row[j] = hx[s + j];
                    row[hidden + j] = hx[d + j];
                    row[2 * hidden + j] = (hx[s + j] - hx[d + j]).abs();
                }
                let mut at = 3 * hidden;
                row[at..at + edge_dim]
                    .copy_from_slice(&g.edge_features[edge * edge_dim..(edge + 1) * edge_dim]);
                at += edge_dim;
                if let Some(table) = relations {
                    let ca = g.node_contour_ids[src[edge] as usize] as usize;
                    let cb = g.node_contour_ids[dst[edge] as usize] as usize;
                    let from = (ca * c + cb) * CONTOUR_RELATIONS;
                    row[at..at + CONTOUR_RELATIONS]
                        .copy_from_slice(&table[from..from + CONTOUR_RELATIONS]);
                    at += CONTOUR_RELATIONS;
                }
                if let Some(line) = summary {
                    row[at..at + hidden].copy_from_slice(line);
                }
            });

        let rows = scored.len();
        let mut first = vec![0.0f32; rows * hidden];
        linear(
            &feats,
            rows,
            self.w.get("classifier.0.weight"),
            self.w.get("classifier.0.bias"),
            &mut first,
        );
        relu(&mut first);
        let half = hidden / 2;
        let mut second = vec![0.0f32; rows * half];
        linear(
            &first,
            rows,
            self.w.get("classifier.3.weight"),
            self.w.get("classifier.3.bias"),
            &mut second,
        );
        relu(&mut second);
        let mut out = vec![0.0f32; rows];
        linear(
            &second,
            rows,
            self.w.get("classifier.5.weight"),
            self.w.get("classifier.5.bias"),
            &mut out,
        );
        for v in out.iter_mut() {
            *v = sigmoid(*v);
        }
        out
    }
}
