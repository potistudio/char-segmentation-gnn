//! Postprocessing: thresholded edge probabilities -> union-find connected
//! components -> contour groups representing character segments.

use std::collections::HashMap;

use glyph_core::GraphSample;

struct UnionFind {
    parent: Vec<u32>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n as u32).collect(),
        }
    }

    fn find(&mut self, x: u32) -> u32 {
        let mut root = x;
        while self.parent[root as usize] != root {
            root = self.parent[root as usize];
        }
        let mut cur = x;
        while self.parent[cur as usize] != root {
            let next = self.parent[cur as usize];
            self.parent[cur as usize] = root;
            cur = next;
        }
        root
    }

    fn union(&mut self, a: u32, b: u32) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra != rb {
            self.parent[ra as usize] = rb;
        }
    }
}

/// Groups contours into character segments.
///
/// Cross-contour edges are aggregated by max probability per contour pair.
/// Contours whose pair score clears `threshold` are merged with union-find.
/// Each contour is an atomic unit (intra-contour connectivity is implicit).
///
/// Returns contour-id groups sorted by leftmost node for stable output.
pub fn group_characters(g: &GraphSample, probs: &[f32], threshold: f32) -> Vec<Vec<u32>> {
    let n = g.num_nodes as usize;
    if n == 0 {
        return Vec::new();
    }
    let e = g.num_edges();
    let num_contours = g
        .node_contour_ids
        .iter()
        .copied()
        .max()
        .map(|m| m + 1)
        .unwrap_or(0) as usize;
    let (src, dst) = g.edge_index.split_at(e);

    let mut pair_prob: HashMap<(u32, u32), f32> = HashMap::new();
    for i in 0..e {
        let ca = g.node_contour_ids[src[i] as usize];
        let cb = g.node_contour_ids[dst[i] as usize];
        if ca == cb {
            continue;
        }
        let key = if ca < cb { (ca, cb) } else { (cb, ca) };
        let entry = pair_prob.entry(key).or_insert(0.0);
        if probs[i] > *entry {
            *entry = probs[i];
        }
    }

    let mut uf = UnionFind::new(num_contours);
    for (&(a, b), &prob) in &pair_prob {
        if prob >= threshold {
            uf.union(a, b);
        }
    }

    let mut groups: HashMap<u32, Vec<u32>> = HashMap::new();
    for contour in 0..num_contours as u32 {
        groups.entry(uf.find(contour)).or_default().push(contour);
    }

    let node_dim = g.node_dim as usize;
    let contour_min_x = |cids: &[u32]| -> f32 {
        let mut min_x = f32::INFINITY;
        for node in 0..n {
            if cids.contains(&g.node_contour_ids[node]) {
                min_x = min_x.min(g.node_features[node * node_dim]);
            }
        }
        min_x
    };
    let mut out: Vec<Vec<u32>> = groups
        .into_values()
        .map(|mut v| {
            v.sort_unstable();
            v
        })
        .collect();
    out.sort_by(|a, b| contour_min_x(a).total_cmp(&contour_min_x(b)));
    out
}

/// Ground-truth contour groups derived from stored char ids (for eval).
pub fn ground_truth_groups(g: &GraphSample) -> Vec<Vec<u32>> {
    let mut by_char: HashMap<u32, Vec<u32>> = HashMap::new();
    let n = g.num_nodes as usize;
    for node in 0..n {
        let cid = g.node_contour_ids[node];
        let ch = g.node_char_ids[node];
        let v = by_char.entry(ch).or_default();
        if !v.contains(&cid) {
            v.push(cid);
        }
    }
    let mut out: Vec<Vec<u32>> = by_char
        .into_values()
        .map(|mut v| {
            v.sort_unstable();
            v
        })
        .collect();
    out.sort();
    out
}

/// Set equality of groupings, ignoring order.
pub fn groups_equal(a: &[Vec<u32>], b: &[Vec<u32>]) -> bool {
    let mut sa: Vec<Vec<u32>> = a.to_vec();
    let mut sb: Vec<Vec<u32>> = b.to_vec();
    sa.sort();
    sb.sort();
    sa == sb
}
