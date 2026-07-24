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
        // Path compression.
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
/// Nodes joined by edges whose probability clears `threshold` are merged
/// with union-find. Each contour is then assigned to the component that
/// holds the majority of its nodes (individual noisy nodes cannot split a
/// contour), and contours sharing a component form one character.
///
/// Returns contour-id groups sorted by leftmost node for stable output.
pub fn group_characters(g: &GraphSample, probs: &[f32], threshold: f32) -> Vec<Vec<u32>> {
    let n = g.num_nodes as usize;
    let e = g.num_edges();
    let mut uf = UnionFind::new(n);
    let (src, dst) = g.edge_index.split_at(e);
    for i in 0..e {
        if probs[i] >= threshold {
            uf.union(src[i], dst[i]);
        }
    }

    // Majority vote: contour -> component.
    let mut votes: HashMap<(u32, u32), u32> = HashMap::new();
    for node in 0..n as u32 {
        let comp = uf.find(node);
        let contour = g.node_contour_ids[node as usize];
        *votes.entry((contour, comp)).or_insert(0) += 1;
    }
    let mut contour_comp: HashMap<u32, (u32, u32)> = HashMap::new();
    for (&(contour, comp), &count) in &votes {
        let entry = contour_comp.entry(contour).or_insert((comp, 0));
        if count > entry.1 {
            *entry = (comp, count);
        }
    }

    // Component -> contour list.
    let mut groups: HashMap<u32, Vec<u32>> = HashMap::new();
    for (&contour, &(comp, _)) in &contour_comp {
        groups.entry(comp).or_default().push(contour);
    }

    // Sort each group and order groups by leftmost x for readability.
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
