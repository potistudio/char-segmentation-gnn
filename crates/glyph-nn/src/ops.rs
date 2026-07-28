//! Dense primitives and the incoming-edge index the aggregations read.

use rayon::prelude::*;

use crate::weights::View;

/// LayerNorm's epsilon in torch, which is what these weights were trained with.
const LAYER_NORM_EPS: f32 = 1e-5;

/// Rows per task in [`linear`] and [`layer_norm`]. `matrixmultiply::sgemm` is
/// single threaded, so the split has to happen here; below this the rayon
/// hand-off costs more than the work, which matters because the contour layers
/// call these with a few dozen rows while the classifier calls them with
/// thousands.
const ROW_BLOCK: usize = 128;

/// `y = x W^T + b`, with `x` as `[m, k]` and `w` as `[n, k]` -- torch's
/// `nn.Linear` layout, so the dump needs no transposing.
///
/// Splitting over `m` is exact rather than approximate: each output row depends
/// on one input row, so the blocks do not overlap and the sum order inside a
/// row is unchanged.
pub fn linear(x: &[f32], m: usize, w: View<'_>, b: View<'_>, out: &mut [f32]) {
    let (n, k) = (w.rows, w.cols);
    debug_assert_eq!(x.len(), m * k);
    debug_assert_eq!(out.len(), m * n);
    if m == 0 {
        return;
    }
    if m <= ROW_BLOCK {
        gemm_block(x, m, w, b, out);
        return;
    }
    out.par_chunks_mut(ROW_BLOCK * n)
        .zip(x.par_chunks(ROW_BLOCK * k))
        .for_each(|(out_block, x_block)| {
            gemm_block(x_block, x_block.len() / k, w, b, out_block);
        });
}

fn gemm_block(x: &[f32], m: usize, w: View<'_>, b: View<'_>, out: &mut [f32]) {
    let (n, k) = (w.rows, w.cols);
    for row in out.chunks_exact_mut(n) {
        row.copy_from_slice(&b.data[..n]);
    }
    // SAFETY: shapes are checked by the caller's debug_asserts and the block
    // split above; the strides describe `w` transposed without moving it.
    unsafe {
        matrixmultiply::sgemm(
            m,
            k,
            n,
            1.0,
            x.as_ptr(),
            k as isize,
            1,
            w.data.as_ptr(),
            1,
            w.stride as isize,
            1.0,
            out.as_mut_ptr(),
            n as isize,
            1,
        );
    }
}

/// In-place LayerNorm over each row of `[m, n]`.
///
/// Rows are independent, so the split changes nothing about the arithmetic.
/// It is worth doing because this runs eight times over `[nodes, hidden]` in a
/// four-layer model -- once per attention round and once per contour round.
pub fn layer_norm(x: &mut [f32], n: usize, w: View<'_>, b: View<'_>) {
    let one_row = |row: &mut [f32]| {
        let mean = row.iter().sum::<f32>() / n as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
        let scale = 1.0 / (var + LAYER_NORM_EPS).sqrt();
        for (i, v) in row.iter_mut().enumerate() {
            *v = (*v - mean) * scale * w.data[i] + b.data[i];
        }
    };
    if x.len() / n <= ROW_BLOCK {
        x.chunks_exact_mut(n).for_each(one_row);
    } else {
        x.par_chunks_mut(ROW_BLOCK * n)
            .for_each(|block| block.chunks_exact_mut(n).for_each(one_row));
    }
}

pub fn relu(x: &mut [f32]) {
    for v in x {
        *v = v.max(0.0);
    }
}

/// In-place softmax over each row of `[m, n]`.
pub fn softmax_rows(x: &mut [f32], n: usize) {
    for row in x.chunks_exact_mut(n) {
        let peak = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut total = 0.0;
        for v in row.iter_mut() {
            *v = (*v - peak).exp();
            total += *v;
        }
        let inv = 1.0 / total;
        for v in row {
            *v *= inv;
        }
    }
}

/// Incoming edges grouped by destination node.
///
/// This is the whole reason for a native runtime. Message passing writes each
/// edge's message into its destination row, which as a scatter is a random
/// write that cannot be split across threads without atomics -- onnxruntime's
/// ScatterElements was 43% of CPU time and did not parallelise at all. Indexed
/// the other way it becomes a read: a node walks its own incoming edges and
/// accumulates, so the write is local, the threads are independent, and the
/// only random access left is the gather of source rows.
pub struct Incoming {
    /// `[num_nodes + 1]` prefix sums into [`edges`].
    pub start: Vec<u32>,
    /// Edge ids, grouped by destination.
    pub edges: Vec<u32>,
}

impl Incoming {
    pub fn build(dst: &[u32], num_nodes: usize) -> Self {
        let mut start = vec![0u32; num_nodes + 2];
        for &d in dst {
            start[d as usize + 2] += 1;
        }
        for i in 2..start.len() {
            start[i] += start[i - 1];
        }
        let mut edges = vec![0u32; dst.len()];
        for (edge, &d) in dst.iter().enumerate() {
            let slot = &mut start[d as usize + 1];
            edges[*slot as usize] = edge as u32;
            *slot += 1;
        }
        start.truncate(num_nodes + 1);
        Self { start, edges }
    }

    pub fn of(&self, node: usize) -> &[u32] {
        let (a, b) = (self.start[node] as usize, self.start[node + 1] as usize);
        &self.edges[a..b]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn incoming_groups_every_edge_under_its_destination() {
        // Edge e lands on dst[e]; node 1 gets two, node 3 none.
        let dst = [2u32, 1, 0, 1, 2];
        let inc = Incoming::build(&dst, 4);
        assert_eq!(inc.of(0), &[2]);
        assert_eq!(inc.of(1), &[1, 3]);
        assert_eq!(inc.of(2), &[0, 4]);
        assert!(inc.of(3).is_empty());
        assert_eq!(inc.edges.len(), dst.len());
    }

    #[test]
    fn linear_matches_a_hand_computed_product() {
        // x is [2, 3], w is [2, 3] so y is [2, 2].
        let x = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let wd = [1.0f32, 0.0, -1.0, 0.5, 0.5, 0.5];
        let bd = [10.0f32, -1.0];
        let w = View::new(&wd, 2, 3);
        let b = View::new(&bd, 1, 2);
        let mut out = [0.0f32; 4];
        linear(&x, 2, w, b, &mut out);
        // row0: [1-3+10, 0.5+1+1.5-1] ; row1: [4-6+10, 2+2.5+3-1]
        assert_eq!(out, [8.0, 2.0, 8.0, 6.5]);
    }

    #[test]
    fn layer_norm_centres_and_scales() {
        let mut x = [1.0f32, 2.0, 3.0, 4.0];
        let wd = [1.0f32; 2];
        let bd = [0.0f32; 2];
        let w = View::new(&wd, 1, 2);
        let b = View::new(&bd, 1, 2);
        layer_norm(&mut x, 2, w, b);
        // Each row is two values, so they normalise to -1 and +1.
        for pair in x.chunks_exact(2) {
            assert!((pair[0] + 1.0).abs() < 1e-3, "{pair:?}");
            assert!((pair[1] - 1.0).abs() < 1e-3, "{pair:?}");
        }
    }
}
