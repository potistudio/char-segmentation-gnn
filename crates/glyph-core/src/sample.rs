//! Arc-length based resampling of contours into evenly spaced points.

use crate::path::Contour;

/// A point sampled from a contour, carrying its local tangent and the
/// normalized position `t` along the contour (0..1).
#[derive(Debug, Clone, Copy)]
pub struct SampledPoint {
    pub pos: [f32; 2],
    pub tangent: [f32; 2],
    pub t: f32,
}

/// Controls how densely contours are resampled.
#[derive(Debug, Clone, Copy)]
pub struct SampleConfig {
    /// Target spacing between samples, in normalized units (1.0 == upem).
    pub spacing: f32,
    /// Minimum samples per contour, regardless of its length.
    pub min_points: usize,
    /// Maximum samples per contour, to bound graph size on huge glyphs.
    pub max_points: usize,
}

impl Default for SampleConfig {
    fn default() -> Self {
        Self {
            spacing: 0.05,
            min_points: 6,
            max_points: 64,
        }
    }
}

/// Resamples a closed contour into evenly spaced points along its arc length.
/// `scale` converts font units into normalized units (usually 1/upem).
pub fn resample_contour(contour: &Contour, scale: f32, cfg: &SampleConfig) -> Vec<SampledPoint> {
    let pts = &contour.points;
    let n = pts.len();
    if n < 3 {
        return Vec::new();
    }

    // Cumulative arc length over the closed polygon.
    let mut cum = Vec::with_capacity(n + 1);
    cum.push(0.0f32);
    for i in 0..n {
        let a = pts[i];
        let b = pts[(i + 1) % n];
        let d = ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt();
        cum.push(cum[i] + d);
    }
    let total = *cum.last().unwrap();
    if total <= f32::EPSILON {
        return Vec::new();
    }

    let count = ((total * scale / cfg.spacing).round() as usize)
        .clamp(cfg.min_points, cfg.max_points);
    let step = total / count as f32;

    let mut out = Vec::with_capacity(count);
    let mut seg = 0usize;
    for i in 0..count {
        let target = i as f32 * step;
        while seg + 1 < cum.len() - 1 && cum[seg + 1] < target {
            seg += 1;
        }
        let seg_len = (cum[seg + 1] - cum[seg]).max(f32::EPSILON);
        let frac = (target - cum[seg]) / seg_len;
        let a = pts[seg];
        let b = pts[(seg + 1) % n];
        let pos = [a[0] + (b[0] - a[0]) * frac, a[1] + (b[1] - a[1]) * frac];
        let d = [b[0] - a[0], b[1] - a[1]];
        let dl = (d[0] * d[0] + d[1] * d[1]).sqrt().max(f32::EPSILON);
        out.push(SampledPoint {
            pos,
            tangent: [d[0] / dl, d[1] / dl],
            t: target / total,
        });
    }
    out
}
