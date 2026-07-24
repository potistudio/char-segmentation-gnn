//! Glyph outline extraction: flattens quadratic/cubic Bezier segments into
//! polyline contours in font units.

use ttf_parser::OutlineBuilder;

/// A closed contour represented as a polyline (font units, y-up).
#[derive(Debug, Clone, Default)]
pub struct Contour {
    pub points: Vec<[f32; 2]>,
}

impl Contour {
    pub fn arc_length(&self) -> f32 {
        let n = self.points.len();
        if n < 2 {
            return 0.0;
        }
        let mut len = 0.0;
        for i in 0..n {
            let a = self.points[i];
            let b = self.points[(i + 1) % n];
            len += ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt();
        }
        len
    }

    pub fn centroid(&self) -> [f32; 2] {
        let n = self.points.len().max(1) as f32;
        let mut c = [0.0f32; 2];
        for p in &self.points {
            c[0] += p[0];
            c[1] += p[1];
        }
        [c[0] / n, c[1] / n]
    }

    pub fn bbox(&self) -> ([f32; 2], [f32; 2]) {
        let mut min = [f32::INFINITY; 2];
        let mut max = [f32::NEG_INFINITY; 2];
        for p in &self.points {
            for k in 0..2 {
                min[k] = min[k].min(p[k]);
                max[k] = max[k].max(p[k]);
            }
        }
        (min, max)
    }
}

/// Number of line segments used to flatten each curve segment.
const CURVE_STEPS: usize = 8;

/// Collects glyph outlines into flattened [`Contour`]s.
pub struct ContourBuilder {
    contours: Vec<Contour>,
    current: Vec<[f32; 2]>,
    last: [f32; 2],
}

impl ContourBuilder {
    pub fn new() -> Self {
        Self {
            contours: Vec::new(),
            current: Vec::new(),
            last: [0.0, 0.0],
        }
    }

    pub fn finish(mut self) -> Vec<Contour> {
        self.flush();
        self.contours
    }

    fn flush(&mut self) {
        if self.current.len() >= 3 {
            self.contours.push(Contour {
                points: std::mem::take(&mut self.current),
            });
        } else {
            self.current.clear();
        }
    }
}

impl Default for ContourBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl OutlineBuilder for ContourBuilder {
    fn move_to(&mut self, x: f32, y: f32) {
        self.flush();
        self.last = [x, y];
        self.current.push([x, y]);
    }

    fn line_to(&mut self, x: f32, y: f32) {
        self.last = [x, y];
        self.current.push([x, y]);
    }

    fn quad_to(&mut self, x1: f32, y1: f32, x: f32, y: f32) {
        let p0 = self.last;
        for i in 1..=CURVE_STEPS {
            let t = i as f32 / CURVE_STEPS as f32;
            let u = 1.0 - t;
            let px = u * u * p0[0] + 2.0 * u * t * x1 + t * t * x;
            let py = u * u * p0[1] + 2.0 * u * t * y1 + t * t * y;
            self.current.push([px, py]);
        }
        self.last = [x, y];
    }

    fn curve_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x: f32, y: f32) {
        let p0 = self.last;
        for i in 1..=CURVE_STEPS {
            let t = i as f32 / CURVE_STEPS as f32;
            let u = 1.0 - t;
            let px =
                u.powi(3) * p0[0] + 3.0 * u * u * t * x1 + 3.0 * u * t * t * x2 + t.powi(3) * x;
            let py =
                u.powi(3) * p0[1] + 3.0 * u * u * t * y1 + 3.0 * u * t * t * y2 + t.powi(3) * y;
            self.current.push([px, py]);
        }
        self.last = [x, y];
    }

    fn close(&mut self) {
        self.flush();
    }
}
