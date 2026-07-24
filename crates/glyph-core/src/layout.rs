//! Text shaping via rustybuzz and glyph path extraction via ttf-parser,
//! with optional geometric augmentation for synthetic dataset generation.

use rand::{Rng, RngExt};
use rustybuzz::ttf_parser::GlyphId;
use rustybuzz::{Face, UnicodeBuffer};

use crate::graph::ContourInstance;
use crate::path::ContourBuilder;

/// Augmentation parameters. All distances are in em units (1.0 == upem).
/// The default performs a clean, deterministic layout (inference mode).
#[derive(Debug, Clone, Copy)]
pub struct LayoutConfig {
    /// Extra advance added between glyphs. Negative values squeeze glyphs
    /// together and create the intersecting paths we want to learn on.
    pub tracking: f32,
    /// Max absolute random vertical shift applied per glyph.
    pub baseline_jitter: f32,
    /// Horizontal scale factor applied to the whole line (aspect change).
    pub aspect_x: f32,
    /// Max absolute uniform jitter added to every outline point.
    pub point_noise: f32,
}

impl Default for LayoutConfig {
    fn default() -> Self {
        Self {
            tracking: 0.0,
            baseline_jitter: 0.0,
            aspect_x: 1.0,
            point_noise: 0.0,
        }
    }
}

/// Errors produced while laying out text.
#[derive(Debug)]
pub enum LayoutError {
    BadFont,
    NoGlyphs,
}

impl std::fmt::Display for LayoutError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LayoutError::BadFont => write!(f, "failed to parse font"),
            LayoutError::NoGlyphs => write!(f, "text produced no drawable glyphs"),
        }
    }
}

impl std::error::Error for LayoutError {}

/// Returns true if the byte slice parses as a usable font face.
pub fn layout_probe(font_data: &[u8]) -> bool {
    Face::from_slice(font_data, 0).is_some()
}

/// Shapes `text` with the given font and returns contours placed in layout
/// space (font units, y-up) plus the font's units-per-em.
///
/// Each contour is tagged with a sequential character id derived from the
/// shaping cluster, so ligatures collapse into a single id.
pub fn layout_text<R: Rng>(
    font_data: &[u8],
    text: &str,
    cfg: &LayoutConfig,
    rng: &mut R,
) -> Result<(Vec<ContourInstance>, f32), LayoutError> {
    let face = Face::from_slice(font_data, 0).ok_or(LayoutError::BadFont)?;
    let upem = face.units_per_em() as f32;

    let mut buffer = UnicodeBuffer::new();
    buffer.push_str(text);
    let glyphs = rustybuzz::shape(&face, &[], buffer);

    // Map shaping clusters to sequential character ids.
    let mut clusters: Vec<u32> = glyphs.glyph_infos().iter().map(|g| g.cluster).collect();
    clusters.sort_unstable();
    clusters.dedup();
    let char_id_of = |cluster: u32| clusters.binary_search(&cluster).unwrap() as u32;

    let tracking = cfg.tracking * upem;
    let mut contours_out: Vec<ContourInstance> = Vec::new();
    let mut pen_x = 0.0f32;

    for (info, pos) in glyphs.glyph_infos().iter().zip(glyphs.glyph_positions()) {
        let dx = pen_x + pos.x_offset as f32;
        let dy = pos.y_offset as f32
            + if cfg.baseline_jitter > 0.0 {
                rng.random_range(-cfg.baseline_jitter..=cfg.baseline_jitter) * upem
            } else {
                0.0
            };
        pen_x += pos.x_advance as f32 + tracking;

        let mut builder = ContourBuilder::new();
        if face
            .outline_glyph(GlyphId(info.glyph_id as u16), &mut builder)
            .is_none()
        {
            continue; // whitespace or empty glyph
        }
        let char_id = char_id_of(info.cluster);
        for mut contour in builder.finish() {
            for p in &mut contour.points {
                let mut x = p[0] + dx;
                let mut y = p[1] + dy;
                if cfg.point_noise > 0.0 {
                    let n = cfg.point_noise * upem;
                    x += rng.random_range(-n..=n);
                    y += rng.random_range(-n..=n);
                }
                p[0] = x * cfg.aspect_x;
                p[1] = y;
            }
            contours_out.push(ContourInstance { contour, char_id });
        }
    }

    if contours_out.is_empty() {
        return Err(LayoutError::NoGlyphs);
    }
    Ok((contours_out, upem))
}
