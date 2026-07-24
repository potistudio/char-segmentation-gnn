//! Shared core for the glyph splitter pipeline: outline extraction,
//! resampling, feature computation and graph construction. Used by both
//! the dataset generator and the inference engine so training and
//! inference preprocessing stay bit-identical.

pub mod graph;
pub mod layout;
pub mod path;
pub mod sample;

pub use graph::{
    ContourInstance, EDGE_DIM, GraphConfig, GraphSample, NODE_DIM, UNKNOWN_CHAR, build_graph,
};
pub use layout::{LayoutConfig, LayoutError, layout_text};
pub use path::{Contour, ContourBuilder};
pub use sample::{SampleConfig, SampledPoint, resample_contour};
