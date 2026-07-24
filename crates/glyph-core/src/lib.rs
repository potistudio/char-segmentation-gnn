//! Shared core for the glyph splitter pipeline: outline extraction,
//! resampling, feature computation and graph construction. Used by both
//! the dataset generator and the inference engine so training and
//! inference preprocessing stay bit-identical.

pub mod graph;
pub mod layout;
pub mod path;
pub mod sample;

pub use graph::{build_graph, ContourInstance, GraphConfig, GraphSample, EDGE_DIM, NODE_DIM, UNKNOWN_CHAR};
pub use layout::{layout_text, LayoutConfig, LayoutError};
pub use path::{Contour, ContourBuilder};
pub use sample::{resample_contour, SampleConfig, SampledPoint};
