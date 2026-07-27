//! Native CPU runtime for the glyph edge GNN (SPEED-PLAN Tier C).
//!
//! onnxruntime executes a generic graph: it materialises an `[edges, hidden]`
//! tensor per operator and writes messages into node rows with a scatter, which
//! is a random write it cannot split across threads. Measured, that single
//! operator was 43% of CPU time and twelve threads bought 1.39x.
//!
//! This crate exists because the caller already owns the graph. Building the
//! incoming-edge index up front turns every aggregation into a contiguous read
//! per node, so the write is local and the loop splits cleanly, and the
//! elementwise chain that ONNX spreads across `Add`/`Mul`/`LeakyRelu` collapses
//! into the one pass that computes the attention logits.
//!
//! Weights come from `glyph_gnn.export_weights`, not from the ONNX file.

pub mod model;
pub mod ops;
pub mod weights;

pub use model::Model;
pub use weights::{HParams, LoadError, Weights};
