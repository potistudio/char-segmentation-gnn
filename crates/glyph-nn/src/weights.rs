//! Reads the `.gnnw` dump written by `glyph_gnn.export_weights`.

use std::collections::HashMap;
use std::path::Path;

use serde::Deserialize;

const MAGIC: &[u8; 4] = b"GNNW";
const FORMAT_VERSION: u32 = 1;

#[derive(Debug, Deserialize)]
struct Entry {
    name: String,
    shape: Vec<usize>,
    /// Byte offset into the data section.
    offset: usize,
}

#[derive(Debug, Deserialize)]
struct Header {
    hparams: HParams,
    tensors: Vec<Entry>,
}

/// The subset of the checkpoint's hyperparameters that changes the forward pass.
#[derive(Debug, Clone, Copy, Deserialize)]
pub struct HParams {
    pub node_dim: usize,
    pub edge_dim: usize,
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub contour_dim: usize,
    pub context: bool,
    pub contours: bool,
    pub relations: bool,
}

/// A borrowed, row-major view of one tensor.
#[derive(Debug, Clone, Copy)]
pub struct View<'a> {
    pub data: &'a [f32],
    pub rows: usize,
    pub cols: usize,
}

/// Every tensor of one checkpoint, in a single allocation.
pub struct Weights {
    data: Vec<f32>,
    index: HashMap<String, (usize, usize, usize)>,
    pub hparams: HParams,
}

#[derive(Debug)]
pub enum LoadError {
    Io(std::io::Error),
    Format(String),
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadError::Io(e) => write!(f, "{e}"),
            LoadError::Format(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for LoadError {}

impl From<std::io::Error> for LoadError {
    fn from(e: std::io::Error) -> Self {
        LoadError::Io(e)
    }
}

fn bad(msg: impl Into<String>) -> LoadError {
    LoadError::Format(msg.into())
}

impl Weights {
    pub fn load(path: &Path) -> Result<Self, LoadError> {
        let raw = std::fs::read(path)?;
        if raw.len() < 12 || &raw[..4] != MAGIC {
            return Err(bad(format!("{} is not a .gnnw file", path.display())));
        }
        let version = u32::from_le_bytes(raw[4..8].try_into().unwrap());
        if version != FORMAT_VERSION {
            return Err(bad(format!(
                "{} is format version {version}, expected {FORMAT_VERSION}; re-run \
                 glyph_gnn.export_weights",
                path.display()
            )));
        }
        let header_len = u32::from_le_bytes(raw[8..12].try_into().unwrap()) as usize;
        let body = 12 + header_len;
        if raw.len() < body {
            return Err(bad("truncated header"));
        }
        let header: Header =
            serde_json::from_slice(&raw[12..body]).map_err(|e| bad(format!("bad header: {e}")))?;

        let floats = (raw.len() - body) / 4;
        let mut data = vec![0.0f32; floats];
        for (i, slot) in data.iter_mut().enumerate() {
            let at = body + i * 4;
            *slot = f32::from_le_bytes(raw[at..at + 4].try_into().unwrap());
        }

        let mut index = HashMap::with_capacity(header.tensors.len());
        for entry in &header.tensors {
            if entry.offset % 4 != 0 {
                return Err(bad(format!("{} is not 4-byte aligned", entry.name)));
            }
            // A bias is 1-D and a `att` parameter is 3-D; both are read as a
            // single row, which is how every consumer here wants them.
            let (rows, cols) = match entry.shape.as_slice() {
                [] => (1, 1),
                [n] => (1, *n),
                [r, c] => (*r, *c),
                rest => (1, rest.iter().product()),
            };
            let start = entry.offset / 4;
            if start + rows * cols > floats {
                return Err(bad(format!("{} runs past the end of the file", entry.name)));
            }
            index.insert(entry.name.clone(), (start, rows, cols));
        }
        Ok(Self {
            data,
            index,
            hparams: header.hparams,
        })
    }

    /// Panics if the tensor is absent: the set of names is fixed by the model
    /// definition, so a miss is a build error rather than bad input.
    pub fn get(&self, name: &str) -> View<'_> {
        self.try_get(name)
            .unwrap_or_else(|| panic!("checkpoint has no tensor {name:?}"))
    }

    pub fn try_get(&self, name: &str) -> Option<View<'_>> {
        let &(start, rows, cols) = self.index.get(name)?;
        Some(View {
            data: &self.data[start..start + rows * cols],
            rows,
            cols,
        })
    }
}
