//! Kitshot's optional Rust extension, imported in Python as `kitshot_native`.
//!
//! Every function here has a Python version in the `kitshot` package that gives the same
//! result, and the tool runs without this extension (kitshot/native.py). It only makes two
//! CPU-bound pieces faster:
//!
//! - [`figure`]: the scan that finds the figure against the studio backdrop
//!   (`qc.head_top`, `picture.face_box`), on a small copy made by [`resample`], a bit-exact
//!   port of Pillow's bicubic resize.
//! - [`portrait`]: the published picture, a master cropped to 3:4, scaled and saved as an
//!   optimised PNG (`publish.published_bytes`).
//!
//! The geometry (which box to crop, which size to publish, the thresholds) stays in Python;
//! Rust receives it as arguments, so there is one source of truth for each number.

pub mod figure;
pub mod portrait;
pub mod resample;

use pyo3::prelude::*;

/// Rust versions of Kitshot's CPU-bound picture work. See native/src/lib.rs.
#[pymodule]
mod kitshot_native {
    #[pymodule_export]
    use super::figure::Grid;
    #[pymodule_export]
    use super::portrait::{encode_portrait, encode_portraits};
}
