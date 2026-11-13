//! The figure scan behind `qc.head_top` and `picture.face_box` (facegen/figure.py).
//!
//! A [`Grid`] is a small RGB copy of a picture. `Grid.scaled` makes it from the full picture
//! with a bit-exact port of Pillow's resize (resample.rs), so the Rust and Python scans see
//! the same pixels. The backdrop colour is the per-channel median of the two top corners;
//! a pixel whose summed RGB difference from it is over `diff` is foreground.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::resample::resize_rgb;

fn wrong_size(pixels: &[u8], width: usize, height: usize) -> PyErr {
    PyValueError::new_err(format!(
        "{} bytes is not a {width}x{height} RGB picture",
        pixels.len()
    ))
}

/// An RGB colour.
pub type Rgb = (u8, u8, u8);

/// A small RGB picture, row by row, three bytes a pixel.
#[pyclass(frozen, module = "facegen_native")]
pub struct Grid {
    #[pyo3(get)]
    width: usize,
    #[pyo3(get)]
    height: usize,
    pixels: Vec<u8>,
}

impl Grid {
    /// A grid from raw RGB bytes; None if the length does not match the size.
    pub fn from_rgb(pixels: Vec<u8>, width: usize, height: usize) -> Option<Self> {
        let expected = width.checked_mul(height)?.checked_mul(3)?;
        (pixels.len() == expected).then_some(Self {
            width,
            height,
            pixels,
        })
    }

    fn pixel(&self, x: usize, y: usize) -> Rgb {
        let i = (y * self.width + x) * 3;
        (self.pixels[i], self.pixels[i + 1], self.pixels[i + 2])
    }

    fn is_foreground(&self, x: usize, y: usize, bg: Rgb, diff: u32) -> bool {
        let (r, g, b) = self.pixel(x, y);
        let d = r.abs_diff(bg.0) as u32 + g.abs_diff(bg.1) as u32 + b.abs_diff(bg.2) as u32;
        d > diff
    }

    /// The backdrop colour, or None if the corner does not fit in the grid.
    pub fn backdrop_of(&self, corner: usize) -> Option<Rgb> {
        if corner == 0 || corner > self.width || corner > self.height {
            return None;
        }
        // The same pixels, in the same order, as the Python version: rows 0..corner, and in
        // each the left `corner` columns and then the right `corner` columns.
        let xs: Vec<usize> = (0..corner).chain(self.width - corner..self.width).collect();
        let mut channels: [Vec<u8>; 3] = Default::default();
        for y in 0..corner {
            for &x in &xs {
                let (r, g, b) = self.pixel(x, y);
                channels[0].push(r);
                channels[1].push(g);
                channels[2].push(b);
            }
        }
        // Python's `sorted(values)[len(values) // 2]`: the upper median for an even count.
        let median = |values: &mut Vec<u8>| {
            values.sort_unstable();
            values[values.len() / 2]
        };
        let [r, g, b] = &mut channels;
        Some((median(r), median(g), median(b)))
    }

    /// The first row with more than `need` foreground pixels.
    pub fn first_row_of(&self, bg: Rgb, diff: u32, need: usize) -> Option<usize> {
        (0..self.height).find(|&y| {
            (0..self.width)
                .filter(|&x| self.is_foreground(x, y, bg, diff))
                .count()
                > need
        })
    }

    /// Foreground pixels in each column, counted over rows y0..y1.
    pub fn column_weights_of(&self, bg: Rgb, diff: u32, y0: usize, y1: usize) -> Vec<usize> {
        (0..self.width)
            .map(|x| {
                (y0..y1)
                    .filter(|&y| self.is_foreground(x, y, bg, diff))
                    .count()
            })
            .collect()
    }
}

#[pymethods]
impl Grid {
    /// A grid from `width * height * 3` bytes of RGB, as `Image.tobytes()` gives them.
    #[new]
    fn new(pixels: &[u8], width: usize, height: usize) -> PyResult<Self> {
        Self::from_rgb(pixels.to_vec(), width, height)
            .ok_or_else(|| wrong_size(pixels, width, height))
    }

    /// A grid of `grid_width` x `grid_height` from a full-size picture (`width * height * 3`
    /// bytes of RGB), scaled exactly as Pillow's `Image.resize` would (resample.rs). The GIL
    /// is released while it scales.
    #[staticmethod]
    fn scaled(
        py: Python<'_>,
        pixels: &[u8],
        width: usize,
        height: usize,
        grid_width: usize,
        grid_height: usize,
    ) -> PyResult<Self> {
        if width.checked_mul(height).and_then(|n| n.checked_mul(3)) != Some(pixels.len()) {
            return Err(wrong_size(pixels, width, height));
        }
        if [width, height, grid_width, grid_height].contains(&0) {
            return Err(PyValueError::new_err("a size is zero"));
        }
        let small = py.detach(|| resize_rgb(pixels, width, height, grid_width, grid_height));
        Ok(Self {
            width: grid_width,
            height: grid_height,
            pixels: small,
        })
    }

    /// The grid's pixels as RGB bytes, as `Image.tobytes()` gives them.
    fn tobytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.pixels)
    }

    /// The backdrop colour: the per-channel median of the two top `corner` x `corner` squares.
    fn backdrop(&self, corner: usize) -> PyResult<Rgb> {
        self.backdrop_of(corner).ok_or_else(|| {
            PyValueError::new_err(format!(
                "corner {corner} does not fit a {}x{} grid",
                self.width, self.height
            ))
        })
    }

    /// The first row with more than `need` foreground pixels, or None.
    fn first_row(&self, backdrop: Rgb, diff: u32, need: usize) -> Option<usize> {
        self.first_row_of(backdrop, diff, need)
    }

    /// Foreground pixels in each column, counted over rows y0..y1 (empty if y0 >= y1).
    fn column_weights(
        &self,
        backdrop: Rgb,
        diff: u32,
        y0: usize,
        y1: usize,
    ) -> PyResult<Vec<usize>> {
        if y1 > self.height {
            return Err(PyValueError::new_err(format!(
                "row {y1} is past the grid's {} rows",
                self.height
            )));
        }
        Ok(self.column_weights_of(backdrop, diff, y0, y1))
    }

    fn __repr__(&self) -> String {
        format!("Grid({}x{})", self.width, self.height)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BG: Rgb = (200, 200, 205);
    const SKIN: Rgb = (240, 205, 180);

    /// A 10x8 grid of backdrop with a 2-wide "head" in columns 4..6 from row 3 down.
    fn sample() -> Grid {
        let (w, h) = (10, 8);
        let mut pixels = Vec::with_capacity(w * h * 3);
        for y in 0..h {
            for x in 0..w {
                let (r, g, b) = if y >= 3 && (4..6).contains(&x) {
                    SKIN
                } else {
                    BG
                };
                pixels.extend([r, g, b]);
            }
        }
        Grid::from_rgb(pixels, w, h).unwrap()
    }

    #[test]
    fn rejects_a_wrong_length() {
        assert!(Grid::from_rgb(vec![0; 10], 2, 2).is_none());
    }

    #[test]
    fn backdrop_is_the_corner_colour() {
        assert_eq!(sample().backdrop_of(2), Some(BG));
        assert_eq!(sample().backdrop_of(0), None);
        assert_eq!(sample().backdrop_of(9), None);
    }

    #[test]
    fn backdrop_takes_the_upper_median() {
        // Two pixels: Python's sorted([10, 30])[1] is 30.
        let grid = Grid::from_rgb(vec![10, 10, 10, 30, 30, 30], 2, 1).unwrap();
        assert_eq!(grid.backdrop_of(1), Some((30, 30, 30)));
    }

    #[test]
    fn first_row_finds_the_head() {
        let grid = sample();
        assert_eq!(grid.first_row_of(BG, 60, 1), Some(3));
        assert_eq!(grid.first_row_of(BG, 60, 2), None); // never more than 2 in a row
    }

    #[test]
    fn column_weights_count_the_band() {
        let weights = sample().column_weights_of(BG, 60, 2, 6);
        assert_eq!(weights, vec![0, 0, 0, 0, 3, 3, 0, 0, 0, 0]);
        assert_eq!(sample().column_weights_of(BG, 60, 5, 5), vec![0; 10]);
    }
}
