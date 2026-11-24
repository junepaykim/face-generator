//! The figure scan behind `qc.head_top` and `picture.face_box` (kitshot/figure.py).
//!
//! A [`Grid`] is a small RGB copy of a picture. `Grid.scaled` makes it from the full picture
//! with a bit-exact port of Pillow's resize (resample.rs), so the Rust and Python scans see
//! the same pixels.
//!
//! The backdrop is followed down the picture at each side ([`Grid::edges_of`]): it starts as
//! the per-channel median of the top `corner` x `corner` square on that side, and in each row
//! the median of that side's outer `corner` pixels replaces it if within `diff` of it, so a
//! gradient is followed and hair or an arm reaching the edge is not. A pixel is backdrop if
//! its summed RGB difference from the row's left colour, right colour, or the colour between
//! them at its place across is at most `diff`; otherwise it is foreground.

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

/// The backdrop's colour at the left and the right end of one row.
pub type Edge = (Rgb, Rgb);

fn distance(a: Rgb, b: Rgb) -> u32 {
    a.0.abs_diff(b.0) as u32 + a.1.abs_diff(b.1) as u32 + a.2.abs_diff(b.2) as u32
}

/// A small RGB picture, row by row, three bytes a pixel.
#[pyclass(frozen, module = "kitshot_native")]
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

    fn is_foreground(&self, x: usize, y: usize, edge: Edge, diff: u32) -> bool {
        let p = self.pixel(x, y);
        let (left, right) = edge;
        if distance(p, left) <= diff || distance(p, right) <= diff {
            return false;
        }
        let span = (self.width - 1) as u32;
        if span == 0 {
            return true;
        }
        let x = x as u32;
        // The colour between the two ends at x, as Python's `(l * (span - x) + r * x) // span`.
        let mix = |l: u8, r: u8| ((l as u32 * (span - x) + r as u32 * x) / span) as u8;
        let between = (
            mix(left.0, right.0),
            mix(left.1, right.1),
            mix(left.2, right.2),
        );
        distance(p, between) > diff
    }

    /// The per-channel median of the pixels at `xs` in rows `ys`: Python's
    /// `sorted(values)[len(values) // 2]`, the upper median for an even count.
    fn median_of(&self, xs: &[usize], ys: std::ops::Range<usize>) -> Rgb {
        let mut channels: [Vec<u8>; 3] = Default::default();
        for y in ys {
            for &x in xs {
                let (r, g, b) = self.pixel(x, y);
                channels[0].push(r);
                channels[1].push(g);
                channels[2].push(b);
            }
        }
        let median = |values: &mut Vec<u8>| {
            values.sort_unstable();
            values[values.len() / 2]
        };
        let [r, g, b] = &mut channels;
        (median(r), median(g), median(b))
    }

    /// The backdrop at both ends of each row, followed down from the top corners while it
    /// changes by at most `diff` from one row to the next; None if the corner does not fit.
    pub fn edges_of(&self, corner: usize, diff: u32) -> Option<Vec<Edge>> {
        if corner == 0 || corner > self.width || corner > self.height {
            return None;
        }
        let sides: [Vec<usize>; 2] = [
            (0..corner).collect(),
            (self.width - corner..self.width).collect(),
        ];
        let mut current = [
            self.median_of(&sides[0], 0..corner),
            self.median_of(&sides[1], 0..corner),
        ];
        let mut out = Vec::with_capacity(self.height);
        for y in 0..self.height {
            for (side, xs) in sides.iter().enumerate() {
                let row = self.median_of(xs, y..y + 1);
                if distance(row, current[side]) <= diff {
                    current[side] = row;
                }
            }
            out.push((current[0], current[1]));
        }
        Some(out)
    }

    /// The first row with more than `need` foreground pixels.
    pub fn first_row_of(&self, edges: &[Edge], diff: u32, need: usize) -> Option<usize> {
        (0..self.height).find(|&y| {
            (0..self.width)
                .filter(|&x| self.is_foreground(x, y, edges[y], diff))
                .count()
                > need
        })
    }

    /// Foreground pixels in each column, counted over rows y0..y1.
    pub fn column_weights_of(&self, edges: &[Edge], diff: u32, y0: usize, y1: usize) -> Vec<usize> {
        (0..self.width)
            .map(|x| {
                (y0..y1)
                    .filter(|&y| self.is_foreground(x, y, edges[y], diff))
                    .count()
            })
            .collect()
    }

    fn check_edges(&self, edges: &[Edge]) -> PyResult<()> {
        if edges.len() != self.height {
            return Err(PyValueError::new_err(format!(
                "{} edges for the grid's {} rows",
                edges.len(),
                self.height
            )));
        }
        Ok(())
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

    /// The backdrop's colour at both ends of each row, followed down from the top
    /// `corner` x `corner` squares while it changes by at most `diff` a row.
    fn edges(&self, corner: usize, diff: u32) -> PyResult<Vec<Edge>> {
        self.edges_of(corner, diff).ok_or_else(|| {
            PyValueError::new_err(format!(
                "corner {corner} does not fit a {}x{} grid",
                self.width, self.height
            ))
        })
    }

    /// The first row with more than `need` foreground pixels, or None.
    fn first_row(&self, edges: Vec<Edge>, diff: u32, need: usize) -> PyResult<Option<usize>> {
        self.check_edges(&edges)?;
        Ok(self.first_row_of(&edges, diff, need))
    }

    /// Foreground pixels in each column, counted over rows y0..y1 (empty if y0 >= y1).
    fn column_weights(
        &self,
        edges: Vec<Edge>,
        diff: u32,
        y0: usize,
        y1: usize,
    ) -> PyResult<Vec<usize>> {
        self.check_edges(&edges)?;
        if y1 > self.height {
            return Err(PyValueError::new_err(format!(
                "row {y1} is past the grid's {} rows",
                self.height
            )));
        }
        Ok(self.column_weights_of(&edges, diff, y0, y1))
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

    /// Every row's edges are the backdrop.
    fn plain(height: usize) -> Vec<Edge> {
        vec![(BG, BG); height]
    }

    #[test]
    fn edges_are_the_corner_colour_on_a_plain_backdrop() {
        assert_eq!(sample().edges_of(2, 60), Some(plain(8)));
        assert_eq!(sample().edges_of(0, 60), None);
        assert_eq!(sample().edges_of(9, 60), None);
    }

    #[test]
    fn edges_take_the_upper_median() {
        // Rows of 10, 30, 0, 0 with a corner of 2: each left strip is [10, 30], and Python's
        // sorted([10, 30])[1] is 30.
        let row = [10u8, 30, 0, 0];
        let pixels: Vec<u8> = row.iter().chain(row.iter()).flat_map(|&v| [v; 3]).collect();
        let grid = Grid::from_rgb(pixels, 4, 2).unwrap();
        assert_eq!(
            grid.edges_of(2, 60),
            Some(vec![((30, 30, 30), (0, 0, 0)); 2])
        );
    }

    #[test]
    fn edges_follow_a_gradient_but_not_a_jump() {
        // One column: grey getting darker by 10 a row, then a jump of 120 at row 4.
        let shades = [200u8, 190, 180, 170, 50, 50];
        let pixels: Vec<u8> = shades.iter().flat_map(|&v| [v, v, v]).collect();
        let grid = Grid::from_rgb(pixels, 1, 6).unwrap();
        let left: Vec<u8> = grid
            .edges_of(1, 60)
            .unwrap()
            .iter()
            .map(|e| e.0.0)
            .collect();
        assert_eq!(left, vec![200, 190, 180, 170, 170, 170]);
    }

    #[test]
    fn a_sideways_gradient_is_backdrop() {
        // One row from black to white, 0 to 252 in steps of 28; the middle is far from both
        // ends but on the line between them.
        let pixels: Vec<u8> = (0..10u8).flat_map(|x| [x * 28; 3]).collect();
        let grid = Grid::from_rgb(pixels, 10, 1).unwrap();
        let edges = grid.edges_of(1, 60).unwrap();
        assert_eq!(grid.column_weights_of(&edges, 60, 0, 1), vec![0; 10]);
    }

    #[test]
    fn first_row_finds_the_head() {
        let grid = sample();
        assert_eq!(grid.first_row_of(&plain(8), 60, 1), Some(3));
        assert_eq!(grid.first_row_of(&plain(8), 60, 2), None); // never more than 2 in a row
    }

    #[test]
    fn column_weights_count_the_band() {
        let weights = sample().column_weights_of(&plain(8), 60, 2, 6);
        assert_eq!(weights, vec![0, 0, 0, 0, 3, 3, 0, 0, 0, 0]);
        assert_eq!(sample().column_weights_of(&plain(8), 60, 5, 5), vec![0; 10]);
    }
}
