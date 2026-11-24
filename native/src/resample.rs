//! Pillow's bicubic resize, the default of `Image.resize`, for 8-bit RGB. Bit for bit.
//!
//! A port of Pillow 13.0.0's `src/libImaging/Resample.c` (`precompute_coeffs`,
//! `normalize_coeffs_8bpc`, the 8-bit passes and `ImagingResampleInner`), for the whole
//! picture as the box. The figure scan uses it so that the Rust and Python scans see the
//! same pixels; `tests/test_native.py` compares it with the installed Pillow and fails if a
//! Pillow upgrade changes the resample.
//!
//! The details that make it exact, all as in the C:
//! - weights in f64, in the same order of operations, then rounded half away from zero to
//!   fixed point with 22 fractional bits;
//! - each pass accumulates in i32 from a half-unit bias and clips to 0..=255, so the
//!   picture between the two passes is 8-bit;
//! - the horizontal pass runs first, over only the rows the vertical pass reads, unless the
//!   height shrinks by more than twice as much as the width (new in Pillow 13).
//!
//! Unlike Pillow, each pass is split across cores by rows, in a small rayon pool of its own;
//! the rows are independent, so the result is the same.

use std::sync::OnceLock;

use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};

const PRECISION_BITS: u32 = 32 - 8 - 2;
const CHANNELS: usize = 3;

/// Pillow's `bicubic_filter`, a = -0.5, support 2.
fn bicubic(x: f64) -> f64 {
    const A: f64 = -0.5;
    let x = if x < 0.0 { -x } else { x };
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * A
    } else {
        0.0
    }
}
const SUPPORT: f64 = 2.0;

/// The fixed-point weights of one pass: for each output position, the first input index
/// and the weights of the inputs from there on.
struct Coeffs {
    bounds: Vec<(usize, usize)>,
    ksize: usize,
    weights: Vec<i32>,
}

impl Coeffs {
    /// Weights for scaling `in_size` positions to `out_size` (`precompute_coeffs` with the
    /// box from 0 to `in_size`, then `normalize_coeffs_8bpc`).
    fn new(in_size: usize, out_size: usize) -> Self {
        // `(double)(in1 - in0) / outSize`: the box edges are floats in Pillow.
        let (in0, in1) = (0f32, in_size as f32);
        let scale = f64::from(in1 - in0) / out_size as f64;
        let filterscale = if scale < 1.0 { 1.0 } else { scale };
        let support = SUPPORT * filterscale;
        let ksize = support.ceil() as usize * 2 + 1;
        let inv_filterscale = 1.0 / filterscale;

        let mut bounds = Vec::with_capacity(out_size);
        let mut weights = vec![0i32; out_size * ksize];
        let mut k = vec![0f64; ksize];
        for xx in 0..out_size {
            let center = f64::from(in0) + (xx as f64 + 0.5) * scale;
            // C's (int) cast truncates toward zero, as Rust's `as` does.
            let xmin = ((center - support + 0.5) as i64).max(0);
            let xmax = ((center + support + 0.5) as i64).min(in_size as i64) - xmin;
            let count = xmax.max(0) as usize;
            let mut ww = 0.0;
            for (x, w) in k.iter_mut().enumerate().take(count) {
                *w = bicubic(((x as i64 + xmin) as f64 - center + 0.5) * inv_filterscale);
                ww += *w;
            }
            if ww != 0.0 {
                k[..count].iter_mut().for_each(|w| *w /= ww);
            }
            let fixed = &mut weights[xx * ksize..xx * ksize + count];
            for (out, &w) in fixed.iter_mut().zip(&k) {
                let scaled = w * f64::from(1i32 << PRECISION_BITS);
                *out = if w < 0.0 {
                    (-0.5 + scaled) as i32
                } else {
                    (0.5 + scaled) as i32
                };
            }
            bounds.push((xmin as usize, count));
        }
        Self {
            bounds,
            ksize,
            weights,
        }
    }

    fn of(&self, i: usize) -> (usize, &[i32]) {
        let (start, count) = self.bounds[i];
        (start, &self.weights[i * self.ksize..i * self.ksize + count])
    }
}

/// Pillow's `clip8`: the accumulator back to a byte.
fn clip8(acc: i32) -> u8 {
    (acc >> PRECISION_BITS).clamp(0, 255) as u8
}

const HALF: i32 = 1 << (PRECISION_BITS - 1);

/// Threads of the resize's own pool. A scan's resize is only a few milliseconds of work: on
/// this PC (6 performance and 8 efficiency cores, 20 threads) it was fastest on 6–8
/// threads, and waking all 20 made it slower than 4 did (docs/research.md).
const THREADS: usize = 8;

/// The resize's own rayon pool, made on first use. Separate from the global pool that the
/// portrait encoder uses, so neither waits for the other.
fn pool() -> &'static ThreadPool {
    static POOL: OnceLock<ThreadPool> = OnceLock::new();
    POOL.get_or_init(|| {
        let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
        ThreadPoolBuilder::new()
            .num_threads(THREADS.min(cores))
            .thread_name(|i| format!("kitshot-resize-{i}"))
            .build()
            .expect("the resize thread pool")
    })
}

type Pixel = [u8; CHANNELS];

/// One weighted sum per channel over `pixels` and `weights`, in Pillow's order.
fn weighted<'a>(pixels: impl Iterator<Item = &'a Pixel>, weights: &[i32]) -> Pixel {
    let mut acc = [HALF; CHANNELS];
    for (p, &k) in pixels.zip(weights) {
        for c in 0..CHANNELS {
            acc[c] += i32::from(p[c]) * k;
        }
    }
    acc.map(clip8)
}

/// Packed RGB bytes seen as pixels (the length is a whole number of pixels).
fn pixels_of(bytes: &[u8]) -> &[Pixel] {
    bytes.as_chunks::<CHANNELS>().0
}

/// Scale each of `rows` rows of `src` (width `in_w`), from row `offset` on, to `out_w`.
fn horizontal(src: &[u8], in_w: usize, offset: usize, rows: usize, k: &Coeffs) -> Vec<u8> {
    let out_w = k.bounds.len();
    let src = pixels_of(src);
    let mut out = vec![0u8; out_w * rows * CHANNELS];
    out.par_chunks_mut(out_w * CHANNELS)
        .enumerate()
        .for_each(|(yy, line_out)| {
            let line_in = &src[(yy + offset) * in_w..(yy + offset + 1) * in_w];
            for (xx, dst) in line_out
                .as_chunks_mut::<CHANNELS>()
                .0
                .iter_mut()
                .enumerate()
            {
                let (xmin, weights) = k.of(xx);
                *dst = weighted(line_in[xmin..xmin + weights.len()].iter(), weights);
            }
        });
    out
}

/// Scale the columns of `src` (`width` wide) to the coefficients' output height.
fn vertical(src: &[u8], width: usize, k: &Coeffs) -> Vec<u8> {
    let out_h = k.bounds.len();
    let src = pixels_of(src);
    let mut out = vec![0u8; width * out_h * CHANNELS];
    out.par_chunks_mut(width * CHANNELS)
        .enumerate()
        .for_each(|(yy, line_out)| {
            let (ymin, weights) = k.of(yy);
            for (xx, dst) in line_out
                .as_chunks_mut::<CHANNELS>()
                .0
                .iter_mut()
                .enumerate()
            {
                let column = (ymin..ymin + weights.len()).map(|y| &src[y * width + xx]);
                *dst = weighted(column, weights);
            }
        });
    out
}

/// `pixels` (packed RGB, `in_w` x `in_h`) scaled to `out_w` x `out_h`, exactly as Pillow's
/// `Image.resize((out_w, out_h))` with its default filter gives them. Panics if the length
/// does not match or a size is zero; the caller checks.
pub fn resize_rgb(pixels: &[u8], in_w: usize, in_h: usize, out_w: usize, out_h: usize) -> Vec<u8> {
    assert_eq!(
        pixels.len(),
        in_w * in_h * CHANNELS,
        "pixels do not match the size"
    );
    assert!(
        in_w > 0 && in_h > 0 && out_w > 0 && out_h > 0,
        "a size is zero"
    );
    pool().install(|| two_passes(pixels, in_w, in_h, out_w, out_h))
}

/// `ImagingResampleInner`: which passes run, and in which order.
fn two_passes(pixels: &[u8], in_w: usize, in_h: usize, out_w: usize, out_h: usize) -> Vec<u8> {
    let need_horizontal = out_w != in_w;
    let need_vertical = out_h != in_h;
    let (dw, dh) = (in_w as i64 - out_w as i64, in_h as i64 - out_h as i64);
    let horizontal_first = !(dh > 0 && dh > dw * 2);

    if horizontal_first {
        let mut vert = (need_horizontal || need_vertical).then(|| Coeffs::new(in_h, out_h));
        let mut current: Option<Vec<u8>> = None;
        if need_horizontal {
            let v = vert.as_mut().expect("vertical coefficients");
            // Only the rows the vertical pass reads; its bounds move up to match.
            let first = v.bounds[0].0;
            let (last_start, last_count) = v.bounds[out_h - 1];
            let last = last_start + last_count;
            v.bounds.iter_mut().for_each(|b| b.0 -= first);
            let horiz = Coeffs::new(in_w, out_w);
            current = Some(horizontal(pixels, in_w, first, last - first, &horiz));
        }
        if need_vertical {
            let v = vert.as_ref().expect("vertical coefficients");
            let src = current.as_deref().unwrap_or(pixels);
            current = Some(vertical(src, out_w, v));
        }
        current.unwrap_or_else(|| pixels.to_vec())
    } else {
        // Here the height shrinks, so the vertical pass is needed.
        let after_vertical = vertical(pixels, in_w, &Coeffs::new(in_h, out_h));
        if need_horizontal {
            horizontal(&after_vertical, in_w, 0, out_h, &Coeffs::new(in_w, out_w))
        } else {
            after_vertical
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pixel (x, y) is (37x, 53y, 29xy), each mod 256: the golden pictures' source.
    fn sample(w: usize, h: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(w * h * 3);
        for y in 0..h {
            for x in 0..w {
                out.extend([(x * 37) % 256, (y * 53) % 256, (x * y * 29) % 256].map(|v| v as u8));
            }
        }
        out
    }

    // Made with Pillow 13.0.0: `sample(w, h).resize(size).tobytes()`.
    #[test]
    fn matches_pillow_downscaling_horizontal_first() {
        let expected = [
            27, 45, 18, 111, 45, 74, 195, 45, 85, 27, 167, 73, 111, 167, 136, 195, 167, 115,
        ];
        assert_eq!(resize_rgb(&sample(7, 5), 7, 5, 3, 2), expected);
    }

    #[test]
    fn matches_pillow_on_a_tall_picture_vertical_first() {
        let expected = [
            7, 94, 15, 56, 94, 92, 104, 94, 91, 7, 116, 37, 56, 116, 138, 104, 116, 102, 7, 123,
            23, 56, 123, 122, 104, 123, 115, 7, 145, 37, 56, 145, 136, 104, 145, 110,
        ];
        assert_eq!(resize_rgb(&sample(4, 20), 4, 20, 3, 4), expected);
    }

    #[test]
    fn matches_pillow_upscaling() {
        let expected = [
            0, 0, 0, 11, 0, 0, 37, 0, 0, 63, 0, 0, 77, 0, 0, 0, 27, 0, 11, 27, 5, 37, 27, 15, 63,
            27, 25, 77, 27, 30, 0, 56, 0, 11, 56, 10, 37, 56, 31, 63, 56, 52, 77, 56, 64,
        ];
        assert_eq!(resize_rgb(&sample(3, 2), 3, 2, 5, 3), expected);
    }

    #[test]
    fn the_same_size_is_a_copy() {
        assert_eq!(resize_rgb(&sample(5, 4), 5, 4, 5, 4), sample(5, 4));
    }

    #[test]
    fn a_flat_picture_stays_flat() {
        let flat = [200u8, 200, 205].repeat(96 * 80);
        let out = resize_rgb(&flat, 96, 80, 13, 11);
        assert!(out.chunks(3).all(|p| p == [200, 200, 205]));
    }

    #[test]
    fn weights_sum_to_one_in_fixed_point() {
        let k = Coeffs::new(1536, 128);
        for i in 0..128 {
            let total: i32 = k.of(i).1.iter().sum();
            assert!(
                (total - (1 << PRECISION_BITS)).abs() <= 64,
                "row {i}: {total}"
            );
        }
    }
}
