//! The published picture (facegen/publish.py): a master cropped to 3:4, scaled with
//! Lanczos-3 and saved as an optimised 8-bit RGB PNG.
//!
//! Python decides the crop box (`picture.aspect_box`) and the size
//! (`picture.published_size`); this module does the pixel work. The batch version releases
//! the GIL and spreads the pictures over all cores with rayon.

use std::fmt;
use std::path::{Path, PathBuf};

use fast_image_resize as fr;
use oxipng::{BitDepth, ColorType, Options, RawImage};
use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rayon::prelude::*;

/// left, top, right, bottom in pixels, as PIL's crop takes them.
pub type CropBox = (u32, u32, u32, u32);

/// Why a picture could not be encoded.
#[derive(Debug)]
pub enum Error {
    /// The master could not be read.
    Io(String),
    /// A bad argument or a picture the encoder rejected.
    Invalid(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(m) | Error::Invalid(m) => f.write_str(m),
        }
    }
}

impl From<Error> for PyErr {
    fn from(e: Error) -> PyErr {
        match e {
            Error::Io(m) => PyOSError::new_err(m),
            Error::Invalid(m) => PyValueError::new_err(m),
        }
    }
}

/// PNG settings: oxipng's default level, but always 8-bit RGB, as Pillow writes it and as
/// FM27 was checked with (Phase 0.2). No palette, grey or bit-depth reductions.
fn png_options() -> Options {
    let mut o = Options::from_preset(2);
    o.bit_depth_reduction = false;
    o.color_type_reduction = false;
    o.palette_reduction = false;
    o.grayscale_reduction = false;
    o
}

/// The master at `path`, cropped to `crop` and scaled to `size`, as PNG bytes.
pub fn encode(path: &Path, crop: CropBox, size: (u32, u32)) -> Result<Vec<u8>, Error> {
    let at = |e: &dyn fmt::Display| format!("{}: {e}", path.display());
    let master = image::open(path)
        .map_err(|e| match e {
            image::ImageError::IoError(io) => Error::Io(at(&io)),
            other => Error::Invalid(at(&other)),
        })?
        .into_rgb8();
    let (width, height) = master.dimensions();
    let pixels = scale(master.into_raw(), (width, height), crop, size)
        .map_err(|m| Error::Invalid(at(&m)))?;
    to_png(pixels, size)
}

/// RGB pixels of a `width` x `height` picture, cropped and scaled.
///
/// Like Pillow's `img.crop(box).resize(size, LANCZOS)`, and like `published_bytes` a crop that
/// is already the right size is not resampled.
pub fn scale(
    pixels: Vec<u8>,
    (width, height): (u32, u32),
    (left, top, right, bottom): CropBox,
    (dw, dh): (u32, u32),
) -> Result<Vec<u8>, String> {
    if !(left < right && top < bottom && right <= width && bottom <= height) {
        return Err(format!(
            "crop ({left}, {top}, {right}, {bottom}) does not fit a {width}x{height} picture"
        ));
    }
    if dw == 0 || dh == 0 {
        return Err(format!("cannot scale to {dw}x{dh}"));
    }
    let (cw, ch) = (right - left, bottom - top);
    if (cw, ch) == (dw, dh) {
        let row = |y: u32| {
            let start = ((y * width + left) * 3) as usize;
            &pixels[start..start + (cw * 3) as usize]
        };
        return Ok((top..bottom).flat_map(row).copied().collect());
    }
    let src = fr::images::Image::from_vec_u8(width, height, pixels, fr::PixelType::U8x3)
        .map_err(|e| e.to_string())?;
    let mut dst = fr::images::Image::new(dw, dh, fr::PixelType::U8x3);
    let options = fr::ResizeOptions::new()
        .resize_alg(fr::ResizeAlg::Convolution(fr::FilterType::Lanczos3))
        .crop(left.into(), top.into(), cw.into(), ch.into());
    fr::Resizer::new()
        .resize(&src, &mut dst, &options)
        .map_err(|e| e.to_string())?;
    Ok(dst.into_vec())
}

/// RGB pixels as an optimised PNG.
pub fn to_png(pixels: Vec<u8>, (width, height): (u32, u32)) -> Result<Vec<u8>, Error> {
    let rgb = ColorType::RGB {
        transparent_color: None,
    };
    RawImage::new(width, height, rgb, BitDepth::Eight, pixels)
        .and_then(|raw| raw.create_optimized_png(&png_options()))
        .map_err(|e| Error::Invalid(format!("PNG encoding failed: {e}")))
}

/// One published picture as PNG bytes: the master at `master`, cropped to `crop` (left, top,
/// right, bottom) and scaled to `size` (width, height).
#[pyfunction]
pub fn encode_portrait<'py>(
    py: Python<'py>,
    master: PathBuf,
    crop: CropBox,
    size: (u32, u32),
) -> PyResult<Bound<'py, PyBytes>> {
    let data = py.detach(|| encode(&master, crop, size))?;
    Ok(PyBytes::new(py, &data))
}

/// Several published pictures at once, on all cores: one (master, crop, size) per picture,
/// the PNG bytes in the same order. The first failure raises.
#[pyfunction]
pub fn encode_portraits<'py>(
    py: Python<'py>,
    jobs: Vec<(PathBuf, CropBox, (u32, u32))>,
) -> PyResult<Vec<Bound<'py, PyBytes>>> {
    let encoded: Result<Vec<Vec<u8>>, Error> = py.detach(|| {
        jobs.par_iter()
            .map(|(master, crop, size)| encode(master, *crop, *size))
            .collect()
    });
    Ok(encoded?.iter().map(|data| PyBytes::new(py, data)).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 4x2 picture whose pixel (x, y) is (x, y, 7).
    fn sample() -> Vec<u8> {
        (0..2u8)
            .flat_map(|y| (0..4u8).flat_map(move |x| [x, y, 7]))
            .collect()
    }

    #[test]
    fn a_crop_of_the_right_size_is_copied() {
        let out = scale(sample(), (4, 2), (1, 0, 3, 2), (2, 2)).unwrap();
        assert_eq!(out, vec![1, 0, 7, 2, 0, 7, 1, 1, 7, 2, 1, 7]);
    }

    #[test]
    fn a_flat_picture_stays_flat_when_scaled() {
        let flat = [10u8, 20, 30].repeat(64 * 48);
        let out = scale(flat, (64, 48), (8, 0, 56, 48), (12, 16)).unwrap();
        assert_eq!(out.len(), 12 * 16 * 3);
        assert!(out.chunks(3).all(|p| p == [10, 20, 30]));
    }

    #[test]
    fn bad_crops_and_sizes_are_refused() {
        assert!(scale(sample(), (4, 2), (0, 0, 5, 2), (2, 2)).is_err());
        assert!(scale(sample(), (4, 2), (2, 0, 2, 2), (2, 2)).is_err());
        assert!(scale(sample(), (4, 2), (0, 0, 4, 2), (0, 2)).is_err());
    }

    #[test]
    fn png_round_trips_as_8_bit_rgb() {
        let pixels: Vec<u8> = (0..16 * 16 * 3).map(|i| (i % 251) as u8).collect();
        let png = to_png(pixels.clone(), (16, 16)).unwrap();
        let back = image::load_from_memory(&png).unwrap();
        assert_eq!(back.color(), image::ColorType::Rgb8);
        assert_eq!(back.into_rgb8().into_raw(), pixels);
    }

    #[test]
    fn a_missing_master_is_an_io_error() {
        let err = encode(Path::new("/nonexistent/master.png"), (0, 0, 1, 1), (1, 1));
        assert!(matches!(err, Err(Error::Io(_))));
    }
}
