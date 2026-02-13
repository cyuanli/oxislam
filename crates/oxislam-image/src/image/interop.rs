//! `From`/`Into` conversions between [`image`] crate types and oxislam types.
//!
//! Requires the `image` feature.

use crate::image::Image;
use crate::pixel::{Gray, Rgb};

// --- Pixel-level conversions ---

impl From<image::Rgb<u8>> for Rgb<u8> {
    fn from(p: image::Rgb<u8>) -> Self { Rgb::new(p[0], p[1], p[2]) }
}

impl From<Rgb<u8>> for image::Rgb<u8> {
    fn from(p: Rgb<u8>) -> Self { image::Rgb([p.r, p.g, p.b]) }
}

// --- Image-level conversions ---

impl From<image::RgbImage> for Image<Rgb<u8>> {
    /// Zero-copy: reinterprets the flat `Vec<u8>` as `Vec<Rgb<u8>>`.
    fn from(img: image::RgbImage) -> Self {
        let (w, h) = (img.width() as usize, img.height() as usize);
        let raw = img.into_raw();
        debug_assert_eq!(raw.len(), w * h * 3);

        // SAFETY: Rgb<u8> is #[repr(C)] with three u8 fields (size 3, align 1),
        // identical layout to the [r,g,b, r,g,b, ...] flat buffer.
        let data = unsafe {
            let ptr = raw.as_ptr() as *mut Rgb<u8>;
            let len = w * h;
            std::mem::forget(raw);
            Vec::from_raw_parts(ptr, len, len)
        };
        Image::new(w, h, w, data)
    }
}

impl From<&Image<Rgb<u8>>> for image::RgbImage {
    fn from(img: &Image<Rgb<u8>>) -> Self {
        let (w, h) = (img.width(), img.height());
        if img.stride() == w {
            // Contiguous: single memcpy via as_bytes reinterpret.
            // SAFETY: Rgb<u8> is #[repr(C)] {r,g,b} — size 3, align 1, no padding.
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(img.data().as_ptr() as *const u8, img.data().len() * 3)
            };
            image::RgbImage::from_raw(w as u32, h as u32, bytes.to_vec())
                .expect("buffer size matches dimensions")
        } else {
            let mut buf = Vec::with_capacity(w * h * 3);
            for row in img.view().rows() {
                // SAFETY: same repr(C) layout reasoning as above.
                let row_bytes: &[u8] =
                    unsafe { std::slice::from_raw_parts(row.as_ptr() as *const u8, row.len() * 3) };
                buf.extend_from_slice(row_bytes);
            }
            image::RgbImage::from_raw(w as u32, h as u32, buf)
                .expect("buffer size matches dimensions")
        }
    }
}

impl From<image::GrayImage> for Image<Gray<u8>> {
    fn from(img: image::GrayImage) -> Self {
        let (w, h) = (img.width() as usize, img.height() as usize);
        Image::from_raw(w, h, w, img.into_raw())
    }
}

impl From<&Image<Gray<u8>>> for image::GrayImage {
    fn from(img: &Image<Gray<u8>>) -> Self {
        let (w, h) = (img.width(), img.height());
        if img.stride() == w {
            image::GrayImage::from_raw(w as u32, h as u32, img.as_raw().to_vec())
                .expect("buffer size matches dimensions")
        } else {
            // Strip stride padding row by row.
            let raw = img.as_raw();
            let mut buf = Vec::with_capacity(w * h);
            for y in 0..h {
                let start = y * img.stride();
                buf.extend_from_slice(&raw[start..start + w]);
            }
            image::GrayImage::from_raw(w as u32, h as u32, buf)
                .expect("buffer size matches dimensions")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgb_pixel_roundtrip() {
        let orig = Rgb::new(10, 20, 30);
        let ext: image::Rgb<u8> = orig.into();
        let back: Rgb<u8> = ext.into();
        assert_eq!(orig, back);
    }

    #[test]
    fn rgb_image_roundtrip() {
        let mut src = image::RgbImage::new(3, 2);
        src.put_pixel(0, 0, image::Rgb([1, 2, 3]));
        src.put_pixel(2, 1, image::Rgb([4, 5, 6]));

        let oxislam: Image<Rgb<u8>> = src.clone().into();
        assert_eq!(oxislam.width(), 3);
        assert_eq!(oxislam.height(), 2);
        assert_eq!(*oxislam.get(0, 0), Rgb::new(1, 2, 3));
        assert_eq!(*oxislam.get(2, 1), Rgb::new(4, 5, 6));

        let back: image::RgbImage = (&oxislam).into();
        assert_eq!(src, back);
    }

    #[test]
    fn gray_image_roundtrip() {
        let src = image::GrayImage::from_raw(2, 2, vec![10, 20, 30, 40]).unwrap();

        let oxislam: Image<Gray<u8>> = src.clone().into();
        assert_eq!(oxislam.width(), 2);
        assert_eq!(oxislam.height(), 2);
        assert_eq!(oxislam.get(0, 0).value, 10);
        assert_eq!(oxislam.get(1, 1).value, 40);

        let back: image::GrayImage = (&oxislam).into();
        assert_eq!(src, back);
    }
}
