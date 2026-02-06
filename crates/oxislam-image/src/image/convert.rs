use super::types::{Image, ImageView};
use crate::parallel::{MaybeSend, MaybeSync, par_row_collect};
use crate::pixel::{Gray, Rgb};

pub trait ConvertTo<T> {
    fn to(&self) -> T;
}

// Gray<u8> -> Gray<f32>
impl ConvertTo<Image<Gray<f32>>> for ImageView<'_, Gray<u8>> {
    fn to(&self) -> Image<Gray<f32>> {
        let (w, h, s, d) = (self.width(), self.height(), self.stride(), self.data());
        Image::new(w, h, w, par_row_collect(w, h, |x, y| Gray::new(d[y * s + x].value as f32 / 255.0)))
    }
}

// Gray<f32> -> Gray<u8>
impl ConvertTo<Image<Gray<u8>>> for ImageView<'_, Gray<f32>> {
    fn to(&self) -> Image<Gray<u8>> {
        let (w, h, s, d) = (self.width(), self.height(), self.stride(), self.data());
        Image::new(w, h, w, par_row_collect(w, h, |x, y| {
            Gray::new((d[y * s + x].value * 255.0).clamp(0.0, 255.0) as u8)
        }))
    }
}

// Rgb<u8> -> Rgb<f32>
impl ConvertTo<Image<Rgb<f32>>> for ImageView<'_, Rgb<u8>> {
    fn to(&self) -> Image<Rgb<f32>> {
        let (w, h, s, d) = (self.width(), self.height(), self.stride(), self.data());
        Image::new(w, h, w, par_row_collect(w, h, |x, y| {
            let px = &d[y * s + x];
            Rgb::new(px.r as f32 / 255.0, px.g as f32 / 255.0, px.b as f32 / 255.0)
        }))
    }
}

// Rgb<f32> -> Rgb<u8>
impl ConvertTo<Image<Rgb<u8>>> for ImageView<'_, Rgb<f32>> {
    fn to(&self) -> Image<Rgb<u8>> {
        let (w, h, s, d) = (self.width(), self.height(), self.stride(), self.data());
        Image::new(w, h, w, par_row_collect(w, h, |x, y| {
            let px = &d[y * s + x];
            Rgb::new(
                (px.r * 255.0).clamp(0.0, 255.0) as u8,
                (px.g * 255.0).clamp(0.0, 255.0) as u8,
                (px.b * 255.0).clamp(0.0, 255.0) as u8,
            )
        }))
    }
}

// Rgb<u8> -> Gray<f32> (direct single-pass)
impl ConvertTo<Image<Gray<f32>>> for ImageView<'_, Rgb<u8>> {
    fn to(&self) -> Image<Gray<f32>> {
        let (w, h, s, d) = (self.width(), self.height(), self.stride(), self.data());
        Image::new(w, h, w, par_row_collect(w, h, |x, y| {
            let px = &d[y * s + x];
            Gray::new((px.r as f32 * 0.299 + px.g as f32 * 0.587 + px.b as f32 * 0.114) / 255.0)
        }))
    }
}

// Rgb<f32> -> Gray<f32>
impl ConvertTo<Image<Gray<f32>>> for ImageView<'_, Rgb<f32>> {
    fn to(&self) -> Image<Gray<f32>> {
        let (w, h, s, d) = (self.width(), self.height(), self.stride(), self.data());
        Image::new(w, h, w, par_row_collect(w, h, |x, y| {
            let px = &d[y * s + x];
            Gray::new(px.r * 0.299 + px.g * 0.587 + px.b * 0.114)
        }))
    }
}

// Identity (clone with compact output)
impl<P: Clone + MaybeSend + MaybeSync> ConvertTo<Image<P>> for ImageView<'_, P> {
    fn to(&self) -> Image<P> {
        let (w, h, s, d) = (self.width(), self.height(), self.stride(), self.data());
        Image::new(w, h, w, par_row_collect(w, h, |x, y| d[y * s + x].clone()))
    }
}
