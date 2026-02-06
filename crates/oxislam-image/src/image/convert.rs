#[cfg(feature = "rayon")]
use rayon::prelude::*;

use super::types::{Image, ImageView};
use crate::pixel::{Gray, Rgb};

pub trait ConvertTo<T> {
    fn to(&self) -> T;
}

#[cfg(feature = "rayon")]
#[inline]
fn convert_rows<S, D, F>(width: usize, height: usize, stride: usize, data: &[S], f: F) -> Vec<D>
where
    S: Sync,
    D: Send,
    F: Fn(&S) -> D + Sync,
{
    (0..height)
        .into_par_iter()
        .flat_map(|y| {
            let row = y * stride;
            (0..width).map(|x| f(&data[row + x])).collect::<Vec<_>>()
        })
        .collect()
}

#[cfg(not(feature = "rayon"))]
#[inline]
fn convert_rows<S, D, F>(width: usize, height: usize, stride: usize, data: &[S], f: F) -> Vec<D>
where
    F: Fn(&S) -> D,
{
    let mut result = Vec::with_capacity(width * height);
    for y in 0..height {
        let row = y * stride;
        for x in 0..width {
            result.push(f(&data[row + x]));
        }
    }
    result
}

// Gray<u8> -> Gray<f32>
impl ConvertTo<Image<Gray<f32>>> for ImageView<'_, Gray<u8>> {
    fn to(&self) -> Image<Gray<f32>> {
        let (width, height, stride, data) =
            (self.width(), self.height(), self.stride(), self.data());
        let result =
            convert_rows(width, height, stride, data, |px| Gray::new(px.value as f32 / 255.0));
        Image::new(width, height, width, result)
    }
}

// Gray<f32> -> Gray<u8>
impl ConvertTo<Image<Gray<u8>>> for ImageView<'_, Gray<f32>> {
    fn to(&self) -> Image<Gray<u8>> {
        let (width, height, stride, data) =
            (self.width(), self.height(), self.stride(), self.data());
        let result = convert_rows(width, height, stride, data, |px| {
            Gray::new((px.value * 255.0).clamp(0.0, 255.0) as u8)
        });
        Image::new(width, height, width, result)
    }
}

// Rgb<u8> -> Rgb<f32>
impl ConvertTo<Image<Rgb<f32>>> for ImageView<'_, Rgb<u8>> {
    fn to(&self) -> Image<Rgb<f32>> {
        let (width, height, stride, data) =
            (self.width(), self.height(), self.stride(), self.data());
        let result = convert_rows(width, height, stride, data, |px| {
            Rgb::new(px.r as f32 / 255.0, px.g as f32 / 255.0, px.b as f32 / 255.0)
        });
        Image::new(width, height, width, result)
    }
}

// Rgb<f32> -> Rgb<u8>
impl ConvertTo<Image<Rgb<u8>>> for ImageView<'_, Rgb<f32>> {
    fn to(&self) -> Image<Rgb<u8>> {
        let (width, height, stride, data) =
            (self.width(), self.height(), self.stride(), self.data());
        let result = convert_rows(width, height, stride, data, |px| {
            Rgb::new(
                (px.r * 255.0).clamp(0.0, 255.0) as u8,
                (px.g * 255.0).clamp(0.0, 255.0) as u8,
                (px.b * 255.0).clamp(0.0, 255.0) as u8,
            )
        });
        Image::new(width, height, width, result)
    }
}

// Rgb<u8> -> Gray<f32> (direct single-pass)
impl ConvertTo<Image<Gray<f32>>> for ImageView<'_, Rgb<u8>> {
    fn to(&self) -> Image<Gray<f32>> {
        let (width, height, stride, data) =
            (self.width(), self.height(), self.stride(), self.data());
        let result = convert_rows(width, height, stride, data, |px| {
            let gray = (px.r as f32 * 0.299 + px.g as f32 * 0.587 + px.b as f32 * 0.114) / 255.0;
            Gray::new(gray)
        });
        Image::new(width, height, width, result)
    }
}

// Rgb<f32> -> Gray<f32>
impl ConvertTo<Image<Gray<f32>>> for ImageView<'_, Rgb<f32>> {
    fn to(&self) -> Image<Gray<f32>> {
        let (width, height, stride, data) =
            (self.width(), self.height(), self.stride(), self.data());
        let result = convert_rows(width, height, stride, data, |px| {
            Gray::new(px.r * 0.299 + px.g * 0.587 + px.b * 0.114)
        });
        Image::new(width, height, width, result)
    }
}

// Identity (clone with compact output)
#[cfg(feature = "rayon")]
impl<P: Clone + Send + Sync> ConvertTo<Image<P>> for ImageView<'_, P> {
    fn to(&self) -> Image<P> {
        let (w, h, s, data) = (self.width(), self.height(), self.stride(), self.data());
        Image::new(w, h, w, convert_rows(w, h, s, data, Clone::clone))
    }
}

#[cfg(not(feature = "rayon"))]
impl<P: Clone> ConvertTo<Image<P>> for ImageView<'_, P> {
    fn to(&self) -> Image<P> {
        let (w, h, s, data) = (self.width(), self.height(), self.stride(), self.data());
        Image::new(w, h, w, convert_rows(w, h, s, data, Clone::clone))
    }
}
