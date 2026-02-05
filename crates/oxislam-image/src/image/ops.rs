#[cfg(not(feature = "rayon"))]
use std::iter::zip;
use std::ops::{Add, Mul, Sub};

#[cfg(feature = "rayon")]
use rayon::prelude::*;

use super::{Image, ImageView};
use crate::pixel::Gray;

#[cfg(feature = "rayon")]
pub fn map<P, Q, F>(img: &ImageView<P>, f: F) -> Image<Q>
where
    F: Fn(&P) -> Q + Sync + Send,
    P: Sync,
    Q: Send,
{
    let w = img.width();
    let h = img.height();
    let f = &f;

    let data: Vec<Q> =
        (0..h).into_par_iter().flat_map_iter(|y| (0..w).map(move |x| f(img.get(x, y)))).collect();

    Image::new(w, h, w, data)
}

#[cfg(not(feature = "rayon"))]
pub fn map<P, Q, F>(img: &ImageView<P>, f: F) -> Image<Q>
where
    F: Fn(&P) -> Q,
{
    let w = img.width();
    let h = img.height();
    let data: Vec<Q> = img.pixels().map(f).collect();
    Image::new(w, h, w, data)
}

#[cfg(feature = "rayon")]
pub fn map2<P, Q, R, F>(a: &ImageView<P>, b: &ImageView<Q>, f: F) -> Image<R>
where
    F: Fn(&P, &Q) -> R + Sync + Send,
    P: Sync,
    Q: Sync,
    R: Send,
{
    assert_eq!(a.width(), b.width());
    assert_eq!(a.height(), b.height());

    let w = a.width();
    let h = a.height();
    let f = &f;

    let data: Vec<R> = (0..h)
        .into_par_iter()
        .flat_map_iter(|y| (0..w).map(move |x| f(a.get(x, y), b.get(x, y))))
        .collect();

    Image::new(w, h, w, data)
}

#[cfg(not(feature = "rayon"))]
pub fn map2<P, Q, R, F>(a: &ImageView<P>, b: &ImageView<Q>, f: F) -> Image<R>
where
    F: Fn(&P, &Q) -> R,
{
    assert_eq!(a.width(), b.width());
    assert_eq!(a.height(), b.height());

    let w = a.width();
    let h = a.height();
    let data: Vec<R> = zip(a.pixels(), b.pixels()).map(|(x, y)| f(x, y)).collect();
    Image::new(w, h, w, data)
}

impl Mul for &Image<Gray<f32>> {
    type Output = Image<Gray<f32>>;

    fn mul(self, rhs: Self) -> Self::Output {
        map2(&self.view(), &rhs.view(), |a, b| Gray::new(a.value * b.value))
    }
}

impl Add for &Image<Gray<f32>> {
    type Output = Image<Gray<f32>>;

    fn add(self, rhs: Self) -> Self::Output {
        map2(&self.view(), &rhs.view(), |a, b| Gray::new(a.value + b.value))
    }
}

impl Sub for &Image<Gray<f32>> {
    type Output = Image<Gray<f32>>;

    fn sub(self, rhs: Self) -> Self::Output {
        map2(&self.view(), &rhs.view(), |a, b| Gray::new(a.value - b.value))
    }
}
