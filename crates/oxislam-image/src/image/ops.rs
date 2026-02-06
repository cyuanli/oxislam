use std::ops::{Add, Mul, Sub};

use super::{Image, ImageView};
use crate::parallel::{MaybeSend, MaybeSync, par_row_collect};
use crate::pixel::Gray;

pub fn map<P: MaybeSync, Q: MaybeSend, F>(img: &ImageView<P>, f: F) -> Image<Q>
where
    F: Fn(&P) -> Q + MaybeSync,
{
    let w = img.width();
    let h = img.height();
    let f = &f;
    let data = par_row_collect(w, h, |x, y| f(img.get(x, y)));
    Image::new(w, h, w, data)
}

pub fn map2<P: MaybeSync, Q: MaybeSync, R: MaybeSend, F>(
    a: &ImageView<P>,
    b: &ImageView<Q>,
    f: F,
) -> Image<R>
where
    F: Fn(&P, &Q) -> R + MaybeSync,
{
    assert_eq!(a.width(), b.width());
    assert_eq!(a.height(), b.height());

    let w = a.width();
    let h = a.height();
    let f = &f;
    let data = par_row_collect(w, h, |x, y| f(a.get(x, y), b.get(x, y)));
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
