use std::ops::Range;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

// Conditional Send/Sync bounds: require Send/Sync only when rayon is enabled.

#[cfg(feature = "rayon")]
pub trait MaybeSend: Send {}
#[cfg(feature = "rayon")]
impl<T: Send> MaybeSend for T {}

#[cfg(not(feature = "rayon"))]
pub trait MaybeSend {}
#[cfg(not(feature = "rayon"))]
impl<T> MaybeSend for T {}

#[cfg(feature = "rayon")]
pub trait MaybeSync: Sync {}
#[cfg(feature = "rayon")]
impl<T: Sync> MaybeSync for T {}

#[cfg(not(feature = "rayon"))]
pub trait MaybeSync {}
#[cfg(not(feature = "rayon"))]
impl<T> MaybeSync for T {}

/// Collect pixels in row-major order, parallelizing across rows when rayon is enabled.
pub fn par_row_collect<T: MaybeSend, F>(width: usize, height: usize, f: F) -> Vec<T>
where
    F: Fn(usize, usize) -> T + MaybeSync,
{
    #[cfg(feature = "rayon")]
    {
        let f = &f;
        (0..height).into_par_iter().flat_map_iter(|y| (0..width).map(move |x| f(x, y))).collect()
    }
    #[cfg(not(feature = "rayon"))]
    {
        let mut data = Vec::with_capacity(width * height);
        for y in 0..height {
            for x in 0..width {
                data.push(f(x, y));
            }
        }
        data
    }
}

/// Flat-map over a range, parallelizing when rayon is enabled.
pub fn par_flat_map<T: MaybeSend, F, I>(range: Range<usize>, f: F) -> Vec<T>
where
    F: Fn(usize) -> I + MaybeSync,
    I: IntoIterator<Item = T>,
{
    #[cfg(feature = "rayon")]
    {
        let f = &f;
        range.into_par_iter().flat_map_iter(|i| f(i)).collect()
    }
    #[cfg(not(feature = "rayon"))]
    {
        range.flat_map(f).collect()
    }
}

/// Filter-map over a Vec, parallelizing when rayon is enabled.
pub fn par_filter_map<T: MaybeSend, U: MaybeSend, F>(items: Vec<T>, f: F) -> Vec<U>
where
    F: Fn(T) -> Option<U> + MaybeSync,
{
    #[cfg(feature = "rayon")]
    {
        let f = &f;
        items.into_par_iter().filter_map(|item| f(item)).collect()
    }
    #[cfg(not(feature = "rayon"))]
    {
        items.into_iter().filter_map(f).collect()
    }
}
