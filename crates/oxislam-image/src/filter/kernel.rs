use crate::image::{Image, ImageView};
use crate::parallel::par_row_collect;
use crate::pixel::Gray;

pub type Kernel<const N: usize> = [[f32; N]; N];

fn compute_pixel<const N: usize>(
    image: &ImageView<Gray<f32>>,
    kernel: &Kernel<N>,
    cx: usize,
    cy: usize,
) -> Gray<f32> {
    let half = (N / 2) as isize;
    let w = image.width();
    let h = image.height();
    let mut sum = 0.0;
    for (ky, kernel_row) in kernel.iter().enumerate() {
        let iy = cy as isize + ky as isize - half;
        if iy < 0 || iy >= h as isize {
            continue;
        }
        for (kx, kernel_val) in kernel_row.iter().enumerate() {
            let ix = cx as isize + kx as isize - half;
            if ix < 0 || ix >= w as isize {
                continue;
            }
            sum += image.get(ix as usize, iy as usize).value * kernel_val;
        }
    }
    Gray::new(sum)
}

/// Apply a kernel to an image, producing an output of the same dimensions.
///
/// Out-of-bounds pixels are treated as zero.
pub fn apply_kernel<const N: usize>(
    image: &ImageView<Gray<f32>>,
    kernel: &Kernel<N>,
) -> Image<Gray<f32>> {
    let w = image.width();
    let h = image.height();
    assert!(w >= N && h >= N, "Image must be at least {N}x{N}");

    let data = par_row_collect(w, h, |x, y| compute_pixel(image, kernel, x, y));

    Image::new(w, h, w, data)
}

fn apply_1d_kernel<const N: usize>(
    image: &ImageView<Gray<f32>>,
    kernel_1d: &[f32; N],
    horizontal: bool,
) -> Image<Gray<f32>> {
    let w = image.width();
    let h = image.height();
    let half = (N / 2) as isize;
    let limit = if horizontal { w } else { h } as isize;

    let data = par_row_collect(w, h, |x, y| {
        let center = if horizontal { x } else { y } as isize;
        let mut sum = 0.0;
        for (k, &kv) in kernel_1d.iter().enumerate() {
            let i = center + k as isize - half;
            if i >= 0 && i < limit {
                let (px, py) = if horizontal { (i as usize, y) } else { (x, i as usize) };
                sum += image.get(px, py).value * kv;
            }
        }
        Gray::new(sum)
    });

    Image::new(w, h, w, data)
}

/// Apply a separable 1D kernel as two passes (horizontal then vertical).
pub(super) fn apply_separable_kernel<const N: usize>(
    image: &ImageView<Gray<f32>>,
    kernel_1d: &[f32; N],
) -> Image<Gray<f32>> {
    let w = image.width();
    let h = image.height();

    assert!(w >= N && h >= N, "Image must be at least {N}x{N}");

    let intermediate = apply_1d_kernel(image, kernel_1d, true);
    apply_1d_kernel(&intermediate.view(), kernel_1d, false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn convolution_identity_kernel() {
        let data: Vec<Gray<f32>> = (1..=16).map(|v| Gray::new(v as f32)).collect();
        let img = Image::new(4, 4, 4, data);

        #[rustfmt::skip]
        let identity: Kernel<3> = [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ];

        let out = apply_kernel(&img.view(), &identity);

        assert_eq!(out.width(), 4);
        assert_eq!(out.height(), 4);
        assert_eq!(out.get(0, 0).value, 1.0);
        assert_eq!(out.get(1, 1).value, 6.0);
        assert_eq!(out.get(2, 1).value, 7.0);
        assert_eq!(out.get(1, 2).value, 10.0);
        assert_eq!(out.get(2, 2).value, 11.0);
        assert_eq!(out.get(3, 3).value, 16.0);
    }
}
