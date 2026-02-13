use super::kernel::{Kernel, apply_separable_kernel};
use crate::image::{Image, ImageView};
use crate::pixel::Gray;

const GAUSSIAN_3X3_1D: [f32; 3] = [1.0 / 4.0, 2.0 / 4.0, 1.0 / 4.0];
const GAUSSIAN_5X5_1D: [f32; 5] = [1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0];
const GAUSSIAN_7X7_1D: [f32; 7] =
    [1.0 / 64.0, 6.0 / 64.0, 15.0 / 64.0, 20.0 / 64.0, 15.0 / 64.0, 6.0 / 64.0, 1.0 / 64.0];

// Keep the 2D 3x3 constant for external use (e.g. Harris corner detection).
#[rustfmt::skip]
pub const GAUSSIAN_3X3: Kernel<3> = [
    [1.0/16.0, 2.0/16.0, 1.0/16.0],
    [2.0/16.0, 4.0/16.0, 2.0/16.0],
    [1.0/16.0, 2.0/16.0, 1.0/16.0],
];

mod sealed {
    use super::*;

    pub trait GaussianKernel {
        fn apply(image: &ImageView<Gray<f32>>) -> Image<Gray<f32>>;
    }

    pub enum Size<const N: usize> {}

    impl GaussianKernel for Size<3> {
        fn apply(image: &ImageView<Gray<f32>>) -> Image<Gray<f32>> {
            apply_separable_kernel(image, &GAUSSIAN_3X3_1D)
        }
    }

    impl GaussianKernel for Size<5> {
        fn apply(image: &ImageView<Gray<f32>>) -> Image<Gray<f32>> {
            apply_separable_kernel(image, &GAUSSIAN_5X5_1D)
        }
    }

    impl GaussianKernel for Size<7> {
        fn apply(image: &ImageView<Gray<f32>>) -> Image<Gray<f32>> {
            apply_separable_kernel(image, &GAUSSIAN_7X7_1D)
        }
    }
}

/// Apply a Gaussian blur with a kernel of size `N` (3, 5, or 7).
pub fn gaussian<const N: usize>(image: &ImageView<Gray<f32>>) -> Image<Gray<f32>>
where
    sealed::Size<N>: sealed::GaussianKernel,
{
    let _span = crate::trace::span!("gaussian");
    use sealed::GaussianKernel;
    sealed::Size::<N>::apply(image)
}

#[cfg(test)]
mod tests {
    use super::super::kernel::apply_kernel;
    use super::*;

    fn outer_product<const N: usize>(v: &[f32; N]) -> Kernel<N> {
        let mut k = [[0.0; N]; N];
        for (r, row) in k.iter_mut().enumerate() {
            for (c, cell) in row.iter_mut().enumerate() {
                *cell = v[r] * v[c];
            }
        }
        k
    }

    fn assert_images_equal(a: &Image<Gray<f32>>, b: &Image<Gray<f32>>) {
        assert_eq!(a.width(), b.width());
        assert_eq!(a.height(), b.height());
        for y in 0..a.height() {
            for x in 0..a.width() {
                let av = a.get(x, y).value;
                let bv = b.get(x, y).value;
                assert!((av - bv).abs() < 1e-6, "Mismatch at ({x}, {y}): {av} vs {bv}");
            }
        }
    }

    fn test_image() -> Image<Gray<f32>> {
        let w = 32;
        let h = 24;
        let data: Vec<Gray<f32>> = (0..w * h).map(|i| Gray::new((i as f32 * 0.37).sin())).collect();
        Image::new(w, h, w, data)
    }

    #[test]
    fn kernel_1d_outer_product_matches_2d_constant() {
        let computed = outer_product(&GAUSSIAN_3X3_1D);
        for r in 0..3 {
            for c in 0..3 {
                assert!(
                    (computed[r][c] - GAUSSIAN_3X3[r][c]).abs() < 1e-7,
                    "3x3 constant mismatch at [{r}][{c}]: {} vs {}",
                    computed[r][c],
                    GAUSSIAN_3X3[r][c],
                );
            }
        }
    }

    #[test]
    fn separable_matches_2d_gaussian_3x3() {
        let img = test_image();
        let separable = gaussian::<3>(&img.view());
        let full_2d = apply_kernel(&img.view(), &GAUSSIAN_3X3);
        assert_images_equal(&separable, &full_2d);
    }

    #[test]
    fn separable_matches_2d_gaussian_5x5() {
        let img = test_image();
        let separable = gaussian::<5>(&img.view());
        let full_2d = apply_kernel(&img.view(), &outer_product(&GAUSSIAN_5X5_1D));
        assert_images_equal(&separable, &full_2d);
    }

    #[test]
    fn separable_matches_2d_gaussian_7x7() {
        let img = test_image();
        let separable = gaussian::<7>(&img.view());
        let full_2d = apply_kernel(&img.view(), &outer_product(&GAUSSIAN_7X7_1D));
        assert_images_equal(&separable, &full_2d);
    }
}
